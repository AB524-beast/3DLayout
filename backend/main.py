import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import cv2
import math
import json
import os
import logging
import time

from tracing import setup_tracing
from database import save_project, save_rooms
from model_inference import segment_rooms_ml, is_model_available

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s\t%(name)s\t%(asctime)s\t%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("3d-layout")

app = FastAPI(title="Orthogonal Blueprint Spatial Modeler")
setup_tracing(app)

_raw_cors = os.environ.get("CORS_ORIGINS", "").strip()
_cors_origins = [o.strip() for o in _raw_cors.split(",") if o.strip()] if _raw_cors else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("CORS origins: %s", _cors_origins)


@app.get("/")
async def root():
    return {"status": "running", "service": "3d-layout-backend"}


@app.get("/health")
async def health():
    return {"status": "ok", "model_available": is_model_available()}


class RoomSpecification(BaseModel):
    name: str
    floorAssigned: int
    isOpenSpace: bool
    roomSqFt: float


class ProceduralGenerationPayload(BaseModel):
    total_sq_ft: float
    total_floors: int
    rooms: List[RoomSpecification]


class JSONRoom(BaseModel):
    label: str
    centerX: float
    centerY: float
    width: float
    height: float
    polygon_points: List[List[float]]


class SaveLayoutRequest(BaseModel):
    name: str = "Untitled Layout"
    image_url: Optional[str] = None
    total_floors: int = 1
    rooms: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def polygon_to_walls(points_m: List[List[float]]) -> List[Dict[str, float]]:
    walls = []
    n = len(points_m)
    for i in range(n):
        x1, y1 = points_m[i]
        x2, y2 = points_m[(i + 1) % n]
        walls.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return walls


def _polygon_area(pts):
    n = len(pts)
    area = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _bbox_from_polygon(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_intersection_area(bb1, bb2):
    x_overlap = max(0, min(bb1[2], bb2[2]) - max(bb1[0], bb2[0]))
    y_overlap = max(0, min(bb1[3], bb2[3]) - max(bb1[1], bb2[1]))
    return x_overlap * y_overlap


def _snap_orthogonal_strict(pts: np.ndarray, tol_deg: float = 10.0) -> np.ndarray:
    n = len(pts)
    if n < 4:
        return pts.copy().astype(np.int32)
    snapped = pts.copy().astype(np.float64)

    for _ in range(3):
        changed = False
        for i in range(n):
            p1 = snapped[i]
            p2 = snapped[(i + 1) % n]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.hypot(dx, dy)
            if length < 1.0:
                continue
            angle = abs(math.degrees(math.atan2(abs(dy), abs(dx))))
            if angle < tol_deg:
                new_y = p1[1]
                if abs(p2[1] - new_y) > 0.5:
                    snapped[(i + 1) % n][1] = new_y
                    changed = True
            elif angle > 90.0 - tol_deg:
                new_x = p1[0]
                if abs(p2[0] - new_x) > 0.5:
                    snapped[(i + 1) % n][0] = new_x
                    changed = True
        if not changed:
            break

    for i in range(n):
        p1 = snapped[i]
        p2 = snapped[(i + 1) % n]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = abs(math.degrees(math.atan2(abs(dy), abs(dx))))
        if angle < tol_deg:
            snapped[(i + 1) % n][1] = p1[1]
        elif angle > 90.0 - tol_deg:
            snapped[(i + 1) % n][0] = p1[0]

    return snapped.astype(np.int32)


def _collapse_short_edges(pts: np.ndarray, min_len: float) -> np.ndarray:
    pts = pts.astype(np.float64).tolist()
    changed = True
    while changed and len(pts) > 4:
        changed = False
        for i in range(len(pts)):
            p1 = pts[i]
            p2 = pts[(i + 1) % len(pts)]
            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            if dist < min_len:
                del pts[(i + 1) % len(pts)]
                changed = True
                break
    return np.array(pts, dtype=np.float64)


def _snap_orthogonal_alternating(pts: np.ndarray) -> np.ndarray:
    n = len(pts)
    snapped = pts.copy().astype(np.float64)
    last_orientation = None
    for i in range(n):
        p1 = snapped[i]
        p2 = snapped[(i + 1) % n]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        if max(abs(dx), abs(dy)) < 0.5:
            continue
        prefer_horizontal = abs(dx) >= abs(dy)
        if last_orientation == "H":
            prefer_horizontal = False
        elif last_orientation == "V":
            prefer_horizontal = True
        if prefer_horizontal:
            snapped[(i + 1) % n][1] = p1[1]
            last_orientation = "H"
        else:
            snapped[(i + 1) % n][0] = p1[0]
            last_orientation = "V"
    return snapped.astype(np.int32)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _preprocess_adaptive(gray: np.ndarray) -> Dict[str, np.ndarray]:
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(bilateral)

    adaptive_thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 25, 12
    )

    return {
        "gray": gray,
        "bilateral": bilateral,
        "enhanced": enhanced,
        "adaptive_thresh": adaptive_thresh,
    }


# ---------------------------------------------------------------------------
# Wall detection - multi-strategy
# ---------------------------------------------------------------------------

def _detect_walls_canny_hough(gray: np.ndarray, w: int, h: int,
                               canny_low: int = 30, canny_high: int = 100,
                               hough_thresh: int = 25) -> np.ndarray:
    edges = cv2.Canny(gray, canny_low, canny_high, apertureSize=3)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel_dilate, iterations=1)

    total_px = w * h
    min_line_len = max(15, int(math.sqrt(total_px) * 0.02))
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=hough_thresh, minLineLength=min_line_len, maxLineGap=20,
    )

    wall_mask = np.zeros((h, w), dtype=np.uint8)
    if lines is None:
        return wall_mask

    ANGLE_TOL_DEG = 10.0
    for line in lines:
        coords = np.asarray(line).reshape(-1)
        if coords.size < 4:
            continue
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        dx, dy = x2 - x1, y2 - y1
        angle = math.degrees(math.atan2(abs(dy), abs(dx)))
        is_horizontal = angle < ANGLE_TOL_DEG
        is_vertical = angle > 90 - ANGLE_TOL_DEG
        if not (is_horizontal or is_vertical):
            continue
        if is_horizontal:
            y2 = y1
        else:
            x2 = x1
        cv2.line(wall_mask, (x1, y1), (x2, y2), 255, thickness=4)

    return wall_mask


def _detect_walls_adaptive(adaptive_thresh: np.ndarray, w: int, h: int) -> np.ndarray:
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    total_px = w * h
    min_area = total_px * 0.001
    wall_mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            wall_mask[labels == i] = 255

    return wall_mask


def _detect_walls_morphological_gradient(gray: np.ndarray, w: int, h: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    _, wall_mask = cv2.threshold(grad, 30, 255, cv2.THRESH_BINARY)

    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, close_k, iterations=1)

    return wall_mask


def _combine_wall_masks(masks: list) -> np.ndarray:
    if not masks:
        return np.zeros_like(masks[0]) if masks else np.zeros((1, 1), dtype=np.uint8)
    combined = masks[0]
    for m in masks[1:]:
        combined = cv2.bitwise_or(combined, m)
    return combined


def _build_wall_mask_full(gray: np.ndarray, enhanced: np.ndarray,
                          adaptive_thresh: np.ndarray,
                          w: int, h: int, params: dict) -> np.ndarray:
    canny_low = params.get("canny_low", 30)
    canny_high = params.get("canny_high", 100)
    hough_thresh = params.get("hough_thresh", 25)

    hough_mask = _detect_walls_canny_hough(enhanced, w, h, canny_low, canny_high, hough_thresh)
    adaptive_mask = _detect_walls_adaptive(adaptive_thresh, w, h)
    gradient_mask = _detect_walls_morphological_gradient(enhanced, w, h)

    combined = _combine_wall_masks([hough_mask, adaptive_mask, gradient_mask])

    total_px = w * h
    if cv2.countNonZero(combined) < total_px * 0.005:
        _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
        min_wall_area = total_px * 0.002
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_wall_area:
                combined[labels == i] = 255

    close_size = params.get("close_size", 9)
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_k, iterations=2)

    return combined


# ---------------------------------------------------------------------------
# Room extraction from wall mask
# ---------------------------------------------------------------------------

def _remove_border_region(rooms_bin: np.ndarray) -> np.ndarray:
    h, w = rooms_bin.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    filled = rooms_bin.copy()

    seeds = [
        (0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1),
        (0, w // 2), (h - 1, w // 2), (h // 2, 0), (h // 2, w - 1),
        (0, w // 4), (0, 3 * w // 4),
        (h - 1, w // 4), (h - 1, 3 * w // 4),
        (h // 4, 0), (3 * h // 4, 0),
        (h // 4, w - 1), (3 * h // 4, w - 1),
    ]
    for (sy, sx) in seeds:
        sy = max(0, min(sy, h - 1))
        sx = max(0, min(sx, w - 1))
        if filled[sy, sx] == 255:
            cv2.floodFill(filled, flood_mask, (sx, sy), 128)

    result = rooms_bin.copy()
    result[filled == 128] = 0
    return result


def _separate_rooms_watershed(rooms_bin: np.ndarray, w: int, h: int,
                               dist_thresh: float = 0.35) -> np.ndarray:
    smooth_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    rooms_bin = cv2.morphologyEx(rooms_bin, cv2.MORPH_CLOSE, smooth_k, iterations=1)
    rooms_bin = cv2.morphologyEx(rooms_bin, cv2.MORPH_OPEN, smooth_k, iterations=1)

    dist = cv2.distanceTransform(rooms_bin, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    _, sure_fg = cv2.threshold(dist_norm, dist_thresh, 1.0, cv2.THRESH_BINARY)
    sure_fg = (sure_fg * 255).astype(np.uint8)

    sure_bg = cv2.dilate(rooms_bin, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=3)

    unknown = cv2.subtract(sure_bg, sure_fg)

    num_markers, markers = cv2.connectedComponents(sure_fg)
    if num_markers <= 1:
        return rooms_bin

    markers = markers + 1
    markers[unknown == 255] = 0

    img_for_ws = cv2.cvtColor(rooms_bin, cv2.COLOR_GRAY2BGR)
    cv2.watershed(img_for_ws, markers)

    result = np.zeros_like(rooms_bin)
    for label in range(2, markers.max() + 1):
        result[markers == label] = 255

    return result


def _rooms_from_wall_mask(walls: np.ndarray, w: int, h: int,
                          px_to_meter: float, min_room_area_px: float,
                          dist_thresh: float) -> List[Dict[str, Any]]:
    rooms_bin = cv2.bitwise_not(walls)
    rooms_bin = _remove_border_region(rooms_bin)

    separate_mask = _separate_rooms_watershed(rooms_bin, w, h, dist_thresh)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        separate_mask, connectivity=8
    )

    rooms = []
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_room_area_px:
            continue

        comp_mask = np.uint8(labels == label_id) * 255

        comp_mask = cv2.morphologyEx(comp_mask, cv2.MORPH_CLOSE,
                                      cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                                      iterations=1)

        cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        cnt_area = cv2.contourArea(cnt)
        if cnt_area < min_room_area_px:
            continue

        rect = cv2.minAreaRect(cnt)
        (_, _), (rect_w, rect_h), _ = rect
        rect_area = rect_w * rect_h
        fill_ratio = cnt_area / rect_area if rect_area > 0 else 0

        if fill_ratio > 0.72:
            box = cv2.boxPoints(rect)
            approx = box.reshape(-1, 1, 2).astype(np.int32)
        else:
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            escalate = 0.02
            while len(approx) > 8 and escalate < 0.08:
                escalate += 0.01
                approx = cv2.approxPolyDP(cnt, escalate * cv2.arcLength(cnt, True), True)
            if len(approx) < 4:
                continue
            approx_pts = approx.reshape(-1, 2)
            perim = cv2.arcLength(cnt, True)
            min_edge_len = max(8.0, perim * 0.02)
            approx_pts = _collapse_short_edges(approx_pts, min_edge_len)
            if len(approx_pts) < 4:
                continue
            snapped = _snap_orthogonal_alternating(approx_pts)
            approx = snapped.reshape(-1, 1, 2)

        xs_px = [int(pt[0][0]) for pt in approx]
        ys_px = [int(pt[0][1]) for pt in approx]

        edge_margin = max(3, int(min(w, h) * 0.005))
        if (min(xs_px) <= edge_margin or min(ys_px) <= edge_margin or
                max(xs_px) >= w - edge_margin or max(ys_px) >= h - edge_margin):
            continue

        raw_pts = [pt[0] for pt in approx]
        pts_m = [[(px - w / 2.0) / px_to_meter, (py - h / 2.0) / px_to_meter] for px, py in raw_pts]
        xs = [p[0] for p in pts_m]
        ys = [p[1] for p in pts_m]
        bb_w = max(xs) - min(xs)
        bb_h = max(ys) - min(ys)

        if bb_w < 0.5 or bb_h < 0.5:
            continue
        if bb_w > 25 or bb_h > 25:
            continue

        min_side, max_side = min(bb_w, bb_h), max(bb_w, bb_h)
        if max_side > 0 and min_side / max_side < 0.10:
            continue

        img_w_m, img_h_m = w / px_to_meter, h / px_to_meter
        if bb_w > img_w_m * 0.90 and bb_h > img_h_m * 0.90:
            continue

        polygon_m = pts_m
        area_m2 = round(_polygon_area(polygon_m), 2)
        if area_m2 < 0.3:
            continue

        rooms.append({
            "label": f"Room {len(rooms) + 1}",
            "dimensions": f"{bb_w:.1f}m x {bb_h:.1f}m",
            "centerX": sum(xs) / len(xs),
            "centerY": sum(ys) / len(ys),
            "elevationZ": 0.0,
            "isOpenSpace": False,
            "walls": polygon_to_walls(polygon_m),
            "area": area_m2,
            "_px_area": float(cnt_area),
        })

    rooms.sort(key=lambda r: r.get("_px_area", 0), reverse=True)
    for i, r in enumerate(rooms):
        r["label"] = f"Room {i + 1}"
        r.pop("_px_area", None)

    return rooms


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_result(rooms: List[Dict[str, Any]], w: int, h: int, px_to_meter: float) -> float:
    if not rooms:
        return -1.0

    total_area = sum(r["area"] for r in rooms)
    image_area = (w / px_to_meter) * (h / px_to_meter)
    coverage = total_area / image_area if image_area > 0 else 0
    if coverage < 0.10 or coverage > 1.10:
        return -1.0

    bboxes = []
    for r in rooms:
        xs = [p[0] for p in r.get("walls", []) if "x1" in p]
        ys = [p[1] for p in r.get("walls", []) if "y1" in p]
        if not xs or not ys:
            continue
        bboxes.append((min(xs), min(ys), max(xs), max(ys)))

    overlap_penalty = 0.0
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            inter = _bbox_intersection_area(bboxes[i], bboxes[j])
            area_i = (bboxes[i][2] - bboxes[i][0]) * (bboxes[i][3] - bboxes[i][1])
            area_j = (bboxes[j][2] - bboxes[j][0]) * (bboxes[j][3] - bboxes[j][1])
            min_area = min(area_i, area_j) if min(area_i, area_j) > 0 else 1
            overlap_penalty += inter / min_area

    sliver_penalty = 0.0
    areas = [r["area"] for r in rooms]
    if areas:
        sorted_areas = sorted(areas)
        median_area = sorted_areas[len(sorted_areas) // 2]
        for a in areas:
            if median_area > 0 and a < median_area * 0.10:
                sliver_penalty += 0.4

    coverage_score = 1.0 - abs(coverage - 0.60)
    room_count_score = min(len(rooms), 10) / 10.0

    score = coverage_score * 0.50 + room_count_score * 0.25
    score -= overlap_penalty * 0.4
    score -= sliver_penalty

    return max(score, -1.0)


# ---------------------------------------------------------------------------
# Main segmentation pipeline
# ---------------------------------------------------------------------------

def _result_is_plausible(rooms: list, orig_w: int, orig_h: int, px_to_meter: float) -> bool:
    if not rooms or len(rooms) < 2:
        return False
    total_area_m2 = sum(r["area"] for r in rooms)
    image_area_m2 = (orig_w / px_to_meter) * (orig_h / px_to_meter)
    coverage = total_area_m2 / image_area_m2 if image_area_m2 > 0 else 0
    return 0.15 <= coverage <= 1.05


def _segment_rooms(gray: np.ndarray, w: int, h: int,
                   px_to_meter: float) -> List[Dict[str, Any]]:
    total_px = w * h
    min_room_area_px = total_px * 0.006

    prep = _preprocess_adaptive(gray)

    PARAM_SETS = [
        {"canny_low": 30, "canny_high": 100, "hough_thresh": 25, "close_size": 9, "dist_thresh": 0.33},
        {"canny_low": 50, "canny_high": 150, "hough_thresh": 40, "close_size": 9, "dist_thresh": 0.35},
        {"canny_low": 20, "canny_high": 80, "hough_thresh": 20, "close_size": 11, "dist_thresh": 0.30},
        {"canny_low": 50, "canny_high": 150, "hough_thresh": 40, "close_size": 7, "dist_thresh": 0.40},
        {"canny_low": 40, "canny_high": 120, "hough_thresh": 30, "close_size": 13, "dist_thresh": 0.28},
        {"canny_low": 25, "canny_high": 90, "hough_thresh": 18, "close_size": 15, "dist_thresh": 0.25},
    ]

    best_rooms: List[Dict[str, Any]] = []
    best_score = -1.0

    for params in PARAM_SETS:
        try:
            walls = _build_wall_mask_full(
                prep["gray"], prep["enhanced"], prep["adaptive_thresh"],
                w, h, params
            )
            rooms = _rooms_from_wall_mask(
                walls, w, h, px_to_meter, min_room_area_px, params["dist_thresh"]
            )
            score = _score_result(rooms, w, h, px_to_meter)
            if score > best_score:
                best_score = score
                best_rooms = rooms
        except Exception as e:
            logger.debug("Parameter set failed: %s - %s", params, e)
            continue

    if best_score < 0.35:
        logger.info("Low score %.2f, trying CAD double-line fallback", best_score)
        try:
            edges = cv2.Canny(prep["enhanced"], 20, 80, apertureSize=3)
            min_line_len = max(10, int(math.sqrt(total_px) * 0.015))
            lines = cv2.HoughLinesP(
                edges, rho=1, theta=np.pi / 180,
                threshold=15, minLineLength=min_line_len, maxLineGap=8,
            )
            cad_walls = np.zeros((h, w), dtype=np.uint8)
            if lines is not None:
                for line in lines:
                    coords = np.asarray(line).reshape(-1)
                    if coords.size < 4:
                        continue
                    x1, y1, x2, y2 = (int(coords[0]), int(coords[1]),
                                        int(coords[2]), int(coords[3]))
                    dx, dy = x2 - x1, y2 - y1
                    angle = math.degrees(math.atan2(abs(dy), abs(dx)))
                    if not (angle < 10 or angle > 80):
                        continue
                    if angle < 10:
                        y2 = y1
                    else:
                        x2 = x1
                    cv2.line(cad_walls, (x1, y1), (x2, y2), 255, thickness=2)

            bridge_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            cad_walls = cv2.morphologyEx(cad_walls, cv2.MORPH_CLOSE, bridge_k, iterations=2)
            cad_rooms = _rooms_from_wall_mask(cad_walls, w, h, px_to_meter, min_room_area_px, 0.33)
            cad_score = _score_result(cad_rooms, w, h, px_to_meter)
            if cad_score > best_score:
                best_score = cad_score
                best_rooms = cad_rooms
        except Exception as e:
            logger.warning("CAD fallback failed: %s", e)

    return best_rooms


def extract_walls_via_contours(image_bytes: bytes) -> List[Dict[str, Any]]:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_color is None:
        raise ValueError("Could not decode image bytes")

    h, w = img_color.shape[:2]
    px_to_meter = h / 14.0

    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    rooms = _segment_rooms(gray, w, h, px_to_meter)

    return rooms


def build_response(rooms: List[Dict[str, Any]], floors: int) -> dict:
    return {
        "rooms": rooms,
        "totalRooms": len(rooms),
        "totalFloors": floors,
        "calculatedSqFt": round(sum(r["area"] for r in rooms) * 10.764, 1),
    }


def _rdp_simplify(pts, eps_px=3.0):
    if len(pts) < 3:
        return pts
    pts = pts.copy()
    stack = [(0, len(pts) - 1)]
    mask = [True] * len(pts)
    while stack:
        first, last = stack.pop()
        if last - first < 2:
            continue
        x1, y1 = pts[first]
        x2, y2 = pts[last]
        dx, dy = x2 - x1, y2 - y1
        denom = dx * dx + dy * dy
        max_dist = 0
        max_idx = first
        for i in range(first + 1, last):
            xi, yi = pts[i]
            if denom == 0:
                dist = (xi - x1) ** 2 + (yi - y1) ** 2
            else:
                t = ((xi - x1) * dx + (yi - y1) * dy) / denom
                if t < 0:
                    dist = (xi - x1) ** 2 + (yi - y1) ** 2
                elif t > 1:
                    dist = (xi - x2) ** 2 + (yi - y2) ** 2
                else:
                    px = x1 + t * dx
                    py = y1 + t * dy
                    dist = (xi - px) ** 2 + (yi - py) ** 2
            if dist > max_dist:
                max_dist = dist
                max_idx = i
        if max_dist > eps_px * eps_px:
            stack.append((first, max_idx))
            stack.append((max_idx, last))
        else:
            for i in range(first + 1, last):
                mask[i] = False
    return [pts[i] for i in range(len(pts)) if mask[i]]


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/v1/process-layout/image")
async def process_layout_image(file: UploadFile = File(...), floors: int = Query(1)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid format. Expected an image file.")

    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")

        t0 = time.time()
        rooms = extract_walls_via_contours(image_bytes)
        method = "opencv"

        nparr = np.frombuffer(image_bytes, np.uint8)
        img_check = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        classical_ok = False
        if img_check is not None:
            h_c, w_c = img_check.shape[:2]
            px_to_meter_c = h_c / 14.0
            classical_ok = _result_is_plausible(rooms, w_c, h_c, px_to_meter_c)

        if not classical_ok:
            logger.info("Classical result implausible, trying ML model")
            ml_rooms = segment_rooms_ml(image_bytes)
            if ml_rooms and img_check is not None:
                if _result_is_plausible(ml_rooms, w_c, h_c, px_to_meter_c):
                    rooms = ml_rooms
                    method = "ml"

        elapsed = time.time() - t0
        logger.info("Segmentation completed: method=%s rooms=%d time=%.2fs", method, len(rooms), elapsed)

        response = build_response(rooms, floors)
        response["segmentationMethod"] = method
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Image processing failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/api/v1/process-layout/sample")
async def process_layout_sample(floors: int = Query(1)):
    json_path = os.path.join(os.path.dirname(__file__), "blueprint_rooms.json")
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Sample layout file not found.")

    with open(json_path) as f:
        raw = json.load(f)

    all_xs = [p[0] for r in raw["rooms"] for p in r["polygon_points"]]
    all_ys = [p[1] for r in raw["rooms"] for p in r["polygon_points"]]
    img_w = max(all_xs) - min(all_xs) + 200
    img_h = max(all_ys) - min(all_ys) + 200
    px_to_meter = img_h / 14.0
    cx = (min(all_xs) + max(all_xs)) / 2.0
    cy = (min(all_ys) + max(all_ys)) / 2.0

    rooms_out = []
    for r in raw["rooms"]:
        pts_px = r["polygon_points"]
        if len(pts_px) < 3:
            continue

        xs_px = [p[0] for p in pts_px]
        ys_px = [p[1] for p in pts_px]
        pw = max(xs_px) - min(xs_px)
        ph = max(ys_px) - min(ys_px)
        if pw < 50 or ph < 50:
            continue
        min_dim_px = min(pw, ph)
        max_dim_px = max(pw, ph)
        if max_dim_px > 0 and min_dim_px / max_dim_px < 0.12:
            continue

        simplified = _rdp_simplify(pts_px, eps_px=5.0)
        if len(simplified) < 3:
            simplified = pts_px
        snapped = _snap_orthogonal_strict(np.array(simplified, dtype=np.int32).reshape(-1, 2))
        snapped_pts = snapped.tolist()

        pts_m = [[(px - cx) / px_to_meter, (py - cy) / px_to_meter] for px, py in snapped_pts]
        xs = [p[0] for p in pts_m]
        ys = [p[1] for p in pts_m]
        bb_w = max(xs) - min(xs)
        bb_h = max(ys) - min(ys)

        min_side = min(bb_w, bb_h)
        area_m2 = round(_polygon_area(pts_m), 2)

        if min_side < 0.8 or area_m2 < 1.0:
            continue

        floors_out = polygon_to_walls(pts_m)

        rooms_out.append({
            "label": r["label"],
            "dimensions": f"{bb_w:.1f}m x {bb_h:.1f}m",
            "centerX": sum(xs) / len(xs),
            "centerY": sum(ys) / len(ys),
            "elevationZ": 0.0,
            "isOpenSpace": False,
            "walls": floors_out,
            "area": area_m2,
        })

    return build_response(rooms_out, floors)


@app.post("/api/v1/process-layout/json")
async def process_layout_json(payload: List[JSONRoom], floors: int = Query(1)):
    if not payload:
        raise HTTPException(status_code=400, detail="No rooms provided.")

    min_x = min(r.centerX - r.width / 2.0 for r in payload)
    max_x = max(r.centerX + r.width / 2.0 for r in payload)
    min_y = min(r.centerY - r.height / 2.0 for r in payload)
    max_y = max(r.centerY + r.height / 2.0 for r in payload)
    canvas_w = max_x - min_x
    canvas_h = max_y - min_y
    canvas_cx = (min_x + max_x) / 2.0
    canvas_cy = (min_y + max_y) / 2.0
    target_dim = 14.0
    px_to_meter = max(canvas_w, canvas_h) / target_dim
    if px_to_meter <= 0:
        raise HTTPException(status_code=400, detail="Invalid room dimensions.")

    rooms_output = []
    for r in payload:
        raw_points = r.polygon_points

        xs_px = [p[0] for p in raw_points]
        ys_px = [p[1] for p in raw_points]
        pw = max(xs_px) - min(xs_px)
        ph = max(ys_px) - min(ys_px)
        if pw < 50 or ph < 50:
            continue
        min_dim_px = min(pw, ph)
        max_dim_px = max(pw, ph)
        if max_dim_px > 0 and min_dim_px / max_dim_px < 0.12:
            continue

        pts_m = [[(px - canvas_cx) / px_to_meter, (py - canvas_cy) / px_to_meter]
                 for px, py in raw_points]
        xs = [p[0] for p in pts_m]
        ys = [p[1] for p in pts_m]
        rw = max(xs) - min(xs)
        rh = max(ys) - min(ys)
        area_m2 = round(_polygon_area(pts_m), 2)

        rooms_output.append({
            "label": r.label,
            "dimensions": f"{rw:.1f}m x {rh:.1f}m",
            "centerX": sum(xs) / len(xs),
            "centerY": sum(ys) / len(ys),
            "elevationZ": 0.0,
            "isOpenSpace": False,
            "walls": polygon_to_walls(pts_m),
            "area": area_m2,
        })

    return build_response(rooms_output, floors)


@app.post("/api/v1/process-layout/procedural")
async def process_layout_procedural(payload: ProceduralGenerationPayload):
    if not payload.rooms:
        raise HTTPException(status_code=400, detail="No rooms specified.")

    rooms_output = []
    rooms_per_floor = math.ceil(len(payload.rooms) / payload.total_floors)
    grid_size = math.ceil(math.sqrt(rooms_per_floor))
    floor_counters = {}
    for idx, r_spec in enumerate(payload.rooms):
        fl = r_spec.floorAssigned
        if fl not in floor_counters:
            floor_counters[fl] = 0
        current_floor_idx = floor_counters[fl]
        floor_counters[fl] += 1
        col = current_floor_idx % grid_size
        row = current_floor_idx // grid_size
        room_area_meters = r_spec.roomSqFt / 10.764
        side = math.sqrt(room_area_meters)
        cx = (col * side) - ((grid_size * side) / 2.0) + (side / 2.0)
        cy = (row * side) - ((grid_size * side) / 2.0) + (side / 2.0)
        rooms_output.append({
            "label": f"Lvl {fl} - {r_spec.name}",
            "dimensions": f"{side:.1f}m x {side:.1f}m",
            "centerX": cx,
            "centerY": cy,
            "elevationZ": float((fl - 1) * 3.0),
            "isOpenSpace": r_spec.isOpenSpace,
            "walls": [
                {"x1": cx - side / 2, "y1": cy - side / 2, "x2": cx + side / 2, "y2": cy - side / 2},
                {"x1": cx + side / 2, "y1": cy - side / 2, "x2": cx + side / 2, "y2": cy + side / 2},
                {"x1": cx + side / 2, "y1": cy + side / 2, "x2": cx - side / 2, "y2": cy + side / 2},
                {"x1": cx - side / 2, "y1": cy + side / 2, "x2": cx - side / 2, "y2": cy - side / 2},
            ],
            "area": float(room_area_meters),
        })
    return {
        "rooms": rooms_output,
        "totalRooms": len(rooms_output),
        "totalFloors": payload.total_floors,
        "calculatedSqFt": payload.total_sq_ft,
    }


@app.post("/api/v1/projects")
async def create_project(payload: SaveLayoutRequest, authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")

    token = authorization.replace("Bearer ", "")
    try:
        from database import get_supabase
        sb = get_supabase()
        user = sb.auth.get_user(token)
        user_id = user.user.id
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

    project = save_project(user_id, payload.name, payload.image_url, payload.total_floors)
    saved_rooms = save_rooms(project["id"], payload.rooms)

    return {
        "project": project,
        "rooms": saved_rooms,
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    reload = os.environ.get("ENVIRONMENT", "development") == "development"
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=reload)
