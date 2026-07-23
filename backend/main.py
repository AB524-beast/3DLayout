import uvicorn
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
import math
from shapely.geometry import Polygon
import json
import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tracing import setup_tracing
from database import save_project, save_rooms
from model_inference import segment_rooms_ml, is_model_available
from yolo_inference import (
    is_yolo_available, detect_objects, decode_image_to_cv2,
    build_wall_mask_from_yolo, segment_rooms_from_yolo_walls,
    get_yolo_room_labels,
)

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

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_DIMENSION = 8192
MIN_IMAGE_DIMENSION = 64


@app.get("/")
async def root():
    return {"status": "running", "service": "3d-layout-backend"}


@app.get("/health")
async def health():
    return {"status": "ok", "model_available": is_model_available(),
            "yolo_available": is_yolo_available()}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

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


def _polygon_area(pts: List[Tuple[float, float]]) -> float:
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


def _point_in_polygon(px: float, py: float, polygon: List[Tuple[float, float]]) -> bool:
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _polygons_overlap(poly_a: List[Tuple[float, float]],
                      poly_b: List[Tuple[float, float]]) -> bool:
    for pt in poly_b:
        if _point_in_polygon(pt[0], pt[1], poly_a):
            return True
    for pt in poly_a:
        if _point_in_polygon(pt[0], pt[1], poly_b):
            return True
    return False


def _snap_orthogonal(pts: np.ndarray, tol_deg: float = 8.0) -> np.ndarray:
    n = len(pts)
    snapped = pts.copy().astype(np.float64)
    for i in range(n):
        p1 = snapped[i]
        p2 = snapped[(i + 1) % n]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        if max(abs(dx), abs(dy)) < 0.5:
            continue
        angle = abs(math.degrees(math.atan2(abs(dy), abs(dx))))
        if angle < tol_deg:
            snapped[(i + 1) % n][1] = p1[1]
        elif angle > 90.0 - tol_deg:
            snapped[(i + 1) % n][0] = p1[0]
    return snapped.astype(np.int32)


def _collapse_short_edges(pts: np.ndarray, min_len: float) -> np.ndarray:
    """Removes spurious short micro-segments that cause staircase/zigzag
    artifacts when each tiny edge gets snapped individually."""
    pts = pts.astype(np.float64).tolist()
    changed = True
    while changed and len(pts) > 4:
        changed = False
        for i in range(len(pts)):
            p1, p2 = pts[i], pts[(i + 1) % len(pts)]
            if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) < min_len:
                del pts[(i + 1) % len(pts)]
                changed = True
                break
    return np.array(pts, dtype=np.float64)


def _snap_orthogonal_alternating(pts: np.ndarray) -> np.ndarray:
    """Snaps each edge to horizontal or vertical, forcing strict H/V
    alternation — prevents two same-orientation snaps in a row from
    creating a staircase instead of one corner."""
    n = len(pts)
    snapped = pts.copy().astype(np.float64)
    last_orientation = None
    for i in range(n):
        p1, p2 = snapped[i], snapped[(i + 1) % n]
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


def _make_valid_simple_polygon(pts: np.ndarray) -> np.ndarray:
    """Guarantees a valid, non-self-intersecting polygon using Shapely's
    buffer(0) repair — fixes the bowtie/spike artifact that alternating
    orthogonal snapping can produce on complex shapes. Falls back to the
    convex hull (mathematically guaranteed simple) if repair still fails."""
    try:
        poly = Polygon(pts)
        if poly.is_valid and poly.area > 0:
            return pts
        repaired = poly.buffer(0)
        if repaired.geom_type == "Polygon" and repaired.area > 0:
            coords = np.array(repaired.exterior.coords[:-1], dtype=np.int32)
            if len(coords) >= 4:
                return coords
        hull = cv2.convexHull(pts.astype(np.int32))
        return hull.reshape(-1, 2)
    except Exception:
        hull = cv2.convexHull(pts.astype(np.int32))
        return hull.reshape(-1, 2)


# ============================================================
# WALL DETECTION — multiple sensitivity profiles feed watershed
# ============================================================

def _build_wall_mask(gray: np.ndarray, w: int, h: int,
                      canny_low: int, canny_high: int,
                      hough_thresh: int, min_wall_frac: float) -> np.ndarray:
    """One wall-detection pass at a given sensitivity: Hough straight-line
    detection (clean but can miss faint walls) unioned with Otsu threshold
    (catches faint/gray walls Hough misses)."""
    total_px = w * h

    edges = cv2.Canny(gray, canny_low, canny_high, apertureSize=3)
    min_line_len = max(15, int(math.sqrt(total_px) * 0.02))
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=hough_thresh, minLineLength=min_line_len, maxLineGap=20,
    )

    hough_walls = np.zeros((h, w), dtype=np.uint8)
    ANGLE_TOL_DEG = 10.0
    if lines is not None:
        for line in lines:
            coords = np.asarray(line).reshape(-1)
            if coords.size < 4:
                continue
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            dx, dy = x2 - x1, y2 - y1
            angle = math.degrees(math.atan2(abs(dy), abs(dx)))
            is_h = angle < ANGLE_TOL_DEG
            is_v = angle > 90 - ANGLE_TOL_DEG
            if not (is_h or is_v):
                continue
            if is_h:
                y2 = y1
            else:
                x2 = x1
            cv2.line(hough_walls, (x1, y1), (x2, y2), 255, thickness=4)

    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    thresh_walls = np.zeros_like(dark)
    min_wall_area_px = total_px * min_wall_frac
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_wall_area_px:
            thresh_walls[labels == i] = 255

    walls = cv2.bitwise_or(hough_walls, thresh_walls)
    if cv2.countNonZero(walls) < total_px * 0.01:
        walls = dark
    return walls


def _build_wall_mask_double_line(gray: np.ndarray, w: int, h: int) -> np.ndarray:
    """Tuned for CAD-style double-line walls (two thin parallel lines with
    a gap for wall thickness) — a large closing kernel bridges the gap
    into one solid wall."""
    edges = cv2.Canny(gray, 20, 80, apertureSize=3)
    total_px = w * h
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=15, minLineLength=max(10, int(math.sqrt(total_px) * 0.015)),
        maxLineGap=8,
    )

    wall_mask = np.zeros((h, w), dtype=np.uint8)
    if lines is not None:
        for line in lines:
            coords = np.asarray(line).reshape(-1)
            if coords.size < 4:
                continue
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            dx, dy = x2 - x1, y2 - y1
            angle = math.degrees(math.atan2(abs(dy), abs(dx)))
            if not (angle < 10 or angle > 80):
                continue
            if angle < 10:
                y2 = y1
            else:
                x2 = x1
            cv2.line(wall_mask, (x1, y1), (x2, y2), 255, thickness=2)

    bridge_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, bridge_k, iterations=2)
    return wall_mask


def _remove_border_region(rooms_bin: np.ndarray) -> np.ndarray:
    """Floods inward from the image border and blanks out whatever open
    region touches it — the page margin/scan background can never be
    mistaken for a room."""
    h, w = rooms_bin.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    filled = rooms_bin.copy()
    seeds = [
        (0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1),
        (0, w // 2), (h - 1, w // 2), (h // 2, 0), (h // 2, w - 1),
    ]
    for (sy, sx) in seeds:
        if filled[sy, sx] == 255:
            cv2.floodFill(filled, flood_mask, (sx, sy), 128)
    result = rooms_bin.copy()
    result[filled == 128] = 0
    return result


# ============================================================
# WATERSHED SEGMENTATION — wall mask -> room polygons
# ============================================================

def _rooms_from_wall_mask(walls: np.ndarray, w: int, h: int,
                          px_to_meter: float, min_room_area_px: float,
                          dist_thresh: float) -> List[Dict[str, Any]]:
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, close_k, iterations=2)

    rooms_bin = cv2.bitwise_not(walls)
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    rooms_bin = cv2.morphologyEx(rooms_bin, cv2.MORPH_OPEN, open_k, iterations=1)
    rooms_bin = _remove_border_region(rooms_bin)

    smooth_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    rooms_bin = cv2.morphologyEx(rooms_bin, cv2.MORPH_CLOSE, smooth_k, iterations=1)
    rooms_bin = cv2.morphologyEx(rooms_bin, cv2.MORPH_OPEN, smooth_k, iterations=1)

    # ---- Distance transform + watershed: separates rooms whose shared
    # wall has gaps/breaks, by seeding from each room's confident interior
    # peak rather than trusting the outer contour alone ----
    dist = cv2.distanceTransform(rooms_bin, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    _, sure_fg = cv2.threshold(dist_norm, dist_thresh, 1.0, cv2.THRESH_BINARY)
    sure_fg = (sure_fg * 255).astype(np.uint8)

    num_markers, markers = cv2.connectedComponents(sure_fg)
    if num_markers <= 1:
        return []

    markers = markers + 1
    unknown = cv2.subtract(rooms_bin, sure_fg)
    markers[unknown == 255] = 0

    img_for_ws = cv2.cvtColor(rooms_bin, cv2.COLOR_GRAY2BGR)
    cv2.watershed(img_for_ws, markers)

    rooms = []
    for label in range(2, markers.max() + 1):
        mask = np.uint8(markers == label) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < min_room_area_px:
            continue

        rect = cv2.minAreaRect(cnt)
        (_, _), (rect_w, rect_h), _ = rect
        rect_area = rect_w * rect_h
        fill_ratio = area / rect_area if rect_area > 0 else 0

        if fill_ratio > 0.72:
            box = cv2.boxPoints(rect)
            approx = box.reshape(-1, 1, 2).astype(np.int32)
        else:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            escalate = 0.02
            while len(approx) > 10 and escalate < 0.08:
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
            snapped = _make_valid_simple_polygon(snapped)  # <- Shapely repair
            approx = snapped.reshape(-1, 1, 2)

        xs_px = [int(pt[0][0]) for pt in approx]
        ys_px = [int(pt[0][1]) for pt in approx]
        edge_margin = 3
        if (min(xs_px) <= edge_margin or min(ys_px) <= edge_margin or
                max(xs_px) >= w - edge_margin or max(ys_px) >= h - edge_margin):
            continue

        pts_m = [[(px - w / 2.0) / px_to_meter, (py - h / 2.0) / px_to_meter]
                  for px, py in [pt[0] for pt in approx]]
        xs = [p[0] for p in pts_m]
        ys = [p[1] for p in pts_m]
        bb_w = max(xs) - min(xs)
        bb_h = max(ys) - min(ys)
        if bb_w < 0.8 or bb_h < 0.8 or bb_w > 18 or bb_h > 18:
            continue
        min_side, max_side = min(bb_w, bb_h), max(bb_w, bb_h)
        if max_side > 0 and min_side / max_side < 0.15:
            continue
        img_w_m, img_h_m = w / px_to_meter, h / px_to_meter
        if bb_w > img_w_m * 0.85 and bb_h > img_h_m * 0.85:
            continue

        rooms.append({
            "label": f"Room {len(rooms) + 1}",
            "dimensions": f"{bb_w:.1f}m x {bb_h:.1f}m",
            "centerX": sum(xs) / len(xs),
            "centerY": sum(ys) / len(ys),
            "elevationZ": 0.0,
            "isOpenSpace": False,
            "walls": polygon_to_walls(pts_m),
            "area": round(bb_w * bb_h, 2),
        })
    return rooms


# ============================================================
# SCORING + MULTI-PASS ORCHESTRATION
# ============================================================

def _score_result(rooms: List[Dict[str, Any]], w: int, h: int, px_to_meter: float) -> float:
    if not rooms:
        return -1.0
    total_area = sum(r["area"] for r in rooms)
    image_area = (w / px_to_meter) * (h / px_to_meter)
    coverage = total_area / image_area if image_area > 0 else 0
    if coverage < 0.15 or coverage > 1.05:
        return -1.0

    overlap_penalty = 0.0
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            xi1 = rooms[i]["centerX"] - math.sqrt(rooms[i]["area"]) / 2
            xi2 = rooms[i]["centerX"] + math.sqrt(rooms[i]["area"]) / 2
            yi1 = rooms[i]["centerY"] - math.sqrt(rooms[i]["area"]) / 2
            yi2 = rooms[i]["centerY"] + math.sqrt(rooms[i]["area"]) / 2
            xj1 = rooms[j]["centerX"] - math.sqrt(rooms[j]["area"]) / 2
            xj2 = rooms[j]["centerX"] + math.sqrt(rooms[j]["area"]) / 2
            yj1 = rooms[j]["centerY"] - math.sqrt(rooms[j]["area"]) / 2
            yj2 = rooms[j]["centerY"] + math.sqrt(rooms[j]["area"]) / 2
            ox = max(0, min(xi2, xj2) - max(xi1, xj1))
            oy = max(0, min(yi2, yj2) - max(yi1, yj1))
            if ox > 0 and oy > 0:
                overlap_penalty += (ox * oy) / min(rooms[i]["area"], rooms[j]["area"])

    areas = [r["area"] for r in rooms]
    median_area = sorted(areas)[len(areas) // 2]
    sliver_penalty = sum(0.3 for r in rooms if median_area > 0 and r["area"] < median_area * 0.15)

    coverage_score = 1.0 - abs(coverage - 0.65)
    room_count_score = min(len(rooms), 8) / 8.0

    score = coverage_score * 0.55 + room_count_score * 0.20
    score -= overlap_penalty * 0.5
    score -= sliver_penalty
    return score


# ============================================================
# Main segmentation pipeline
# ============================================================

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
    min_room_area_px = total_px * 0.008

    PARAM_SETS = [
        {"canny_low": 30, "canny_high": 100, "hough_thresh": 25, "min_wall_frac": 0.003, "dist_thresh": 0.35},
        {"canny_low": 50, "canny_high": 150, "hough_thresh": 40, "min_wall_frac": 0.003, "dist_thresh": 0.35},
        {"canny_low": 20, "canny_high": 80, "hough_thresh": 20, "min_wall_frac": 0.002, "dist_thresh": 0.30},
        {"canny_low": 50, "canny_high": 150, "hough_thresh": 40, "min_wall_frac": 0.004, "dist_thresh": 0.45},
    ]

    best_rooms: List[Dict[str, Any]] = []
    best_score = -1.0

    for params in PARAM_SETS:
        try:
            walls = _build_wall_mask(gray, w, h, params["canny_low"], params["canny_high"],
                                      params["hough_thresh"], params["min_wall_frac"])
            rooms = _rooms_from_wall_mask(walls, w, h, px_to_meter, min_room_area_px,
                                          params["dist_thresh"])
            score = _score_result(rooms, w, h, px_to_meter)
            if score > best_score:
                best_score = score
                best_rooms = rooms
        except Exception:
            continue

    if best_score < 0.4:
        try:
            cad_walls = _build_wall_mask_double_line(gray, w, h)
            cad_rooms = _rooms_from_wall_mask(cad_walls, w, h, px_to_meter, min_room_area_px, 0.35)
            cad_score = _score_result(cad_rooms, w, h, px_to_meter)
            if cad_score > best_score:
                best_score = cad_score
                best_rooms = cad_rooms
        except Exception:
            pass

    return best_rooms


def extract_walls_via_contours(image_bytes: bytes) -> List[Dict[str, Any]]:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_color is None:
        raise ValueError("Could not decode image bytes")

    h, w = img_color.shape[:2]

    if w < MIN_IMAGE_DIMENSION or h < MIN_IMAGE_DIMENSION:
        raise ValueError(f"Image too small ({w}x{h}). Minimum is {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION}.")
    if w > MAX_IMAGE_DIMENSION or h > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / max(w, h)
        img_color = cv2.resize(img_color, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = img_color.shape[:2]
        logger.info("Resized large image to %dx%d", w, h)

    px_to_meter = h / 14.0

    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    rooms = _segment_rooms(gray, w, h, px_to_meter)

    return rooms


def build_response(rooms: List[Dict[str, Any]], floors: int,
                   method: str = "unknown", elapsed: float = 0.0) -> dict:
    return {
        "rooms": rooms,
        "totalRooms": len(rooms),
        "totalFloors": floors,
        "calculatedSqFt": round(sum(r["area"] for r in rooms) * 10.764, 1),
        "segmentationMethod": method,
        "processingTimeMs": round(elapsed * 1000, 1),
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
    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")
        if len(image_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {MAX_UPLOAD_BYTES // (1024 * 1024)}MB."
            )

        t0 = time.time()

        img_color = decode_image_to_cv2(image_bytes)
        h_orig, w_orig = img_color.shape[:2]

        if w_orig < MIN_IMAGE_DIMENSION or h_orig < MIN_IMAGE_DIMENSION:
            raise HTTPException(
                status_code=400,
                detail=f"Image too small ({w_orig}x{h_orig}). Minimum is {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION}.",
            )
        if w_orig > MAX_IMAGE_DIMENSION or h_orig > MAX_IMAGE_DIMENSION:
            scale = MAX_IMAGE_DIMENSION / max(w_orig, h_orig)
            img_color = cv2.resize(img_color, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)
            h_orig, w_orig = img_color.shape[:2]
            logger.info("Resized large image to %dx%d", w_orig, h_orig)

        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        px_to_meter = h_orig / 14.0
        rooms = []
        method = "none"

        # ── PRIMARY: YOLO object detection → wall mask → room segmentation ──
        yolo_detections = detect_objects(image_bytes)
        if yolo_detections:
            logger.info("YOLO detected %d objects", len(yolo_detections))
            yolo_walls = build_wall_mask_from_yolo(gray, yolo_detections, w_orig, h_orig)
            if yolo_walls is not None:
                yolo_rooms = segment_rooms_from_yolo_walls(
                    yolo_walls, gray, w_orig, h_orig, px_to_meter,
                )
                if yolo_rooms and _result_is_plausible(yolo_rooms, w_orig, h_orig, px_to_meter):
                    rooms = yolo_rooms
                    method = "yolo"

        # ── FALLBACK 1: Classical OpenCV edge-based segmentation ──
        if not rooms:
            logger.info("YOLO pipeline produced no rooms, falling back to OpenCV")
            rooms = _segment_rooms(gray, w_orig, h_orig, px_to_meter)
            if rooms:
                method = "opencv"

        # ── FALLBACK 2: ML ONNX room segmenter ──
        if not rooms or not _result_is_plausible(rooms, w_orig, h_orig, px_to_meter):
            logger.info("Trying ML ONNX segmenter")
            ml_rooms = segment_rooms_ml(image_bytes)
            if ml_rooms and _result_is_plausible(ml_rooms, w_orig, h_orig, px_to_meter):
                rooms = ml_rooms
                method = "ml"

        # ── Enrich rooms with YOLO-detected objects (doors, windows, etc.) ──
        if yolo_detections and rooms:
            rooms = get_yolo_room_labels(yolo_detections, rooms, w_orig, h_orig)

        elapsed = time.time() - t0
        logger.info("Segmentation completed: method=%s rooms=%d time=%.2fs",
                     method, len(rooms), elapsed)

        response = build_response(rooms, floors, method=method, elapsed=elapsed)
        if yolo_detections:
            response["yoloObjects"] = [
                {"class": d["class"], "confidence": round(d["confidence"], 3),
                 "bbox": d["bbox"]}
                for d in yolo_detections
            ]
        return response

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
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
        snapped = _snap_orthogonal(np.array(simplified, dtype=np.int32).reshape(-1, 2))
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

    return build_response(rooms_out, floors, method="sample")


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

    return build_response(rooms_output, floors, method="json")


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
        "segmentationMethod": "procedural",
        "processingTimeMs": 0,
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

    try:
        project = save_project(user_id, payload.name, payload.image_url, payload.total_floors)
        saved_rooms = save_rooms(project["id"], payload.rooms)
    except Exception as e:
        logger.error("Failed to save project: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to save project.")

    return {
        "project": project,
        "rooms": saved_rooms,
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    reload = os.environ.get("ENVIRONMENT", "development") == "development"
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=reload)
