import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
import cv2
import math
import json
import os

app = FastAPI(title="Orthogonal Blueprint Spatial Modeler")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RoomSpecification(BaseModel):
    name: str
    floorAssigned: int
    isOpenSpace: bool
    roomSqFt: float


class ProceduralGenerationPayload(BaseModel):
    total_sq_ft: float
    total_floors: int
    rooms: List[RoomSpecification]


def polygon_to_walls(points_m: List[List[float]]) -> List[Dict[str, float]]:
    walls = []
    n = len(points_m)
    for i in range(n):
        x1, y1 = points_m[i]
        x2, y2 = points_m[(i + 1) % n]
        walls.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return walls


def _remove_noise(binary: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if num_labels < 3:
        return opened
    areas = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, num_labels)]
    areas.sort(reverse=True)
    largest = areas[0][0]
    keep = {i for a, i in areas if a >= largest * 0.005}
    result = np.zeros_like(opened)
    for i in keep:
        result[labels == i] = 255
    if np.max(result) == 0:
        return binary
    return result


def _snap_orthogonal(pts: np.ndarray, tol_deg: float = 8.0) -> np.ndarray:
    n = len(pts)
    snapped = pts.copy().astype(np.float64)
    for i in range(n):
        p1 = snapped[i]
        p2 = snapped[(i + 1) % n]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if max(abs(dx), abs(dy)) < 0.5:
            continue
        angle = abs(math.degrees(math.atan2(abs(dy), abs(dx))))
        if angle < tol_deg:
            snapped[(i + 1) % n][1] = p1[1]
        elif angle > 90.0 - tol_deg:
            snapped[(i + 1) % n][0] = p1[0]
    return snapped.astype(np.int32)


def _extract_rooms_from_binary(binary: np.ndarray, w: int, h: int, px_to_meter: float,
                                min_rel_area: float = 0.003, max_rel_area: float = 0.85,
                                min_dim_m: float = 0.5, max_dim_m: float = 18.0) -> List[Dict[str, Any]]:
    rooms = []
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or contours is None or len(contours) == 0:
        return rooms
    hierarchy = hierarchy[0]

    contour_areas = np.array([cv2.contourArea(c) for c in contours])
    depths = np.zeros(len(contours), dtype=np.int32)
    for i in range(len(contours)):
        p = hierarchy[i][3]
        d = 0
        while p != -1:
            d += 1
            p = hierarchy[p][3]
        depths[i] = d

    for idx, cnt in enumerate(contours):
        parent = hierarchy[idx][3]
        if parent == -1:
            continue
        parent_area = contour_areas[parent]
        child_area = contour_areas[idx]

        if depths[idx] != 1:
            continue
        if parent_area <= 0 or child_area < parent_area * 0.05:
            continue

        area_px = child_area
        if area_px < (w * h * min_rel_area) or area_px > (w * h * max_rel_area):
            continue

        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue

        epsilon = 0.012 * peri
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) < 4:
            continue

        approx_pts = approx.reshape(-1, 2)
        snapped = _snap_orthogonal(approx_pts)
        approx = snapped.reshape(-1, 1, 2)

        pts_m = []
        for point in approx:
            px, py = point[0]
            mx = (px - w / 2.0) / px_to_meter
            my = (py - h / 2.0) / px_to_meter
            pts_m.append([mx, my])

        xs = [p[0] for p in pts_m]
        ys = [p[1] for p in pts_m]
        bb_w = max(xs) - min(xs)
        bb_h = max(ys) - min(ys)

        if bb_w < min_dim_m or bb_h < min_dim_m or bb_w > max_dim_m or bb_h > max_dim_m:
            continue

        min_side = min(bb_w, bb_h)
        max_side = max(bb_w, bb_h)
        if max_side > 0 and min_side / max_side < 0.10:
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


def _close_img(binary: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)


def _edge_based_binary(gray: np.ndarray) -> np.ndarray:
    med = np.median(gray)
    low = int(max(0, 0.3 * med))
    high = int(min(255, 1.2 * med))
    edges = cv2.Canny(gray, low, high, apertureSize=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=3)
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    return cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, close_k, iterations=1)


def _segment_via_watershed(gray: np.ndarray, w: int, h: int,
                            px_to_meter: float) -> List[Dict[str, Any]]:
    med = np.median(gray)
    low = int(max(0, 0.3 * med))
    high = int(min(255, 1.2 * med))
    edges = cv2.Canny(gray, low, high, apertureSize=3)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    barriers = cv2.dilate(edges, kernel3, iterations=4)
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    barriers = cv2.morphologyEx(barriers, cv2.MORPH_CLOSE, close_k, iterations=2)

    inv = cv2.bitwise_not(barriers)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, sure_fg = cv2.threshold(dist, 0.25 * 255, 255, cv2.THRESH_BINARY)
    sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_OPEN, kernel3, iterations=2)

    _, markers = cv2.connectedComponents(sure_fg.astype(np.uint8))
    markers = markers + 1
    markers[barriers == 255] = 0

    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color, markers)

    rooms = []
    for label in range(2, markers.max() + 1):
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[markers == label] = 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel3, iterations=1)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        epsilon = 0.012 * peri
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) < 4:
            continue
        approx_pts = approx.reshape(-1, 2)
        snapped = _snap_orthogonal(approx_pts)
        approx = snapped.reshape(-1, 1, 2)
        pts_m = []
        for point in approx:
            px, py = point[0]
            mx = (px - w / 2.0) / px_to_meter
            my = (py - h / 2.0) / px_to_meter
            pts_m.append([mx, my])
        xs = [p[0] for p in pts_m]
        ys = [p[1] for p in pts_m]
        bb_w = max(xs) - min(xs)
        bb_h = max(ys) - min(ys)
        if bb_w < 0.5 or bb_h < 0.5 or bb_w > 18 or bb_h > 18:
            continue
        min_side = min(bb_w, bb_h)
        max_side = max(bb_w, bb_h)
        if max_side > 0 and min_side / max_side < 0.10:
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


def _try_strategy(binary_fn, w: int, h: int, px_to_meter: float, **kwargs) -> List[Dict[str, Any]]:
    try:
        binary = binary_fn()
        if binary is None or np.max(binary) == 0:
            return []
        binary = _remove_noise(binary)
        return _extract_rooms_from_binary(binary, w, h, px_to_meter, **kwargs)
    except Exception:
        return []


def _detect_walls_via_lines(gray: np.ndarray, img_color: np.ndarray,
                             w: int, h: int, px_to_meter: float) -> List[Dict[str, Any]]:
    edges = cv2.Canny(gray, 40, 120, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, threshold=int(min(w, h) * 0.08),
                            minLineLength=int(min(w, h) * 0.04),
                            maxLineGap=int(min(w, h) * 0.02))
    if lines is None or len(lines) < 4:
        return []

    wall_mask = np.zeros((h, w), dtype=np.uint8)
    for line in lines:
        line = line.flatten()
        x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
        cv2.line(wall_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, 8)

    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, close_k, iterations=2)
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_OPEN, close_k, iterations=1)
    wall_mask = _remove_noise(wall_mask)

    inv = cv2.bitwise_not(wall_mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    if num_labels < 3:
        return []

    total_px = w * h
    candidates = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < total_px * 0.003 or area > total_px * 0.85:
            continue
        mask_i = (labels == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        epsilon = 0.012 * peri
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) < 4:
            continue
        approx_pts = approx.reshape(-1, 2)
        snapped = _snap_orthogonal(approx_pts)
        approx = snapped.reshape(-1, 1, 2)

        pts_m = []
        for point in approx:
            px, py = point[0]
            mx = (px - w / 2.0) / px_to_meter
            my = (py - h / 2.0) / px_to_meter
            pts_m.append([mx, my])

        xs = [p[0] for p in pts_m]
        ys = [p[1] for p in pts_m]
        bb_w = max(xs) - min(xs)
        bb_h = max(ys) - min(ys)

        if bb_w < 0.5 or bb_h < 0.5 or bb_w > 18 or bb_h > 18:
            continue
        min_side = min(bb_w, bb_h)
        max_side = max(bb_w, bb_h)
        if max_side > 0 and min_side / max_side < 0.10:
            continue
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0 and area / hull_area < 0.35:
            continue

        candidates.append({
            "label": f"Room {len(candidates) + 1}",
            "dimensions": f"{bb_w:.1f}m x {bb_h:.1f}m",
            "centerX": sum(xs) / len(xs),
            "centerY": sum(ys) / len(ys),
            "elevationZ": 0.0,
            "isOpenSpace": False,
            "walls": polygon_to_walls(pts_m),
            "area": round(bb_w * bb_h, 2),
        })

    return candidates


def extract_walls_via_contours(image_bytes: bytes) -> List[Dict[str, Any]]:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_color is None:
        return []

    h, w = img_color.shape[:2]
    px_to_meter = h / 14.0

    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    _, _, value = cv2.split(hsv)
    enhanced_hsv = clahe.apply(value)

    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    blurred_hsv = cv2.GaussianBlur(enhanced_hsv, (5, 5), 0)

    # Additional preprocessed versions for robustness
    blurred_raw = cv2.GaussianBlur(gray, (3, 3), 0)
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    blurred_bilateral = cv2.GaussianBlur(bilateral, (3, 3), 0)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    blurred_norm = cv2.GaussianBlur(norm, (3, 3), 0)

    best_rooms = []
    best_score = -1

    def dedup_rooms(rooms_list):
        if len(rooms_list) < 2:
            return rooms_list
        kept = []
        for r in rooms_list:
            dup = False
            for k in kept:
                if abs(r["centerX"] - k["centerX"]) < 0.5 and abs(r["centerY"] - k["centerY"]) < 0.5:
                    dup = True
                    break
            if not dup:
                kept.append(r)
        return kept

    def score_rooms(rooms):
        if not rooms:
            return -1, []
        rooms = dedup_rooms(rooms)
        if not rooms:
            return -1, []
        n = len(rooms)
        areas = [r["area"] for r in rooms]
        mean_area = sum(areas) / n
        valid_rooms = sum(1 for a in areas if 1.5 < a < 80)
        s = valid_rooms * 20
        if 2 < mean_area < 60:
            s += 20
        if n >= 3:
            s += 15
        if n >= 5:
            s += 10
        if n >= 8:
            s += 5
        return s, rooms

    def try_strat(fn, **kwargs):
        rooms = _try_strategy(fn, w, h, px_to_meter, **kwargs)
        nonlocal best_score, best_rooms
        s, deduped = score_rooms(rooms)
        if s > best_score:
            best_score = s
            best_rooms = deduped

    # --- Threshold-based strategies with multiple kernel sizes ---
    for ks in [5, 7, 11, 15, 21, 31, 41, 51, 61, 71]:

        try_strat(lambda ksv=ks: _close_img(cv2.threshold(blurred, 0, 255,
                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1], ksv))

        try_strat(lambda ksv=ks: _close_img(cv2.threshold(blurred_hsv, 0, 255,
                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1], ksv))

        try_strat(lambda ksv=ks: _close_img(cv2.adaptiveThreshold(blurred, 255,
                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3), ksv),
                   min_rel_area=0.002, min_dim_m=0.3)

        try_strat(lambda ksv=ks: _close_img(cv2.adaptiveThreshold(blurred, 255,
                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 4), ksv),
                   min_rel_area=0.002, min_dim_m=0.3)

        try_strat(lambda ksv=ks: _close_img(cv2.adaptiveThreshold(blurred, 255,
                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5), ksv),
                   min_rel_area=0.002, min_dim_m=0.3)

        try_strat(lambda ksv=ks: _close_img(cv2.threshold(blurred_raw, 0, 255,
                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1], ksv))

        try_strat(lambda ksv=ks: _close_img(cv2.threshold(blurred_bilateral, 0, 255,
                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1], ksv),
                   min_rel_area=0.002, min_dim_m=0.3)

        try_strat(lambda ksv=ks: _close_img(cv2.threshold(blurred_norm, 0, 255,
                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1], ksv))

        # Also try THRESH_BINARY + invert (for light walls on dark background)
        try_strat(lambda ksv=ks: cv2.bitwise_not(_close_img(cv2.threshold(blurred, 0, 255,
                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], ksv)))

        try_strat(lambda ksv=ks: cv2.bitwise_not(_close_img(cv2.threshold(blurred_hsv, 0, 255,
                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], ksv)))

        try_strat(lambda ksv=ks: cv2.bitwise_not(_close_img(cv2.threshold(blurred_norm, 0, 255,
                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], ksv)))

    # --- Edge-based strategy (Canny + dilate + close) ---
    try_strat(lambda: _edge_based_binary(bilateral))

    # --- Watershed fallback ---
    if best_score < 20:
        ws_rooms = _segment_via_watershed(bilateral, w, h, px_to_meter)
        s, deduped = score_rooms(ws_rooms)
        if s > best_score:
            best_score = s
            best_rooms = deduped

    # --- Wide close fallback ---
    if best_score < 20:
        _best_wide_score = -1
        _best_wide_rooms = []
        for src in [blurred, blurred_raw, blurred_norm]:
            _, bin_src = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            for wk in [71, 101, 151]:
                wide = _close_img(bin_src, wk)
                clean = _remove_noise(wide)
                r = _extract_rooms_from_binary(clean, w, h, px_to_meter)
                s2, deduped2 = score_rooms(r)
                if s2 > _best_wide_score:
                    _best_wide_score = s2
                    _best_wide_rooms = deduped2
        if _best_wide_score > best_score:
            best_score = _best_wide_score
            best_rooms = _best_wide_rooms

    # --- Hough line fallback ---
    line_rooms = _detect_walls_via_lines(gray, img_color, w, h, px_to_meter)
    s, deduped = score_rooms(line_rooms)
    if s > best_score:
        best_score = s
        best_rooms = deduped

    return best_rooms


def build_response(rooms: List[Dict[str, Any]], floors: int) -> dict:
    return {
        "rooms": rooms,
        "totalRooms": len(rooms),
        "totalFloors": floors,
        "calculatedSqFt": round(sum(r["area"] for r in rooms) * 10.764, 1),
    }


class JSONRoom(BaseModel):
    label: str
    centerX: float
    centerY: float
    width: float
    height: float
    polygon_points: List[List[float]]


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

    rooms_out = []
    for r in raw["rooms"]:
        pts_px = r["polygon_points"]
        if len(pts_px) < 3:
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
        area_px2 = 0.0
        n = len(snapped_pts)
        for i in range(n):
            x1, y1 = snapped_pts[i]
            x2, y2 = snapped_pts[(i + 1) % n]
            area_px2 += (x1 * y2) - (x2 * y1)
        area_m2 = round(abs(area_px2) / 2.0 / (px_to_meter ** 2), 2)

        if min_side < 0.8 or area_m2 < 1.0:
            continue

        floors_out = []
        for i in range(len(pts_m)):
            x1, y1 = pts_m[i]
            x2, y2 = pts_m[(i + 1) % len(pts_m)]
            floors_out.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

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


@app.post("/api/v1/process-layout/image")
async def process_layout_image(file: UploadFile = File(...), floors: int = Query(1)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid format.")
    try:
        image_bytes = await file.read()
        rooms = extract_walls_via_contours(image_bytes)
        return build_response(rooms, floors)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
        pts_m = [[(px - canvas_cx) / px_to_meter, (py - canvas_cy) / px_to_meter] for px, py in raw_points]
        xs = [p[0] for p in pts_m]
        ys = [p[1] for p in pts_m]
        rw = max(xs) - min(xs)
        rh = max(ys) - min(ys)
        area_px = 0.0
        n = len(raw_points)
        for i in range(n):
            x1, y1 = raw_points[i]
            x2, y2 = raw_points[(i + 1) % n]
            area_px += (x1 * y2) - (x2 * y1)
        area_m2 = round(abs(area_px) / 2.0 / (px_to_meter ** 2), 2)

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


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
