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
    keep = {i for a, i in areas if a >= largest * 0.02}
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
                                min_rel_area: float = 0.002, max_rel_area: float = 0.80,
                                min_dim_m: float = 0.3, max_dim_m: float = 18.0,
                                use_hierarchy: bool = True, invert_parent: bool = True,
                                gray: np.ndarray = None) -> List[Dict[str, Any]]:
    rooms = []
    method = cv2.RETR_TREE if use_hierarchy else cv2.RETR_EXTERNAL
    contours, hierarchy = cv2.findContours(binary, method, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None and use_hierarchy:
        return rooms
    if contours is None or len(contours) == 0:
        return rooms

    if use_hierarchy:
        hierarchy = hierarchy[0]

    if use_hierarchy:
        contour_areas = np.array([cv2.contourArea(c) for c in contours])
        depths = np.zeros(len(contours), dtype=np.int32)
        for i in range(len(contours)):
            p = hierarchy[i][3]
            d = 0
            while p != -1:
                d += 1
                p = hierarchy[p][3]
            depths[i] = d

    canny_edges = None
    if gray is not None:
        med = np.median(gray)
        low = int(max(0, 0.3 * med))
        high = int(min(255, 1.2 * med))
        canny_edges = cv2.Canny(gray, low, high, apertureSize=3)

    for idx, cnt in enumerate(contours):
        if use_hierarchy:
            parent = hierarchy[idx][3]
            if invert_parent and parent == -1:
                continue
            if not invert_parent and parent != -1:
                continue
            if invert_parent and depths[idx] >= 2:
                child_area = contour_areas[idx]
                parent_area = contour_areas[parent]
                if parent_area <= 0 or child_area < parent_area * 0.08:
                    continue

        area_px = cv2.contourArea(cnt)
        if area_px < (w * h * min_rel_area) or area_px > (w * h * max_rel_area):
            continue

        if gray is not None and area_px >= 500:
            mask = np.zeros(binary.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            interior = gray[mask == 255]
            if len(interior) > 0:
                if np.std(interior) > 55:
                    continue
                if canny_edges is not None:
                    interior_edges = canny_edges[mask == 255]
                    if len(interior_edges) > 0:
                        edge_density = np.sum(interior_edges > 0) / len(interior_edges)
                        if edge_density > 0.08:
                            continue

        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            continue
        solidity = area_px / hull_area
        if solidity < 0.45:
            continue

        epsilon = 0.01 * peri
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

        peri_m = peri / px_to_meter
        perim_rect = 2.0 * (bb_w + bb_h)
        if perim_rect > 0 and peri_m / perim_rect > 2.0:
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


def _try_strategy(binary_fn, w: int, h: int, px_to_meter: float,
                  gray: np.ndarray = None, **kwargs) -> List[Dict[str, Any]]:
    try:
        binary = binary_fn()
        if binary is None or np.max(binary) == 0:
            return []
        binary = _remove_noise(binary)
        return _extract_rooms_from_binary(binary, w, h, px_to_meter, gray=gray, **kwargs)
    except Exception:
        return []


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

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    blurred_hsv = cv2.GaussianBlur(enhanced_hsv, (5, 5), 0)

    candidates = []

    def _close(b, iters=2):
        return cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel, iterations=iters)

    def strat_otsu_gray():
        _, b = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return _close(b, 2)
    candidates.append(_try_strategy(strat_otsu_gray, w, h, px_to_meter, gray=gray))

    def strat_otsu_hsv():
        _, b = cv2.threshold(blurred_hsv, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return _close(b, 2)
    candidates.append(_try_strategy(strat_otsu_hsv, w, h, px_to_meter, gray=gray))

    def strat_adaptive():
        b = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 15, 4)
        return _close(b, 2)
    candidates.append(_try_strategy(strat_adaptive, w, h, px_to_meter,
                                    gray=gray, min_rel_area=0.002, min_dim_m=0.25))

    def strat_canny():
        med = np.median(blurred)
        lower = int(max(0, 0.3 * med))
        upper = int(min(255, 1.2 * med))
        edges = cv2.Canny(blurred, lower, upper, apertureSize=3)
        dilated = cv2.dilate(edges, kernel, iterations=3)
        return cv2.bitwise_not(dilated)
    candidates.append(_try_strategy(strat_canny, w, h, px_to_meter,
                                    gray=gray, use_hierarchy=False))

    best_rooms = []
    best_score = -1

    def dedup_rooms(rooms):
        if len(rooms) < 2:
            return rooms
        kept = []
        for r in rooms:
            dup = False
            for k in kept:
                if abs(r["centerX"] - k["centerX"]) < 0.5 and abs(r["centerY"] - k["centerY"]) < 0.5:
                    dup = True
                    break
            if not dup:
                kept.append(r)
        return kept

    for rooms in candidates:
        if not rooms:
            continue
        rooms = dedup_rooms(rooms)
        n = len(rooms)
        if n == 0:
            continue
        areas = [r["area"] for r in rooms]
        mean_area = sum(areas) / n
        std_area = (sum((a - mean_area) ** 2 for a in areas) / n) ** 0.5 if n > 1 else 0

        score = min(n, 12) * 15
        if 2 < mean_area < 80:
            score += 25
        if all(0.5 < a < 120 for a in areas):
            score += 15
        if std_area < mean_area * 0.9:
            score += 5
        if n >= 2:
            score += 5

        if score > best_score:
            best_score = score
            best_rooms = rooms

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

    rooms_out = []
    for r in raw["rooms"]:
        pts_m = [[(px - cx) / px_to_meter, (py - cy) / px_to_meter] for px, py in r["polygon_points"]]
        xs = [p[0] for p in pts_m]
        ys = [p[1] for p in pts_m]
        bb_w = max(xs) - min(xs)
        bb_h = max(ys) - min(ys)
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
            "area": round(bb_w * bb_h, 2),
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
