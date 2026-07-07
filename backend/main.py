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

try:
    from pipeline import BlueprintWatershedPipeline
except ModuleNotFoundError:
    from backend.pipeline import BlueprintWatershedPipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_ROOMS_PATH = os.path.join(BASE_DIR, "blueprint_rooms.json")

watershed_pipeline = BlueprintWatershedPipeline()

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

def generate_box_walls(cx: float, cy: float, width: float, height: float) -> List[Dict[str, float]]:
    x1, x2 = cx - (width / 2.0), cx + (width / 2.0)
    y1, y2 = cy - (height / 2.0), cy + (height / 2.0)
    return [
        {"x1": x1, "y1": y1, "x2": x2, "y2": y1},
        {"x1": x2, "y1": y1, "x2": x2, "y2": y2},
        {"x1": x2, "y1": y2, "x2": x1, "y2": y2},
        {"x1": x1, "y1": y2, "x2": x1, "y2": y1}
    ]

def _extract_rooms_from_binary(binary: np.ndarray, w: int, h: int, px_to_meter: float,
                                min_rel_area: float = 0.001, max_rel_area: float = 0.85,
                                min_dim_m: float = 0.15, max_dim_m: float = 18.0,
                                use_hierarchy: bool = True, invert_parent: bool = True) -> List[Dict[str, Any]]:
    rooms = []
    method = cv2.RETR_TREE if use_hierarchy else cv2.RETR_EXTERNAL
    contours, hierarchy = cv2.findContours(binary, method, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None and use_hierarchy:
        return rooms
    if contours is None or len(contours) == 0:
        return rooms

    if use_hierarchy:
        hierarchy = hierarchy[0]

    for idx, cnt in enumerate(contours):
        if use_hierarchy:
            parent = hierarchy[idx][3]
            if invert_parent and parent == -1:
                continue
            if not invert_parent and parent != -1:
                continue

        area_px = cv2.contourArea(cnt)
        if area_px < (w * h * min_rel_area) or area_px > (w * h * max_rel_area):
            continue

        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) < 3:
            continue

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

        rooms.append({
            "label": f"Room {len(rooms) + 1}",
            "dimensions": f"{bb_w:.1f}m x {bb_h:.1f}m",
            "centerX": sum(xs) / len(xs),
            "centerY": sum(ys) / len(ys),
            "elevationZ": 0.0,
            "isOpenSpace": False,
            "walls": polygon_to_walls(pts_m),
            "area": round(bb_w * bb_h, 2)
        })

    return rooms


def _try_strategy(binary_fn, label: str, w: int, h: int, px_to_meter: float, **kwargs) -> List[Dict[str, Any]]:
    try:
        binary = binary_fn()
        if binary is None or np.max(binary) == 0:
            return []
        rooms = _extract_rooms_from_binary(binary, w, h, px_to_meter, **kwargs)
        return rooms
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
    _, saturation, value = cv2.split(hsv)
    enhanced_hsv = clahe.apply(value)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    blurred_hsv = cv2.GaussianBlur(enhanced_hsv, (5, 5), 0)

    candidates = []

    # Strategy A: Otsu threshold on CLAHE grayscale
    def otsu_gray():
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    candidates.append(_try_strategy(otsu_gray, "otsu_gray", w, h, px_to_meter))

    # Strategy B: Otsu threshold on HSV value channel
    def otsu_hsv():
        _, binary = cv2.threshold(blurred_hsv, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    candidates.append(_try_strategy(otsu_hsv, "otsu_hsv", w, h, px_to_meter))

    # Strategy C: Adaptive threshold (small block)
    def adaptive_small():
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 3)
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    candidates.append(_try_strategy(adaptive_small, "adaptive_small", w, h, px_to_meter,
                                     min_rel_area=0.002, min_dim_m=0.2))

    # Strategy D: Adaptive threshold (large block)
    def adaptive_large():
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 25, 6)
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    candidates.append(_try_strategy(adaptive_large, "adaptive_large", w, h, px_to_meter))

    # Strategy E: Canny edges + dilation + hierarchy (interior rooms)
    def canny_interior():
        med = np.median(blurred)
        lower = int(max(0, 0.3 * med))
        upper = int(min(255, 1.2 * med))
        edges = cv2.Canny(blurred, lower, upper, apertureSize=3)
        dilated = cv2.dilate(edges, kernel, iterations=3)
        return cv2.bitwise_not(dilated)
    candidates.append(_try_strategy(canny_interior, "canny_interior", w, h, px_to_meter,
                                     invert_parent=True))

    # Strategy F: Canny + external (for open plans)
    def canny_external():
        med = np.median(blurred)
        lower = int(max(0, 0.3 * med))
        upper = int(min(255, 1.2 * med))
        edges = cv2.Canny(blurred, lower, upper, apertureSize=3)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        empty = cv2.bitwise_not(dilated)
        return cv2.morphologyEx(empty, cv2.MORPH_CLOSE, kernel5, iterations=2)
    candidates.append(_try_strategy(canny_external, "canny_external", w, h, px_to_meter,
                                     use_hierarchy=False, min_dim_m=0.5))

    # Strategy G: Mean shift filtering + Otsu
    def meanshift_otsu():
        filtered = cv2.pyrMeanShiftFiltering(img_color, 15, 30)
        gray_f = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        blurred_f = cv2.GaussianBlur(gray_f, (5, 5), 0)
        _, binary = cv2.threshold(blurred_f, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    candidates.append(_try_strategy(meanshift_otsu, "meanshift_otsu", w, h, px_to_meter))

    # Strategy H: Simple binary at 200
    def simple_binary():
        _, binary = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    candidates.append(_try_strategy(simple_binary, "simple_binary", w, h, px_to_meter,
                                     min_rel_area=0.002, min_dim_m=0.2))

    # Strategy I: Flood-fill room detection (invert + remove exterior via flood fill)
    def floodfill_rooms():
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel5, iterations=4)
        inv = cv2.bitwise_not(closed)
        hh, ww = inv.shape
        mask = np.zeros((hh + 2, ww + 2), np.uint8)
        cv2.floodFill(inv, mask, (0, 0), 0)
        cv2.floodFill(inv, mask, (ww - 1, 0), 0)
        cv2.floodFill(inv, mask, (0, hh - 1), 0)
        cv2.floodFill(inv, mask, (ww - 1, hh - 1), 0)
        cv2.floodFill(inv, mask, (ww // 2, 0), 0)
        cv2.floodFill(inv, mask, (ww // 2, hh - 1), 0)
        cv2.floodFill(inv, mask, (0, hh // 2), 0)
        cv2.floodFill(inv, mask, (ww - 1, hh // 2), 0)
        return inv
    candidates.append(_try_strategy(floodfill_rooms, "floodfill_rooms", w, h, px_to_meter,
                                     use_hierarchy=False, min_dim_m=0.15, min_rel_area=0.001))

    # Strategy J: Flood-fill with HSV Otsu base
    def floodfill_hsv():
        _, binary = cv2.threshold(blurred_hsv, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel5, iterations=4)
        inv = cv2.bitwise_not(closed)
        hh, ww = inv.shape
        mask = np.zeros((hh + 2, ww + 2), np.uint8)
        cv2.floodFill(inv, mask, (0, 0), 0)
        cv2.floodFill(inv, mask, (ww - 1, 0), 0)
        cv2.floodFill(inv, mask, (0, hh - 1), 0)
        cv2.floodFill(inv, mask, (ww - 1, hh - 1), 0)
        cv2.floodFill(inv, mask, (ww // 2, 0), 0)
        cv2.floodFill(inv, mask, (ww // 2, hh - 1), 0)
        cv2.floodFill(inv, mask, (0, hh // 2), 0)
        cv2.floodFill(inv, mask, (ww - 1, hh // 2), 0)
        return inv
    candidates.append(_try_strategy(floodfill_hsv, "floodfill_hsv", w, h, px_to_meter,
                                     use_hierarchy=False, min_dim_m=0.15, min_rel_area=0.001))

    # Strategy K: Very aggressive flood-fill — low threshold + large kernel
    def floodfill_aggressive():
        _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
        kernel11 = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel11, iterations=3)
        inv = cv2.bitwise_not(closed)
        hh, ww = inv.shape
        mask = np.zeros((hh + 2, ww + 2), np.uint8)
        cv2.floodFill(inv, mask, (0, 0), 0)
        cv2.floodFill(inv, mask, (ww - 1, 0), 0)
        cv2.floodFill(inv, mask, (0, hh - 1), 0)
        cv2.floodFill(inv, mask, (ww - 1, hh - 1), 0)
        cv2.floodFill(inv, mask, (ww // 2, 0), 0)
        cv2.floodFill(inv, mask, (ww // 2, hh - 1), 0)
        cv2.floodFill(inv, mask, (0, hh // 2), 0)
        cv2.floodFill(inv, mask, (ww - 1, hh // 2), 0)
        return inv
    candidates.append(_try_strategy(floodfill_aggressive, "floodfill_aggressive", w, h, px_to_meter,
                                     use_hierarchy=False, min_dim_m=0.10, min_rel_area=0.001))

    best_rooms = []
    best_score = -1

    # Deduplicate candidates by center proximity
    def dedup_rooms(rooms):
        if len(rooms) < 2:
            return rooms
        kept = []
        for r in rooms:
            dup = False
            for k in kept:
                dx = abs(r["centerX"] - k["centerX"])
                dy = abs(r["centerY"] - k["centerY"])
                if dx < 0.5 and dy < 0.5:
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
        if 0.5 < mean_area < 80:
            score += 20
        if std_area < mean_area * 0.8:
            score += 3
        if n >= 2:
            score += 5
        if mean_area > 5:
            score += 10
        if all(a < 300 for a in areas):
            score += 10

        if score > best_score:
            best_score = score
            best_rooms = rooms

    if not best_rooms:
        kernel15 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(cv2.bitwise_not(enhanced), cv2.MORPH_CLOSE, kernel15, iterations=2)
        inv = cv2.bitwise_not(closed)
        hh, ww = inv.shape
        mask = np.zeros((hh + 2, ww + 2), np.uint8)
        cv2.floodFill(inv, mask, (0, 0), 0)
        cv2.floodFill(inv, mask, (ww - 1, 0), 0)
        cv2.floodFill(inv, mask, (0, hh - 1), 0)
        cv2.floodFill(inv, mask, (ww - 1, hh - 1), 0)
        last_rooms = _extract_rooms_from_binary(inv, w, h, px_to_meter,
                                                  use_hierarchy=False,
                                                  min_dim_m=0.10, min_rel_area=0.001,
                                                  max_rel_area=0.95)
        if last_rooms:
            return last_rooms
    return best_rooms

def simplify_polygon(points_px: List[List[float]], epsilon_ratio: float = 0.01) -> List[List[float]]:
    pts = np.array(points_px, dtype=np.int32).reshape(-1, 1, 2)
    perimeter = cv2.arcLength(pts, True)
    epsilon = max(epsilon_ratio * perimeter, 1.0)
    approx = cv2.approxPolyDP(pts, epsilon, True)
    return approx.reshape(-1, 2).tolist()

def polygon_area_px(points_px: List[List[float]]) -> float:
    n = len(points_px)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x1, y1 = points_px[i]
        x2, y2 = points_px[(i + 1) % n]
        area += (x1 * y2) - (x2 * y1)
    return abs(area) / 2.0

def polygon_to_walls(points_m: List[List[float]]) -> List[Dict[str, float]]:
    walls = []
    n = len(points_m)
    for i in range(n):
        x1, y1 = points_m[i]
        x2, y2 = points_m[(i + 1) % n]
        walls.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return walls

def load_sample_rooms_from_cache(target_dim: float = 14.0) -> List[Dict[str, Any]]:
    if not os.path.exists(SAMPLE_ROOMS_PATH):
        return []

    with open(SAMPLE_ROOMS_PATH, "r") as f:
        cached = json.load(f)

    raw_rooms = cached.get("rooms", [])
    if not raw_rooms:
        return []

    min_x = min(r["centerX"] - r["width"] / 2.0 for r in raw_rooms)
    max_x = max(r["centerX"] + r["width"] / 2.0 for r in raw_rooms)
    min_y = min(r["centerY"] - r["height"] / 2.0 for r in raw_rooms)
    max_y = max(r["centerY"] + r["height"] / 2.0 for r in raw_rooms)

    canvas_w = max_x - min_x
    canvas_h = max_y - min_y
    canvas_cx = (min_x + max_x) / 2.0
    canvas_cy = (min_y + max_y) / 2.0

    px_to_meter = max(canvas_w, canvas_h) / target_dim
    if px_to_meter <= 0:
        return []

    rooms_output = []
    for r in raw_rooms:
        raw_points = r.get("polygon_points") or []

        if len(raw_points) >= 3:
            simplified_px = simplify_polygon(raw_points)
            if len(simplified_px) < 3:
                simplified_px = raw_points

            points_m = [
                [(px - canvas_cx) / px_to_meter, (py - canvas_cy) / px_to_meter]
                for px, py in simplified_px
            ]

            area_px = polygon_area_px(simplified_px)
            area_m2 = round(area_px / (px_to_meter ** 2), 2)

            xs = [p[0] for p in points_m]
            ys = [p[1] for p in points_m]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            rw = max(xs) - min(xs)
            rh = max(ys) - min(ys)

            rooms_output.append({
                "label": r.get("label", f"Room {len(rooms_output) + 1}"),
                "dimensions": f"{rw:.1f}m x {rh:.1f}m",
                "centerX": cx,
                "centerY": cy,
                "elevationZ": 0.0,
                "isOpenSpace": False,
                "walls": polygon_to_walls(points_m),
                "area": area_m2
            })
        else:
            cx = (r["centerX"] - canvas_cx) / px_to_meter
            cy = (r["centerY"] - canvas_cy) / px_to_meter
            rw = r["width"] / px_to_meter
            rh = r["height"] / px_to_meter

            rooms_output.append({
                "label": r.get("label", f"Room {len(rooms_output) + 1}"),
                "dimensions": f"{rw:.1f}m x {rh:.1f}m",
                "centerX": cx,
                "centerY": cy,
                "elevationZ": 0.0,
                "isOpenSpace": False,
                "walls": generate_box_walls(cx, cy, rw, rh),
                "area": round(rw * rh, 2)
            })

    return rooms_output

@app.get("/api/v1/process-layout/sample")
async def process_layout_sample(floors: int = Query(1)):
    detected_rooms = load_sample_rooms_from_cache()
    if not detected_rooms:
        raise HTTPException(status_code=404, detail="No cached sample layout available.")

    return {
        "rooms": detected_rooms,
        "totalRooms": len(detected_rooms),
        "totalFloors": floors,
        "calculatedSqFt": round(sum(r["area"] for r in detected_rooms) * 10.764, 1)
    }

@app.post("/api/v1/process-layout/image")
async def process_layout_image(
    file: UploadFile = File(...),
    floors: int = Query(1),
    method: str = Query("auto", description="auto | contour | watershed | json")
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid architectural format stream.")
    try:
        image_bytes = await file.read()
        detected_rooms: List[Dict[str, Any]] = []

        if method == "json":
            detected_rooms = load_sample_rooms_from_cache()
        
        if not detected_rooms and method in ("auto", "contour"):
            detected_rooms = extract_walls_via_contours(image_bytes)

        if not detected_rooms and (method == "watershed" or method == "auto"):
            try:
                watershed_rooms = watershed_pipeline.extract_orthogonal_layout(image_bytes)
                if watershed_rooms:
                    detected_rooms = watershed_rooms
            except Exception:
                pass

        return {
            "rooms": detected_rooms,
            "totalRooms": len(detected_rooms),
            "totalFloors": floors,
            "calculatedSqFt": round(sum(r["area"] for r in detected_rooms) * 10.764, 1)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/process-layout/procedural")
async def process_layout_procedural(payload: ProceduralGenerationPayload):
    if not payload.rooms:
         raise HTTPException(status_code=400, detail="Configuration manifest is completely empty.")

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
            "walls": generate_box_walls(cx, cy, side, side),
            "area": float(room_area_meters)
        })

    return {
        "rooms": rooms_output,
        "totalRooms": len(rooms_output),
        "totalFloors": payload.total_floors,
        "calculatedSqFt": payload.total_sq_ft
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
