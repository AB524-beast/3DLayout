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
import logging

from tracing import setup_tracing

logging.basicConfig(level=logging.INFO, format="%(levelname)s\t%(name)s\t%(message)s")

app = FastAPI(title="Orthogonal Blueprint Spatial Modeler")
setup_tracing(app)

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


def _segment_rooms(gray: np.ndarray, w: int, h: int,
                   px_to_meter: float) -> List[Dict[str, Any]]:
    total_px = w * h
    min_room_area_px = total_px * 0.008
    min_wall_area_px = total_px * 0.003

    # ---- Step 1: binary threshold (walls + text = dark) ----
    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ---- Step 2: remove small dark components (text) ----------
    # Text characters are small isolated dark blobs.  Connected-component
    # filtering with a generous size threshold kills them while keeping
    # wall lines (which are long/stretched and belong to larger components
    # or chain into other wall components).
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    walls = np.zeros_like(dark)
    if num_labels > 1:
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_wall_area_px:
                walls[labels == i] = 255

    # If filtering removed everything fall back to the original.
    if cv2.countNonZero(walls) < total_px * 0.01:
        walls = dark

    # ---- Step 3: close gaps in walls ---------------------------
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, close_k, iterations=3)

    # ---- Step 4: thicken walls so narrow passages close --------
    dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    walls = cv2.dilate(walls, dilate_k, iterations=3)

    # ---- Step 5: rooms = white regions = inverted walls --------
    rooms_bin = cv2.bitwise_not(walls)

    # ---- Step 6: find contours of white regions ---------------
    cnts, _ = cv2.findContours(rooms_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rooms = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_room_area_px:
            continue

        epsilon = 0.015 * cv2.arcLength(cnt, True)
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
        if bb_w < 0.8 or bb_h < 0.8 or bb_w > 18 or bb_h > 18:
            continue
        min_side = min(bb_w, bb_h)
        max_side = max(bb_w, bb_h)
        if max_side > 0 and min_side / max_side < 0.15:
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


def extract_walls_via_contours(image_bytes: bytes) -> List[Dict[str, Any]]:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_color is None:
        return []

    h, w = img_color.shape[:2]
    px_to_meter = h / 14.0

    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)

    return _segment_rooms(bilateral, w, h, px_to_meter)


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

        # Pixel-level filtering to discard text-like regions before
        # any further processing.  Text annotations in the neural-net
        # output are typically very small, extremely narrow, or both.
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

        # Text-region filter (pixel level)
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
