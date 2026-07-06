import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
import cv2
import open3d as o3d
import math
import json
import os

from pipeline import BlueprintWatershedPipeline

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

def extract_walls_via_contours(image_bytes: bytes) -> List[Dict[str, Any]]:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []

    h, w = img.shape

    # 1. Clean adaptive thresholding to cleanly isolate interior spaces
    binary = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
    )

    # Structural morphing to bridge minor structural wall breaks or door gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    rooms_output = []

    # 2. CORE STRATEGY: High-Fidelity Axis-Aligned Topology Decomposition
    contours, hierarchy = cv2.findContours(binary_cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for idx, cnt in enumerate(contours):
            if hierarchy[idx][3] == -1 and len(contours) > 1:
                continue

            area_px = cv2.contourArea(cnt)
            if area_px < (w * h * 0.02) or area_px > (w * h * 0.85):
                continue

            rx_px, ry_px, rw_px, rh_px = cv2.boundingRect(cnt)

            px_to_meter = h / 14.0
            cx = ((rx_px + (rw_px / 2.0)) - (w / 2.0)) / px_to_meter
            cy = ((ry_px + (rh_px / 2.0)) - (h / 2.0)) / px_to_meter
            rw = rw_px / px_to_meter
            rh = rh_px / px_to_meter

            if rw < 1.2 or rh < 1.2 or rw > 12.0 or rh > 12.0:
                continue

            label_name = f"Room Space {len(rooms_output) + 1}"
            if len(rooms_output) == 0:
                label_name = "Master Suite"
            elif len(rooms_output) == 1:
                label_name = "Living Room"

            rooms_output.append({
                "label": label_name,
                "dimensions": f"{rw:.1f}m x {rh:.1f}m",
                "centerX": cx,
                "centerY": cy,
                "elevationZ": 0.0,
                "isOpenSpace": False,
                "walls": generate_box_walls(cx, cy, rw, rh),
                "area": round(rw * rh, 2)
            })

    if not rooms_output:
        try:
            y_indices, x_indices = np.where(binary_cleaned > 0)
            if len(x_indices) > 0:
                px_to_meter = h / 14.0
                x_norm = (x_indices - (w / 2.0)) / px_to_meter
                y_norm = (y_indices - (h / 2.0)) / px_to_meter
                pts = np.zeros((len(x_norm), 3))
                pts[:, 0] = x_norm
                pts[:, 2] = y_norm

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts)
                obb = pcd.get_axis_aligned_bounding_box()

                cx, cz = float(obb.get_center()[0]), float(obb.get_center()[2])
                extent = obb.get_extent()
                rw, rh = float(extent[0]), float(extent[2])

                rooms_output.append({
                    "label": "Main Living Area",
                    "dimensions": f"{rw:.1f}m x {rh:.1f}m",
                    "centerX": cx,
                    "centerY": cz,
                    "elevationZ": 0.0,
                    "isOpenSpace": False,
                    "walls": generate_box_walls(cx, cz, rw, rh),
                    "area": round(rw * rh, 2)
                })
        except Exception:
            pass

    if not rooms_output:
        rooms_output.append({
            "label": "Default Living Quarter",
            "dimensions": "7.0m x 7.0m",
            "centerX": 0.0,
            "centerY": 0.0,
            "elevationZ": 0.0,
            "isOpenSpace": False,
            "walls": generate_box_walls(0.0, 0.0, 7.0, 7.0),
            "area": 49.0
        })

    return rooms_output

def simplify_polygon(points_px: List[List[float]], epsilon_ratio: float = 0.01) -> List[List[float]]:
    """
    Reduces the raw pixel-traced polygon (often 50-100+ stair-stepped points
    from raster contour tracing) down to its real structural corners using
    Douglas-Peucker simplification, so we generate one wall segment per real
    wall instead of dozens of tiny sub-pixel fragments.
    """
    pts = np.array(points_px, dtype=np.int32).reshape(-1, 1, 2)
    perimeter = cv2.arcLength(pts, True)
    epsilon = max(epsilon_ratio * perimeter, 1.0)
    approx = cv2.approxPolyDP(pts, epsilon, True)
    return approx.reshape(-1, 2).tolist()

def polygon_area_px(points_px: List[List[float]]) -> float:
    """Shoelace formula — true polygon area, not a bounding-box approximation."""
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
    """Builds one wall segment per polygon edge, tracing the room's real shape
    (notches, bump-outs, non-rectangular corners) instead of a 4-wall box."""
    walls = []
    n = len(points_m)
    for i in range(n):
        x1, y1 = points_m[i]
        x2, y2 = points_m[(i + 1) % n]
        walls.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return walls

def load_sample_rooms_from_cache(target_dim: float = 14.0) -> List[Dict[str, Any]]:
    """
    Uses blueprint_rooms.json as a weighted reference model: rather than
    reducing each room to a bounding-box rectangle, this traces the actual
    polygon_points geometry (simplified to real corners) for every room, so
    the generated 3D walls follow the true recorded shape of each space.
    """
    if not os.path.exists(SAMPLE_ROOMS_PATH):
        return []

    with open(SAMPLE_ROOMS_PATH, "r") as f:
        cached = json.load(f)

    raw_rooms = cached.get("rooms", [])
    if not raw_rooms:
        return []

    # Determine the overall canvas extent (in px) from the union of all rooms
    # so we can normalize every room's position around a shared origin.
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
            # Weighted-model path: trace the real recorded polygon shape.
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
            # Fallback for any malformed entry with no usable polygon: box approximation.
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
    """
    Returns a pre-parsed demo layout loaded from blueprint_rooms.json, using
    its recorded polygon geometry as the room shapes (not bounding boxes).
    """
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
    method: str = Query("auto", description="auto | contour | watershed")
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid architectural format stream.")
    try:
        image_bytes = await file.read()
        detected_rooms: List[Dict[str, Any]] = []

        if method in ("auto", "contour"):
            detected_rooms = extract_walls_via_contours(image_bytes)

        if method == "watershed" or (method == "auto" and not detected_rooms):
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