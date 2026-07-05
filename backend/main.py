import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pipeline import process_blueprint_pipeline
import math

app = FastAPI(title="Unified Blueprint Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def generate_procedural_layout(num_rooms: int, sq_ft: float, floors: int) -> dict:
    """Procedurally slices and builds clean 3D stacked vectors matching specs."""
    # Convert sq ft to total combined structural meters over all floors
    total_area_meters = (sq_ft / 10.764) / floors
    
    # Target 4:3 architectural bounding aspect ratio
    side_y = math.sqrt(total_area_meters / 1.333)
    side_x = side_y * 1.333
    
    half_x = side_x / 2.0
    half_y = side_y / 2.0
    
    rooms_per_floor = math.ceil(num_rooms / floors)
    cols = math.ceil(math.sqrt(rooms_per_floor))
    rows = math.ceil(rooms_per_floor / cols)
    
    room_w = side_x / cols
    room_h = side_y / rows
    
    rooms = []
    global_walls = []
    
    total_assigned = 0
    for f in range(floors):
        floor_z = f * 3.0
        floor_room_count = 0
        
        for r in range(rows):
            for c in range(cols):
                if total_assigned >= num_rooms:
                    break
                if floor_room_count >= rooms_per_floor:
                    break
                
                rx1 = -half_x + (c * room_w)
                ry1 = -half_y + (r * room_h)
                rx2 = rx1 + room_w
                ry2 = ry1 + room_h
                
                room_walls = [
                    {"x1": rx1, "y1": ry1, "x2": rx2, "y2": ry1},
                    {"x1": rx2, "y1": ry1, "x2": rx2, "y2": ry2},
                    {"x1": rx2, "y1": ry2, "x2": rx1, "y2": ry2},
                    {"x1": rx1, "y1": ry2, "x2": rx1, "y2": ry1}
                ]
                
                global_walls.extend(room_walls)
                rooms.append({
                    "label": f"Floor {f+1} - Room {total_assigned + 1}",
                    "dimensions": f"{room_w:.1f}m x {room_h:.1f}m",
                    "centerX": float((rx1 + rx2) / 2.0),
                    "centerY": float((ry1 + ry2) / 2.0),
                    "elevationZ": floor_z,
                    "walls": room_walls,
                    "outline": [{"x": rx1, "y": ry1}, {"x": rx2, "y": ry1}, {"x": rx2, "y": ry2}, {"x": rx1, "y": ry2}],
                    "area": float(room_w * room_h)
                })
                
                floor_room_count += 1
                total_assigned += 1

    for rm in rooms:
        rm["all_walls"] = global_walls

    return {
        "rooms": rooms,
        "labels": [r["label"] for r in rooms],
        "totalRooms": len(rooms),
        "totalFloors": floors,
        "calculatedSqFt": sq_ft
    }


@app.post("/api/v1/process-layout")
async def process_layout(
    file: Optional[UploadFile] = File(None),
    num_rooms: int = Query(3, alias="num_rooms"),
    sq_ft: float = Query(1200.0, alias="sq_ft"),
    floors: int = Query(1, alias="floors")
):
    if file is not None:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid image layout structure.")
        try:
            image_bytes = await file.read()
            return process_blueprint_pipeline(image_bytes, floors=floors)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
            
    return generate_procedural_layout(num_rooms, sq_ft, floors)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)