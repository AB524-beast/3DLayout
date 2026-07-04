import cv2
import numpy as np
import math

# Optional Tesseract import
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("WARNING: pytesseract not installed. OCR room labeling disabled.")


def to_native(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_native(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(to_native(v) for v in obj)
    return obj


def process_blueprint_pipeline(image_bytes: bytes) -> dict:
    """
    Transforms 2D blueprint images into structured 3D room layouts.
    Uses wall line detection + room inference for proper rectangular rooms.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Invalid image format. Could not decode image.")

    height, width = img_bgr.shape[:2]
    pixels_per_meter = max(width, height) / 20.0

    # --- 1. PREPROCESSING FOR WALL DETECTION ---
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    _, binary_walls = cv2.threshold(enhanced, 200, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((5, 5), np.uint8)
    binary_walls = cv2.morphologyEx(binary_walls, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # --- 2. OCR: EXTRACT ROOM LABELS ---
    detected_labels = []
    if TESSERACT_AVAILABLE:
        detected_labels = extract_room_labels(gray, width, height, pixels_per_meter)
        print(f"OCR detected {len(detected_labels)} room labels")
        for lbl in detected_labels:
            print(f"  - '{lbl['text']}' at ({lbl['x']:.1f}, {lbl['y']:.1f})")

    # --- 3. DETECT WALL LINES ---
    edges = cv2.Canny(binary_walls, 50, 150)
    
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=40, 
        minLineLength=min(width, height) * 0.03,
        maxLineGap=min(width, height) * 0.05
    )

    raw_walls = []
    if lines is not None:
        for line in lines:
            try:
                coords = np.array(line).flatten()
                if len(coords) >= 4:
                    x1, y1, x2, y2 = [int(coords[i]) for i in range(4)]
                    
                    raw_walls.append({
                        "x1": float(x1 - width/2) / pixels_per_meter,
                        "y1": float(y1 - height/2) / pixels_per_meter,
                        "x2": float(x2 - width/2) / pixels_per_meter,
                        "y2": float(y2 - height/2) / pixels_per_meter,
                        "length": math.hypot(x2-x1, y2-y1)
                    })
            except Exception:
                continue

    print(f"Detected {len(raw_walls)} raw wall segments")

    # --- 4. MERGE WALLS INTO CONTINUOUS LINES ---
    merged_walls = merge_walls_properly(raw_walls, tolerance=0.4)
    print(f"Merged into {len(merged_walls)} wall lines")

    # --- 5. FIND ROOMS FROM WALL INTERSECTIONS ---
    rooms = find_rooms_from_walls(merged_walls, width, height, pixels_per_meter)
    print(f"Found {len(rooms)} rooms")

    # --- 6. MATCH LABELS TO ROOMS ---
    if detected_labels and rooms:
        rooms = match_labels_to_rooms(rooms, detected_labels)
    
    # --- 7. FALLBACK ---
    if not rooms:
        print("Using fallback room generation")
        rooms = create_fallback_from_walls(merged_walls)

    # Build walls for each room
    for room in rooms:
        if not room.get("walls"):
            room["walls"] = generate_room_walls(room)

    result = {
        "rooms": rooms,
        "labels": [r["label"] for r in rooms if r.get("label")],
        "totalRooms": len(rooms)
    }
    
    return to_native(result)


def extract_room_labels(gray, width, height, pixels_per_meter):
    """
    Extract room name labels from blueprint.
    Filters out dimensions and noise, keeps only room names.
    """
    try:
        ocr_img = cv2.GaussianBlur(gray, (3, 3), 0)
        all_labels = []
        
        for psm in [6, 3, 11, 4]:
            custom_config = f'--oem 3 --psm {psm}'
            ocr_data = pytesseract.image_to_data(
                ocr_img, 
                output_type=pytesseract.Output.DICT, 
                config=custom_config
            )
            
            for i in range(len(ocr_data['text'])):
                text = ocr_data['text'][i].strip()
                conf = int(ocr_data['conf'][i])
                
                if not text or len(text) < 2 or conf < 30:
                    continue
                
                text_lower = text.lower()
                
                # Skip pure numbers and dimensions
                if text.replace("'", "").replace('"', "").replace('.', '').replace(',', '').replace(' ', '').isdigit():
                    continue
                
                # Skip dimension patterns like "11'7", "10'8", "23/10", "22/6"
                if "'" in text or '"' in text or '/' in text:
                    continue
                
                # Skip common non-room words
                skip_words = {'ft', 'sq', 'in', 'mm', 'cm', 'm', 'x', 'by', 'to', 'of', 'the', 'and', 'or', 'a', 'an'}
                if text_lower in skip_words:
                    continue
                
                # Skip if mostly numbers
                digit_count = sum(1 for c in text if c.isdigit())
                if digit_count / len(text) > 0.5:
                    continue
                
                tx = ocr_data['left'][i]
                ty = ocr_data['top'][i]
                tw = ocr_data['width'][i]
                th = ocr_data['height'][i]
                
                all_labels.append({
                    "text": text,
                    "x": float(tx + tw/2 - width/2) / pixels_per_meter,
                    "y": float(ty + th/2 - height/2) / pixels_per_meter,
                    "confidence": conf,
                })
        
        # Remove duplicates
        filtered = []
        for lbl in all_labels:
            is_duplicate = False
            for existing in filtered:
                if (lbl['text'].lower() == existing['text'].lower() and
                    math.hypot(lbl['x'] - existing['x'], lbl['y'] - existing['y']) < 2.0):
                    is_duplicate = True
                    if lbl['confidence'] > existing['confidence']:
                        existing['confidence'] = lbl['confidence']
                    break
            if not is_duplicate:
                filtered.append(lbl)
        
        return filtered
    except Exception as e:
        print(f"OCR extraction error: {e}")
        return []


def merge_walls_properly(walls, tolerance=0.4):
    """
    Merge wall segments that are collinear and close together.
    Produces clean continuous wall lines.
    """
    if not walls:
        return walls
    
    merged = []
    used = set()
    
    for i, w1 in enumerate(walls):
        if i in used:
            continue
        
        group = [w1]
        used.add(i)
        
        for j, w2 in enumerate(walls[i+1:], start=i+1):
            if j in used:
                continue
            
            angle1 = math.atan2(w1["y2"] - w1["y1"], w1["x2"] - w1["x1"])
            angle2 = math.atan2(w2["y2"] - w2["y1"], w2["x2"] - w2["x1"])
            
            angle_diff = abs(angle1 - angle2)
            while angle_diff > math.pi / 2:
                angle_diff = math.pi - angle_diff
            
            dx = w1["x2"] - w1["x1"]
            dy = w1["y2"] - w1["y1"]
            line_len = math.hypot(dx, dy)
            if line_len < 0.01:
                continue
            
            mid2_x = (w2["x1"] + w2["x2"]) / 2
            mid2_y = (w2["y1"] + w2["y2"]) / 2
            
            dist = abs(dy * mid2_x - dx * mid2_y + w1["x2"]*w1["y1"] - w1["y2"]*w1["x1"]) / line_len
            
            d1 = math.hypot(w1["x2"] - w2["x1"], w1["y2"] - w2["y1"])
            d2 = math.hypot(w1["x1"] - w2["x2"], w1["y1"] - w2["y2"])
            d3 = math.hypot(w1["x1"] - w2["x1"], w1["y1"] - w2["y1"])
            d4 = math.hypot(w1["x2"] - w2["x2"], w1["y2"] - w2["y2"])
            min_endpoint_dist = min(d1, d2, d3, d4)
            
            if angle_diff < 0.25 and (dist < tolerance or min_endpoint_dist < tolerance * 2):
                group.append(w2)
                used.add(j)
        
        # Find extreme endpoints
        all_x = []
        all_y = []
        for w in group:
            all_x.extend([w["x1"], w["x2"]])
            all_y.extend([w["y1"], w["y2"]])
        
        dx = abs(max(all_x) - min(all_x))
        dy = abs(max(all_y) - min(all_y))
        
        if dx > dy:
            # Horizontal wall
            points = [(w["x1"], w["y1"]) for w in group] + [(w["x2"], w["y2"]) for w in group]
            points.sort(key=lambda p: p[0])
            avg_y = sum(p[1] for p in points) / len(points)
            merged.append({
                "x1": float(points[0][0]), "y1": float(avg_y),
                "x2": float(points[-1][0]), "y2": float(avg_y)
            })
        else:
            # Vertical wall
            points = [(w["x1"], w["y1"]) for w in group] + [(w["x2"], w["y2"]) for w in group]
            points.sort(key=lambda p: p[1])
            avg_x = sum(p[0] for p in points) / len(points)
            merged.append({
                "x1": float(avg_x), "y1": float(points[0][1]),
                "x2": float(avg_x), "y2": float(points[-1][1])
            })
    
    return merged


def find_rooms_from_walls(walls, width, height, pixels_per_meter):
    """
    Find rectangular rooms by analyzing wall intersections.
    Groups walls into horizontal and vertical, then finds enclosed rectangles.
    """
    if not walls or len(walls) < 4:
        return []
    
    # Separate horizontal and vertical walls
    horiz = []
    vert = []
    
    for w in walls:
        dx = abs(w["x2"] - w["x1"])
        dy = abs(w["y2"] - w["y1"])
        
        if dx > dy * 2:  # Mostly horizontal
            y = (w["y1"] + w["y2"]) / 2
            x1, x2 = min(w["x1"], w["x2"]), max(w["x1"], w["x2"])
            horiz.append({"x1": x1, "x2": x2, "y": y})
        elif dy > dx * 2:  # Mostly vertical
            x = (w["x1"] + w["x2"]) / 2
            y1, y2 = min(w["y1"], w["y2"]), max(w["y1"], w["y2"])
            vert.append({"y1": y1, "y2": y2, "x": x})
    
    print(f"  Horizontal walls: {len(horiz)}, Vertical walls: {len(vert)}")
    
    if len(horiz) < 2 or len(vert) < 2:
        return []
    
    # Find rectangular rooms
    rooms = []
    min_room_size = 1.5
    max_room_size = 15.0
    
    for i, h1 in enumerate(horiz):
        for h2 in horiz[i+1:]:
            y1, y2 = sorted([h1["y"], h2["y"]])
            height_m = y2 - y1
            
            if height_m < min_room_size or height_m > max_room_size:
                continue
            
            # Find vertical walls connecting to both horizontals
            matching_vert = []
            for v in vert:
                if v["y1"] <= y1 + 0.5 and v["y2"] >= y2 - 0.5:
                    x = v["x"]
                    if (h1["x1"] - 0.5 <= x <= h1["x2"] + 0.5 and
                        h2["x1"] - 0.5 <= x <= h2["x2"] + 0.5):
                        matching_vert.append(v)
            
            if len(matching_vert) >= 2:
                matching_vert.sort(key=lambda v: v["x"])
                
                for j in range(len(matching_vert) - 1):
                    v1 = matching_vert[j]
                    v2 = matching_vert[j + 1]
                    
                    width_m = v2["x"] - v1["x"]
                    if width_m < min_room_size or width_m > max_room_size:
                        continue
                    
                    # Check if horizontals span between verticals
                    h1_spans = h1["x1"] - 0.5 <= v1["x"] and h1["x2"] + 0.5 >= v2["x"]
                    h2_spans = h2["x1"] - 0.5 <= v1["x"] and h2["x2"] + 0.5 >= v2["x"]
                    
                    if h1_spans and h2_spans:
                        room_walls = [
                            {"x1": float(v1["x"]), "y1": float(y1), "x2": float(v2["x"]), "y2": float(y1)},
                            {"x1": float(v2["x"]), "y1": float(y1), "x2": float(v2["x"]), "y2": float(y2)},
                            {"x1": float(v2["x"]), "y1": float(y2), "x2": float(v1["x"]), "y2": float(y2)},
                            {"x1": float(v1["x"]), "y1": float(y2), "x2": float(v1["x"]), "y2": float(y1)},
                        ]
                        
                        rooms.append({
                            "label": "Room",
                            "dimensions": f"{width_m:.1f}m x {height_m:.1f}m",
                            "centerX": float((v1["x"] + v2["x"]) / 2),
                            "centerY": float((y1 + y2) / 2),
                            "walls": room_walls,
                            "area": float(width_m * height_m)
                        })
    
    # Remove overlapping rooms
    rooms.sort(key=lambda r: r.get("area", 0), reverse=True)
    filtered = []
    for room in rooms:
        is_overlapping = False
        for existing in filtered:
            dist = math.hypot(room["centerX"] - existing["centerX"], 
                            room["centerY"] - existing["centerY"])
            if dist < 1.0:
                is_overlapping = True
                break
        if not is_overlapping:
            filtered.append(room)
    
    return filtered


def match_labels_to_rooms(rooms, labels):
    """Match OCR labels to rooms by proximity."""
    if not labels or not rooms:
        return rooms
    
    labels = sorted(labels, key=lambda l: l['confidence'], reverse=True)
    used_labels = set()
    
    for room in rooms:
        best_label = None
        best_dist = float('inf')
        
        for i, label in enumerate(labels):
            if i in used_labels:
                continue
            
            dist = math.hypot(room["centerX"] - label["x"], room["centerY"] - label["y"])
            
            if dist < best_dist and dist < 5.0:
                best_dist = dist
                best_label = label
                best_idx = i
        
        if best_label:
            room["label"] = best_label["text"]
            room["labelConfidence"] = best_label["confidence"]
            used_labels.add(best_idx)
    
    return rooms


def create_fallback_from_walls(walls):
    """Create a single room from all walls if room detection fails."""
    if not walls:
        return [{
            "label": "Main Living Space",
            "dimensions": "6.0m x 4.5m",
            "centerX": 0.0,
            "centerY": 0.0,
            "walls": [
                {"x1": -6.0, "y1": -4.5, "x2": 6.0, "y2": -4.5},
                {"x1": 6.0, "y1": -4.5, "x2": 6.0, "y2": 4.5},
                {"x1": 6.0, "y1": 4.5, "x2": -6.0, "y2": 4.5},
                {"x1": -6.0, "y1": 4.5, "x2": -6.0, "y2": -4.5}
            ]
        }]
    
    all_x = [w["x1"] for w in walls] + [w["x2"] for w in walls]
    all_y = [w["y1"] for w in walls] + [w["y2"] for w in walls]
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    room_walls = []
    for w in walls:
        room_walls.append({
            "x1": float(w["x1"]), "y1": float(w["y1"]),
            "x2": float(w["x2"]), "y2": float(w["y2"])
        })
    
    return [{
        "label": "Detected Space",
        "dimensions": f"{abs(max_x - min_x):.1f}m x {abs(max_y - min_y):.1f}m",
        "centerX": float((min_x + max_x) / 2),
        "centerY": float((min_y + max_y) / 2),
        "walls": room_walls
    }]


def generate_room_walls(room):
    """Generate walls from room bounding box."""
    cx = float(room.get("centerX", 0))
    cy = float(room.get("centerY", 0))
    
    dims = room.get("dimensions", "6.0m x 4.5m")
    try:
        parts = dims.replace("m", "").split("x")
        w = float(parts[0].strip()) / 2
        h = float(parts[1].strip()) / 2
    except:
        w, h = 3.0, 2.25
    
    return [
        {"x1": float(cx - w), "y1": float(cy - h), "x2": float(cx + w), "y2": float(cy - h)},
        {"x1": float(cx + w), "y1": float(cy - h), "x2": float(cx + w), "y2": float(cy + h)},
        {"x1": float(cx + w), "y1": float(cy + h), "x2": float(cx - w), "y2": float(cy + h)},
        {"x1": float(cx - w), "y1": float(cy + h), "x2": float(cx - w), "y2": float(cy - h)}
    ]