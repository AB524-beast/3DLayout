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
    Uses straight wall line detection + room corner inference.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Invalid image format. Could not decode image.")

    height, width = img_bgr.shape[:2]
    pixels_per_meter = max(width, height) / 20.0

    # --- 1. PREPROCESSING ---
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Simple threshold: walls are dark lines
    _, wall_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    
    # Remove small noise but keep wall lines intact
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    
    # --- 2. EXTRACT STRAIGHT WALL LINES ---
    edges = cv2.Canny(wall_mask, 50, 150)
    
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=40, 
        minLineLength=min(width, height) * 0.03,  # 3% of image
        maxLineGap=min(width, height) * 0.02       # 2% of image
    )

    raw_walls = []
    if lines is not None:
        for line in lines:
            try:
                coords = np.array(line).flatten()
                if len(coords) >= 4:
                    x1, y1, x2, y2 = [int(coords[i]) for i in range(4)]
                    
                    # Only keep mostly horizontal or vertical lines
                    dx = abs(x2 - x1)
                    dy = abs(y2 - y1)
                    
                    if dx > dy * 3 or dy > dx * 3:  # Must be clearly horizontal or vertical
                        raw_walls.append({
                            "x1": float(x1 - width/2) / pixels_per_meter,
                            "y1": float(y1 - height/2) / pixels_per_meter,
                            "x2": float(x2 - width/2) / pixels_per_meter,
                            "y2": float(y2 - height/2) / pixels_per_meter,
                            "is_horizontal": dx > dy * 3
                        })
            except Exception:
                continue

    print(f"Detected {len(raw_walls)} wall segments")

    # --- 3. MERGE WALLS INTO CONTINUOUS LINES ---
    horiz_walls = merge_lines([w for w in raw_walls if w.get("is_horizontal")], is_horizontal=True, tolerance=0.3)
    vert_walls = merge_lines([w for w in raw_walls if not w.get("is_horizontal")], is_horizontal=False, tolerance=0.3)
    
    print(f"Merged: {len(horiz_walls)} horizontal, {len(vert_walls)} vertical walls")

    # --- 4. FIND ROOMS FROM WALL INTERSECTIONS ---
    rooms = find_rectangular_rooms(horiz_walls, vert_walls)
    print(f"Found {len(rooms)} rectangular rooms")

    # --- 5. OCR: EXTRACT ROOM LABELS ---
    if TESSERACT_AVAILABLE and rooms:
        detected_labels = extract_room_labels(gray, width, height, pixels_per_meter)
        if detected_labels:
            print(f"OCR detected {len(detected_labels)} labels")
            rooms = match_labels_to_rooms(rooms, detected_labels)
    
    # --- 6. BUILD GLOBAL WALL LIST FOR 3D ---
    all_walls = []
    for w in horiz_walls:
        all_walls.append({
            "x1": float(w["x1"]), "y1": float(w["y"]),
            "x2": float(w["x2"]), "y2": float(w["y"])
        })
    for w in vert_walls:
        all_walls.append({
            "x1": float(w["x"]), "y1": float(w["y1"]),
            "x2": float(w["x"]), "y2": float(w["y2"])
        })
    
    # --- 7. FALLBACK ---
    if not rooms:
        rooms = create_fallback_room(all_walls)
    
    # Add all walls to each room for proper 3D rendering
    for room in rooms:
        room["all_walls"] = all_walls  # Global walls for 3D scene
    
    result = {
        "rooms": rooms,
        "labels": [r["label"] for r in rooms if r.get("label") and not r["label"].startswith("Room ")],
        "totalRooms": len(rooms)
    }
    
    return to_native(result)


def merge_lines(lines, is_horizontal, tolerance=0.3):
    """Merge collinear lines into continuous wall segments."""
    if not lines:
        return []
    
    # Normalize
    normalized = []
    for w in lines:
        if is_horizontal:
            x1, x2 = sorted([w["x1"], w["x2"]])
            y = (w["y1"] + w["y2"]) / 2
            normalized.append({"x1": x1, "x2": x2, "y": y})
        else:
            y1, y2 = sorted([w["y1"], w["y2"]])
            x = (w["x1"] + w["x2"]) / 2
            normalized.append({"y1": y1, "y2": y2, "x": x})
    
    # Sort by position
    if is_horizontal:
        normalized.sort(key=lambda w: w["y"])
    else:
        normalized.sort(key=lambda w: w["x"])
    
    # Merge groups
    groups = []
    used = set()
    
    for i, w1 in enumerate(normalized):
        if i in used:
            continue
        
        group = [w1]
        used.add(i)
        
        for j, w2 in enumerate(normalized[i+1:], start=i+1):
            if j in used:
                continue
            
            if is_horizontal:
                pos_diff = abs(w1["y"] - w2["y"])
                overlap = not (w1["x2"] < w2["x1"] - tolerance or w2["x2"] < w1["x1"] - tolerance)
            else:
                pos_diff = abs(w1["x"] - w2["x"])
                overlap = not (w1["y2"] < w2["y1"] - tolerance or w2["y2"] < w1["y1"] - tolerance)
            
            if pos_diff < tolerance and overlap:
                group.append(w2)
                used.add(j)
        
        groups.append(group)
    
    # Merge each group
    merged = []
    for group in groups:
        if is_horizontal:
            all_x = []
            all_y = []
            for w in group:
                all_x.extend([w["x1"], w["x2"]])
                all_y.append(w["y"])
            merged.append({
                "x1": float(min(all_x)),
                "x2": float(max(all_x)),
                "y": float(sum(all_y) / len(all_y))
            })
        else:
            all_x = []
            all_y = []
            for w in group:
                all_y.extend([w["y1"], w["y2"]])
                all_x.append(w["x"])
            merged.append({
                "y1": float(min(all_y)),
                "y2": float(max(all_y)),
                "x": float(sum(all_x) / len(all_x))
            })
    
    return merged


def find_rectangular_rooms(horiz_walls, vert_walls):
    """Find rectangular rooms by finding 4-wall intersections."""
    if len(horiz_walls) < 2 or len(vert_walls) < 2:
        return []
    
    rooms = []
    min_room_size = 2.0
    max_room_size = 12.0
    
    horiz_walls.sort(key=lambda w: w["y"])
    vert_walls.sort(key=lambda w: w["x"])
    
    for i, h_top in enumerate(horiz_walls):
        for h_bottom in horiz_walls[i+1:]:
            y_top = h_top["y"]
            y_bottom = h_bottom["y"]
            height_m = y_bottom - y_top
            
            if height_m < min_room_size or height_m > max_room_size:
                continue
            
            # Find vertical walls spanning this y range
            spanning_vert = []
            for v in vert_walls:
                x = v["x"]
                if v["y1"] <= y_top + 0.5 and v["y2"] >= y_bottom - 0.5:
                    if h_top["x1"] - 0.5 <= x <= h_top["x2"] + 0.5:
                        spanning_vert.append(v)
            
            # Find pairs forming rectangles
            for j in range(len(spanning_vert)):
                for k in range(j+1, len(spanning_vert)):
                    v_left = spanning_vert[j]
                    v_right = spanning_vert[k]
                    
                    x_left = v_left["x"]
                    x_right = v_right["x"]
                    width_m = x_right - x_left
                    
                    if width_m < min_room_size or width_m > max_room_size:
                        continue
                    
                    top_spans = h_top["x1"] - 0.5 <= x_left and h_top["x2"] + 0.5 >= x_right
                    bottom_spans = h_bottom["x1"] - 0.5 <= x_left and h_bottom["x2"] + 0.5 >= x_right
                    
                    if top_spans and bottom_spans:
                        room_walls = [
                            {"x1": float(x_left), "y1": float(y_top), "x2": float(x_right), "y2": float(y_top)},
                            {"x1": float(x_right), "y1": float(y_top), "x2": float(x_right), "y2": float(y_bottom)},
                            {"x1": float(x_right), "y1": float(y_bottom), "x2": float(x_left), "y2": float(y_bottom)},
                            {"x1": float(x_left), "y1": float(y_bottom), "x2": float(x_left), "y2": float(y_top)},
                        ]
                        
                        rooms.append({
                            "label": "Room",
                            "dimensions": f"{width_m:.1f}m x {height_m:.1f}m",
                            "centerX": float((x_left + x_right) / 2),
                            "centerY": float((y_top + y_bottom) / 2),
                            "walls": room_walls,
                            "area": float(width_m * height_m)
                        })
    
    # Remove overlapping
    rooms.sort(key=lambda r: r.get("area", 0), reverse=True)
    filtered = []
    for room in rooms:
        is_overlapping = False
        for existing in filtered:
            overlap = calculate_iou(room, existing)
            if overlap > 0.3:
                is_overlapping = True
                break
        if not is_overlapping:
            filtered.append(room)
    
    return filtered


def calculate_iou(room1, room2):
    """Calculate Intersection over Union."""
    x1_min = min(w["x1"] for w in room1["walls"])
    x1_max = max(w["x1"] for w in room1["walls"])
    y1_min = min(w["y1"] for w in room1["walls"])
    y1_max = max(w["y1"] for w in room1["walls"])
    
    x2_min = min(w["x1"] for w in room2["walls"])
    x2_max = max(w["x1"] for w in room2["walls"])
    y2_min = min(w["y1"] for w in room2["walls"])
    y2_max = max(w["y1"] for w in room2["walls"])
    
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    
    if xi_max <= xi_min or yi_max <= yi_min:
        return 0.0
    
    intersection = (xi_max - xi_min) * (yi_max - yi_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def extract_room_labels(gray, width, height, pixels_per_meter):
    """Extract room name labels from blueprint."""
    try:
        ocr_img = cv2.GaussianBlur(gray, (3, 3), 0)
        all_labels = []
        
        for psm in [6, 3, 11]:
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
                
                if "'" in text or '"' in text or '/' in text:
                    continue
                if text.replace(".", "").replace(",", "").replace(" ", "").isdigit():
                    continue
                
                skip_words = {'ft', 'sq', 'in', 'mm', 'cm', 'm', 'x', 'by', 'to', 'of', 'the', 'and', 'or', 'a', 'an'}
                if text_lower in skip_words:
                    continue
                
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
            
            if dist < best_dist and dist < 6.0:
                best_dist = dist
                best_label = label
                best_idx = i
        
        if best_label:
            room["label"] = best_label["text"]
            room["labelConfidence"] = best_label["confidence"]
            used_labels.add(best_idx)
    
    return rooms


def create_fallback_room(walls):
    """Create a fallback room from all walls."""
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
            ],
            "all_walls": []
        }]
    
    return [{
        "label": "Detected Space",
        "dimensions": "Full Blueprint",
        "centerX": 0.0,
        "centerY": 0.0,
        "walls": walls[:20],
        "all_walls": walls
    }]