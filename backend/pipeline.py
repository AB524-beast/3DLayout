import cv2
import numpy as np
import math

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


def snap_to_ortho(p1, p2, threshold_deg=5.0):
    """Straightens slightly skewed drawing/scanning variations into clean 90-degree lines."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    angle = math.degrees(math.atan2(abs(dy), abs(dx)))
    
    if angle <= threshold_deg:
        mean_y = (y1 + y2) / 2.0
        return (x1, mean_y), (x2, mean_y)
    elif angle >= (90.0 - threshold_deg):
        mean_x = (x1 + x2) / 2.0
        return (mean_x, y1), (mean_x, y2)
    return p1, p2


def process_blueprint_pipeline(image_bytes: bytes, floors: int = 1) -> dict:
    """
    Transforms any 2D blueprint image map into a structured model oriented
    perfectly for an immediate flat ortho top-down view, duplicated across N floors.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Invalid image format. Could not decode image.")

    height, width = img_bgr.shape[:2]
    img_area = width * height

    # --- 1. ENSEMBLE PREPROCESSING ---
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    thresh_fine = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
    thresh_coarse = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 9)
    wall_mask = cv2.bitwise_or(thresh_fine, thresh_coarse)
    
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel_clean)

    room_spaces = cv2.bitwise_not(wall_mask)
    kernel_room = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    room_spaces = cv2.morphologyEx(room_spaces, cv2.MORPH_OPEN, kernel_room)

    # --- 2. UNIVERSAL TOPOLOGY ROOM SEGMENTATION ---
    contours, _ = cv2.findContours(room_spaces, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    base_rooms = []
    all_walls = []
    
    min_x_px, max_x_px = float('inf'), float('-inf')
    min_y_px, max_y_px = float('inf'), float('-inf')
    
    raw_room_data = []

    if contours:
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (img_area * 0.003) or area > (img_area * 0.92):
                continue

            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            if len(approx) >= 4:
                room_pixel_walls = []
                outline_pixel_pts = []
                
                for i in range(len(approx)):
                    p1_raw = approx[i][0]
                    p2_raw = approx[(i + 1) % len(approx)][0]
                    p1_s, p2_s = snap_to_ortho(p1_raw, p2_raw)
                    
                    w_data = {"x1": float(p1_s[0]), "y1": float(p1_s[1]), "x2": float(p2_s[0]), "y2": float(p2_s[1])}
                    room_pixel_walls.append(w_data)
                    outline_pixel_pts.append({"x": w_data["x1"], "y": w_data["y1"]})
                    
                    min_x_px = min(min_x_px, w_data["x1"], w_data["x2"])
                    max_x_px = max(max_x_px, w_data["x1"], w_data["x2"])
                    min_y_px = min(min_y_px, w_data["y1"], w_data["y2"])
                    max_y_px = max(max_y_px, w_data["y1"], w_data["y2"])

                raw_room_data.append({"walls": room_pixel_walls, "outline": outline_pixel_pts, "area_px": area})

    # --- 3. DYNAMIC METRIC VIEWPORT CALIBRATION ---
    bbox_w = max_x_px - min_x_px if max_x_px > min_x_px else width
    bbox_h = max_y_px - min_y_px if max_y_px > min_y_px else height
    pixels_per_meter = max(bbox_w, bbox_h) / 24.0

    for idx, r_data in enumerate(raw_room_data):
        metric_walls = []
        metric_outline = []
        
        for w in r_data["walls"]:
            w_m = {
                "x1": float(w["x1"] - width / 2) / pixels_per_meter,
                "y1": float(w["y1"] - height / 2) / pixels_per_meter,
                "x2": float(w["x2"] - width / 2) / pixels_per_meter,
                "y2": float(w["y2"] - height / 2) / pixels_per_meter
            }
            metric_walls.append(w_m)
            all_walls.append(w_m)
            metric_outline.append({"x": w_m["x1"], "y": w_m["y1"]})

        xs = [p["x"] for p in metric_outline]
        ys = [p["y"] for p in metric_outline]

        base_rooms.append({
            "label": f"Room {idx + 1}",
            "dimensions": f"{(max(xs)-min(xs)):.1f}m x {(max(ys)-min(ys)):.1f}m",
            "centerX": float(sum(xs) / len(xs)),
            "centerY": float(sum(ys) / len(ys)),
            "walls": metric_walls,
            "outline": metric_outline,
            "area": float(r_data["area_px"] / (pixels_per_meter ** 2))
        })

    if TESSERACT_AVAILABLE and base_rooms:
        detected_labels = extract_room_labels(gray, width, height, pixels_per_meter)
        if detected_labels:
            base_rooms = match_labels_to_rooms(base_rooms, detected_labels)

    if not base_rooms:
        base_rooms = create_fallback_room(all_walls, wall_mask, width, height, pixels_per_meter)

    # --- 4. MULTI-FLOOR LAYER GENERATION ---
    final_multi_floor_rooms = []
    final_global_walls = []

    for f in range(floors):
        floor_z = f * 3.0  # Stacking elevation (3 meters per floor)
        for room in base_rooms:
            # Deep copy room structures to prevent shared reference bugs
            copied_room = {
                **room,
                "label": f"F{f+1} - {room['label']}" if floors > 1 else room['label'],
                "elevationZ": floor_z,
                "walls": [{**w} for w in room["walls"]],
                "outline": [{**p} for p in room["outline"]]
            }
            final_multi_floor_rooms.append(copied_room)
            final_global_walls.extend(copied_room["walls"])

    for room in final_multi_floor_rooms:
        room["all_walls"] = final_global_walls

    return to_native({
        "rooms": final_multi_floor_rooms,
        "labels": [r["label"] for r in final_multi_floor_rooms if r.get("label") and not r["label"].startswith("Room ")],
        "totalRooms": len(final_multi_floor_rooms),
        "totalFloors": floors
    })


def extract_room_labels(gray, width, height, pixels_per_meter):
    """Extract room name labels from blueprint using OCR."""
    try:
        ocr_img = cv2.GaussianBlur(gray, (3, 3), 0)
        all_labels = []
        for psm in [6, 3, 11]:
            custom_config = f'--oem 3 --psm {psm}'
            ocr_data = pytesseract.image_to_data(ocr_img, output_type=pytesseract.Output.DICT, config=custom_config)
            for i in range(len(ocr_data['text'])):
                text = ocr_data['text'][i].strip()
                conf = int(ocr_data['conf'][i])
                if not text or len(text) < 2 or conf < 30:
                    continue
                if "'" in text or '"' in text or '/' in text:
                    continue
                if text.replace(".", "").replace(",", "").replace(" ", "").isdigit():
                    continue
                skip_words = {'ft', 'sq', 'in', 'mm', 'cm', 'm', 'x', 'by', 'to', 'of', 'the', 'and', 'or', 'a', 'an'}
                if text.lower() in skip_words:
                    continue
                if sum(1 for c in text if c.isdigit()) / len(text) > 0.5:
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


def build_outline_from_walls(walls):
    return [{"x": float(w["x1"]), "y": float(w["y1"])} for w in walls]


def create_fallback_room(walls, wall_mask=None, width=None, height=None, pixels_per_meter=None):
    default_box = [{
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
        "outline": [
            {"x": -6.0, "y": -4.5}, {"x": 6.0, "y": -4.5},
            {"x": 6.0, "y": 4.5}, {"x": -6.0, "y": 4.5}
        ],
        "all_walls": []
    }]
    if not walls:
        return default_box
    points = []
    for w in walls:
        points.append((w["x1"], w["y1"]))
        points.append((w["x2"], w["y2"]))
    pts_arr = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(pts_arr).reshape(-1, 2)
    if len(hull) < 3:
        return default_box
    outline = [{"x": float(x), "y": float(y)} for x, y in hull]
    hull_walls = []
    for i in range(len(outline)):
        p1 = outline[i]
        p2 = outline[(i + 1) % len(outline)]
        hull_walls.append({"x1": p1["x"], "y1": p1["y"], "x2": p2["x"], "y2": p2["y"]})
    xs = [p["x"] for p in outline]
    ys = [p["y"] for p in outline]
    return [{
        "label": "Detected Space",
        "dimensions": "Full Layout",
        "centerX": float(sum(xs) / len(xs)),
        "centerY": float(sum(ys) / len(ys)),
        "walls": hull_walls,
        "outline": outline,
        "all_walls": walls
    }]