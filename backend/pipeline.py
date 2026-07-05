import cv2
import numpy as np
import math

# Optional Tesseract import for text extraction
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
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
    Advanced Floorplan Parsing Pipeline with Expanded 3D Scene Scaling.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Invalid image format. Could not decode image.")

    height, width = img_bgr.shape[:2]
    
    # 🔥 FIX 1: Broaden scale multiplier (lowering this value scales UP the 3D walls dramatically)
    pixels_per_meter = max(width, height) / 42.0

    # --- 1. WALL ISOLATION & CLEANING ---
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 15, 7
    )

    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    wall_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_clean)

    # --- 2. ROOM SPACE SEGMENTATION ---
    room_spaces = cv2.bitwise_not(wall_mask)
    
    # Smooth away stray layout debris, text lines, and door swings
    kernel_room = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    room_spaces = cv2.morphologyEx(room_spaces, cv2.MORPH_OPEN, kernel_room)

    contours, _ = cv2.findContours(room_spaces, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    rooms = []
    all_walls = []
    img_area = width * height

    if contours:
        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            
            # Skip noise and the root background contour frame
            if area < (img_area * 0.008) or area > (img_area * 0.95):
                continue

            epsilon = 0.015 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            if len(approx) >= 4:
                room_walls = []
                outline_points = []
                
                for i in range(len(approx)):
                    p1 = approx[i][0]
                    p2 = approx[(i + 1) % len(approx)][0]
                    
                    w_data = {
                        "x1": float(p1[0] - width / 2) / pixels_per_meter,
                        "y1": float(p1[1] - height / 2) / pixels_per_meter,
                        "x2": float(p2[0] - width / 2) / pixels_per_meter,
                        "y2": float(p2[1] - height / 2) / pixels_per_meter
                    }
                    room_walls.append(w_data)
                    all_walls.append(w_data)
                    outline_points.append({"x": w_data["x1"], "y": w_data["y1"]})

                xs = [p["x"] for p in outline_points]
                ys = [p["y"] for p in outline_points]
                w_m = max(xs) - min(xs)
                h_m = max(ys) - min(ys)

                rooms.append({
                    "label": f"Room {len(rooms) + 1}",
                    "dimensions": f"{w_m:.1f}m x {h_m:.1f}m",
                    "centerX": float(sum(xs) / len(xs)),
                    "centerY": float(sum(ys) / len(ys)),
                    "walls": room_walls,
                    "outline": outline_points,
                    "area": float(area)
                })

    # --- 3. CHARACTER RECOGNITION (OCR) & CLEAN LABELING ---
    if TESSERACT_AVAILABLE and rooms:
        detected_labels = extract_room_labels(gray, width, height, pixels_per_meter)
        if detected_labels:
            rooms = match_labels_to_rooms(rooms, detected_labels)

    # --- 4. ULTIMATE SAFETY FALLBACK ---
    if not rooms:
        fallback_outline = [
            {"x": -12.0, "y": -9.0}, {"x": 12.0, "y": -9.0},
            {"x": 12.0, "y": 9.0}, {"x": -12.0, "y": 9.0}
        ]
        fallback_walls = [
            {"x1": -12.0, "y1": -9.0, "x2": 12.0, "y2": -9.0},
            {"x1": 12.0, "y1": -9.0, "x2": 12.0, "y2": 9.0},
            {"x1": 12.0, "y1": 9.0, "x2": -12.0, "y2": 9.0},
            {"x1": -12.0, "y1": 9.0, "x2": -12.0, "y2": -9.0}
        ]
        rooms = [{
            "label": "Main Living Space",
            "dimensions": "24.0m x 18.0m",
            "centerX": 0.0,
            "centerY": 0.0,
            "walls": fallback_walls,
            "outline": fallback_outline,
            "area": 432.0
        }]
        all_walls = fallback_walls

    for room in rooms:
        room["all_walls"] = all_walls

    result = {
        "rooms": rooms,
        "labels": [r["label"] for r in rooms if not r["label"].startswith("Room ")],
        "totalRooms": len(rooms)
    }
    
    return to_native(result)


def extract_room_labels(gray, width, height, pixels_per_meter):
    """Parses blueprint tags while omitting small noise parameters."""
    try:
        ocr_img = cv2.GaussianBlur(gray, (3, 3), 0)
        all_labels = []
        
        custom_config = r'--oem 3 --psm 11'
        ocr_data = pytesseract.image_to_data(ocr_img, output_type=pytesseract.Output.DICT, config=custom_config)
        
        # 🔥 FIX 2: Explicit exclusion list for appliance/dimension abbreviations causing clutter
        blacklisted_tags = {"REF", "W/D", "DW", "OVEN", "LINEN", "PANTRY", "KIT", "CLG", "FP"}

        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i].strip().upper()
            conf = int(ocr_data['conf'][i])
            
            if len(text) >= 3 and text.isalpha() and conf > 50:
                if text in blacklisted_tags:
                    continue
                
                tx = ocr_data['left'][i]
                ty = ocr_data['top'][i]
                tw = ocr_data['width'][i]
                th = ocr_data['height'][i]
                
                all_labels.append({
                    "text": text,
                    "x": float(tx + tw/2 - width/2) / pixels_per_meter,
                    "y": float(ty + th/2 - height/2) / pixels_per_meter
                })
        return all_labels
    except Exception:
        return []


def match_labels_to_rooms(rooms, labels):
    """Maps clean text tokens to physical room frames using proximity checks."""
    for label in labels:
        best_room = None
        min_dist = float('inf')
        
        for room in rooms:
            dist = math.hypot(room["centerX"] - label["x"], room["centerY"] - label["y"])
            if dist < min_dist and dist < 8.0:  # Expanded lookahead radius for larger scale
                min_dist = dist
                best_room = room
        
        if best_room:
            best_room["label"] = label["text"]
            
    return rooms