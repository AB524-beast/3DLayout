import cv2
import numpy as np
import pytesseract
import math

def process_blueprint_pipeline(image_bytes: bytes) -> dict:
    """
    Orchestrates the Backend Computer Vision Pipeline:
    Preprocessing -> Structural Line Extraction -> Topology -> Character Recognition
    """
    # Convert incoming payload bytes into an OpenCV matrix array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Invalid source image framework payload.")

    # --- 1. IMAGE PREPROCESSING ---
    # Convert to Grayscale formatting and apply Gaussian smoothing noise attenuation
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive thresholding matrix application to separate foreground blueprint lines
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # --- 2. STRUCTURAL EXTRACTION & LINE DETECTION ---
    # Feature extraction processing using Probabilistic Hough Line Transform
    # Tweaking these parameters isolates sharp continuous structural walls
    min_line_length = 40
    max_line_gap = 10
    lines = cv2.HoughLinesP(
        thresh, rho=1, theta=np.pi/180, threshold=50, 
        minLineLength=min_line_length, maxLineGap=max_line_gap
    )

    extracted_walls = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Convert pixel coordinates to a normalized spatial layout relative to center
            # coordinates evaluation scaling can happen dynamically downstream
            extracted_walls.append({
                "x1": float(x1 - img_bgr.shape[1]/2) / 30, # Simple scale down ratio
                "y1": float(y1 - img_bgr.shape[0]/2) / 30,
                "x2": float(x2 - img_bgr.shape[1]/2) / 30,
                "y2": float(y2 - img_bgr.shape[0]/2) / 30
            })

    # --- 3. CHARACTER RECOGNITION (OCR) & TOPOLOGY ANCHORING ---
    # Tesseract parsing configuration targeting text elements/bounds configurations
    custom_config = r'--oem 3 --psm 11'
    ocr_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=custom_config)
    
    rooms_detected = []
    
    # Loop through isolated string detections to build spatial room labels
    # NOTE: OCR on blueprint images can be noisy; filter down to likely room labels.
    common_room_tokens = {
        "bedroom",
        "bed",
        "bathroom",
        "bath",
        "kitchen",
        "living",
        "livingroom",
        "dining",
        "diningroom",
        "hall",
        "hallway",
        "office",
        "study",
        "laundry",
        "garage",
        "room",
        "restroom",
    }

    for i in range(len(ocr_data['text'])):
        raw_text = ocr_data['text'][i]
        if not raw_text:
            continue

        text = raw_text.strip()
        if not text:
            continue

        # Normalize for matching
        normalized = ''.join(ch for ch in text.lower() if ch.isalnum())

        # Keep only likely room names OR allow alphanumeric tokens longer than 3
        # (but prefer the common list to reduce junk rooms).
        is_room_token = normalized in common_room_tokens
        is_plausible_token = len(text) > 3 and text.isalnum()
        if not (is_room_token or is_plausible_token):
            continue

        tx = ocr_data['left'][i]
        ty = ocr_data['top'][i]
        tw = ocr_data['width'][i]
        th = ocr_data['height'][i]

        # Find bounding center point coordinates of text anchors
        cx = float((tx + tw / 2) - img_bgr.shape[1] / 2) / 30
        cy = float((ty + th / 2) - img_bgr.shape[0] / 2) / 30

        # Match walls by segment midpoint distance to the text anchor.
        # (The previous implementation matched only w['x1'], w['y1'], which is unstable.)
        scored = []
        for w in extracted_walls:
            mx = (w['x1'] + w['x2']) / 2.0
            my = (w['y1'] + w['y2']) / 2.0
            dist = math.hypot(mx - cx, my - cy)
            scored.append((dist, w))

        # Choose up to 4 nearest walls; if OCR is sparse, fall back to extracted_walls[:4]
        local_walls = [w for dist, w in sorted(scored, key=lambda t: t[0]) if dist < 8][:4]
        if not local_walls:
            local_walls = extracted_walls[:4]

        rooms_detected.append({
            "label": text,
            "dimensions": f"{round(abs(tw / 10), 1)}m x {round(abs(th / 10), 1)}m",
            "centerX": cx,
            "centerY": cy,
            "walls": local_walls
        })

    # Fallback default room generation mapping if text reading returns null parameters
    if not rooms_detected:
        rooms_detected.append({
            "label": "Main Studio Hall",
            "dimensions": "8.5m x 6.0m",
            "centerX": 0.0,
            "centerY": 0.0,
            "walls": extracted_walls[:8] if extracted_walls else [
                {"x1": -6, "y1": -4, "x2": 6, "y2": -4},
                {"x1": 6, "y1": -4, "x2": 6, "y2": 4},
                {"x1": 6, "y1": 4, "x2": -6, "y2": 4},
                {"x1": -6, "y1": 4, "x2": -6, "y2": -4}
            ]
        })

    return {"rooms": rooms_detected}