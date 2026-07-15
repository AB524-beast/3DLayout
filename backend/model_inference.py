from typing import List, Dict, Any, Optional
import logging
import os
import math
import numpy as np
import cv2

logger = logging.getLogger(__name__)

ROOM_CLASSES = [
    "Background", "Bedroom", "LivingRoom", "Kitchen", "Bathroom",
    "Dining", "Balcony", "Storage", "Hallway", "Other",
]
OPEN_SPACE_CLASSES = {"LivingRoom", "Hallway"}

_MODEL = None
_MODEL_LOADED = False


def _load_model():
    global _MODEL, _MODEL_LOADED
    if _MODEL_LOADED:
        return
    _MODEL_LOADED = True
    try:
        import onnxruntime as ort
        model_path = os.path.join(
            os.path.dirname(__file__), "models", "room_segmenter.onnx"
        )
        if not os.path.exists(model_path):
            logger.warning("ONNX model not found at %s", model_path)
            return
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 2
        _MODEL = ort.InferenceSession(
            model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"]
        )
        logger.info("Loaded room segmenter model from %s", model_path)
    except Exception as e:
        logger.warning("Failed to load ONNX model: %s", e)
        _MODEL = None


def is_model_available() -> bool:
    _load_model()
    return _MODEL is not None


def _snap_orthogonal_strict(pts: np.ndarray, tol_deg: float = 10.0) -> np.ndarray:
    n = len(pts)
    if n < 4:
        return pts.copy().astype(np.int32)
    snapped = pts.copy().astype(np.float64)

    for _ in range(3):
        changed = False
        for i in range(n):
            p1 = snapped[i]
            p2 = snapped[(i + 1) % n]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.hypot(dx, dy)
            if length < 1.0:
                continue
            angle = abs(math.degrees(math.atan2(abs(dy), abs(dx))))
            if angle < tol_deg:
                new_y = p1[1]
                if abs(p2[1] - new_y) > 0.5:
                    snapped[(i + 1) % n][1] = new_y
                    changed = True
            elif angle > 90.0 - tol_deg:
                new_x = p1[0]
                if abs(p2[0] - new_x) > 0.5:
                    snapped[(i + 1) % n][0] = new_x
                    changed = True
        if not changed:
            break

    for i in range(n):
        p1 = snapped[i]
        p2 = snapped[(i + 1) % n]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = abs(math.degrees(math.atan2(abs(dy), abs(dx))))
        if angle < tol_deg:
            snapped[(i + 1) % n][1] = p1[1]
        elif angle > 90.0 - tol_deg:
            snapped[(i + 1) % n][0] = p1[0]

    return snapped.astype(np.int32)


def _collapse_short_edges(pts: np.ndarray, min_len: float) -> np.ndarray:
    pts = pts.astype(np.float64).tolist()
    changed = True
    while changed and len(pts) > 4:
        changed = False
        for i in range(len(pts)):
            p1 = pts[i]
            p2 = pts[(i + 1) % len(pts)]
            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            if dist < min_len:
                del pts[(i + 1) % len(pts)]
                changed = True
                break
    return np.array(pts, dtype=np.float64)


def _polygon_area(pts):
    n = len(pts)
    area = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _refine_mask_morphologically(mask: np.ndarray) -> np.ndarray:
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    refined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=2)

    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, open_k, iterations=1)

    return refined


def segment_rooms_ml(image_bytes: bytes) -> Optional[List[Dict[str, Any]]]:
    _load_model()
    if _MODEL is None:
        return None

    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_color is None:
            return None

        h_orig, w_orig = img_color.shape[:2]
        img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        blob = resized.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0)

        input_name = _MODEL.get_inputs()[0].name
        output_name = _MODEL.get_outputs()[0].name
        preds = _MODEL.run([output_name], {input_name: blob})[0]

        mask_512 = np.argmax(preds[0], axis=0).astype(np.uint8)
        mask_full = cv2.resize(
            mask_512, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST
        )

        px_to_meter = h_orig / 14.0
        num_classes = int(mask_full.max()) + 1
        rooms = []
        for cls in range(1, num_classes):
            cls_mask = np.uint8(mask_full == cls) * 255
            if cv2.countNonZero(cls_mask) == 0:
                continue

            cls_mask = _refine_mask_morphologically(cls_mask)

            cnts, _ = cv2.findContours(
                cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in cnts:
                area_px = cv2.contourArea(cnt)
                if area_px < (w_orig * h_orig) * 0.004:
                    continue

                rect = cv2.minAreaRect(cnt)
                (rect_cx, rect_cy), (rect_w, rect_h), rect_angle = rect
                rect_area = rect_w * rect_h
                fill_ratio = area_px / rect_area if rect_area > 0 else 0

                if fill_ratio > 0.72:
                    box = cv2.boxPoints(rect)
                    approx_pts = box.reshape(-1, 2).astype(np.int32)
                else:
                    epsilon = 0.02 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    escalate = 0.02
                    while len(approx) > 8 and escalate < 0.08:
                        escalate += 0.01
                        approx = cv2.approxPolyDP(cnt, escalate * cv2.arcLength(cnt, True), True)
                    if len(approx) < 4:
                        continue
                    approx_pts = approx.reshape(-1, 2)

                    perim = cv2.arcLength(cnt, True)
                    min_edge_len = max(8.0, perim * 0.02)
                    approx_pts = _collapse_short_edges(approx_pts, min_edge_len)
                    if len(approx_pts) < 4:
                        continue
                    snapped = _snap_orthogonal_strict(approx_pts)
                    approx_pts = snapped

                xs_px = [int(p[0]) for p in approx_pts]
                ys_px = [int(p[1]) for p in approx_pts]

                edge_margin = max(3, int(min(w_orig, h_orig) * 0.005))
                if (min(xs_px) <= edge_margin or min(ys_px) <= edge_margin or
                        max(xs_px) >= w_orig - edge_margin or
                        max(ys_px) >= h_orig - edge_margin):
                    continue

                pts_m = [
                    [
                        (px - w_orig / 2.0) / px_to_meter,
                        (py - h_orig / 2.0) / px_to_meter,
                    ]
                    for px, py in approx_pts.tolist()
                ]
                xs = [p[0] for p in pts_m]
                ys = [p[1] for p in pts_m]
                bb_w = max(xs) - min(xs)
                bb_h = max(ys) - min(ys)
                if bb_w < 0.5 or bb_h < 0.5:
                    continue
                if bb_w > 25 or bb_h > 25:
                    continue
                min_side = min(bb_w, bb_h)
                max_side = max(bb_w, bb_h)
                if max_side > 0 and min_side / max_side < 0.10:
                    continue

                img_w_m = w_orig / px_to_meter
                img_h_m = h_orig / px_to_meter
                if bb_w > img_w_m * 0.90 and bb_h > img_h_m * 0.90:
                    continue

                class_name = ROOM_CLASSES[cls] if cls < len(ROOM_CLASSES) else "Other"

                n = len(pts_m)
                walls = []
                for i in range(n):
                    x1, y1 = pts_m[i]
                    x2, y2 = pts_m[(i + 1) % n]
                    walls.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

                rooms.append({
                    "label": class_name,
                    "dimensions": f"{bb_w:.1f}m x {bb_h:.1f}m",
                    "centerX": sum(xs) / len(xs),
                    "centerY": sum(ys) / len(ys),
                    "elevationZ": 0.0,
                    "isOpenSpace": class_name in OPEN_SPACE_CLASSES,
                    "walls": walls,
                    "area": round(_polygon_area(pts_m), 2),
                    "_px_area": float(area_px),
                })

        rooms.sort(key=lambda r: r.get("_px_area", 0), reverse=True)
        for r in rooms:
            r.pop("_px_area", None)

        return rooms if rooms else None

    except Exception as e:
        logger.warning("ML segmentation failed, will fall back to OpenCV: %s", e)
        return None
