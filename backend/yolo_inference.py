from typing import List, Dict, Any, Optional, Tuple
import logging
import os
import math
import numpy as np
import cv2
from PIL import Image
import io

logger = logging.getLogger(__name__)

YOLO_CLASSES = [
    "Column", "Curtain Wall", "Dimension", "Door",
    "Railing", "Sliding Door", "Stair Case", "Wall", "Window",
]

WALL_CLASSES = {"Wall", "Curtain Wall"}
OPENING_CLASSES = {"Door", "Sliding Door", "Window"}
STRUCTURAL_CLASSES = WALL_CLASSES | {"Column", "Railing", "Stair Case"}

YOLO_MODEL = None
YOLO_LOADED = False


def _load_yolo_model():
    global YOLO_MODEL, YOLO_LOADED
    if YOLO_LOADED:
        return
    YOLO_LOADED = True
    try:
        import onnxruntime as ort
        model_path = os.path.join(
            os.path.dirname(__file__), "models", "best.onnx"
        )
        if not os.path.exists(model_path):
            logger.warning("YOLO ONNX model not found at %s", model_path)
            return
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 2
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        YOLO_MODEL = ort.InferenceSession(
            model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"]
        )
        logger.info("Loaded YOLO floor-plan model from %s", model_path)
    except Exception as e:
        logger.warning("Failed to load YOLO ONNX model: %s", e)
        YOLO_MODEL = None


def is_yolo_available() -> bool:
    _load_yolo_model()
    return YOLO_MODEL is not None


def decode_image_to_cv2(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes from any supported format into a BGR OpenCV array."""
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img = pil_img.convert("RGB")
        rgb = np.array(pil_img)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr
    except Exception:
        pass
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    raise ValueError("Could not decode image bytes. Supported formats: "
                     "PNG, JPEG, BMP, TIFF, WebP, GIF, ICO, PPM, and more.")


def _letterbox(img: np.ndarray, new_shape: int = 640,
               color: Tuple[int, int, int] = (114, 114, 114)
               ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape - new_unpad[0]) / 2
    dh = (new_shape - new_unpad[1]) / 2
    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def _iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-7)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.45
         ) -> List[int]:
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        ious = _iou_batch(boxes[i:i+1], boxes[order[1:]])[0]
        mask = ious <= iou_thresh
        order = order[1:][mask]
    return keep


def _postprocess(output: np.ndarray, conf_thresh: float = 0.25,
                 iou_thresh: float = 0.45
                 ) -> List[Dict[str, Any]]:
    preds = output[0]
    if preds.ndim == 3:
        preds = preds[0]
    boxes_raw = preds[:4, :]
    class_scores = preds[4:, :]
    max_scores = class_scores.max(axis=0)
    keep_mask = max_scores > conf_thresh
    if not keep_mask.any():
        return []
    boxes_raw = boxes_raw[:, keep_mask]
    class_scores = class_scores[:, keep_mask]
    max_scores = max_scores[keep_mask]
    cx, cy, w, h = boxes_raw[0], boxes_raw[1], boxes_raw[2], boxes_raw[3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    cls_ids = class_scores.argmax(axis=0)
    nms_keep = _nms(boxes, max_scores, iou_thresh)
    detections = []
    for idx in nms_keep:
        cls_id = int(cls_ids[idx])
        detections.append({
            "class": YOLO_CLASSES[cls_id] if cls_id < len(YOLO_CLASSES) else f"class_{cls_id}",
            "classId": cls_id,
            "confidence": float(max_scores[idx]),
            "bbox": {
                "x1": float(boxes[idx, 0]),
                "y1": float(boxes[idx, 1]),
                "x2": float(boxes[idx, 2]),
                "y2": float(boxes[idx, 3]),
            },
        })
    return detections


def detect_objects(image_bytes: bytes, conf_thresh: float = 0.10,
                   iou_thresh: float = 0.45, imgsz: int = 640
                   ) -> Optional[List[Dict[str, Any]]]:
    """Run YOLO floor-plan object detection on any image format."""
    _load_yolo_model()
    if YOLO_MODEL is None:
        return None
    try:
        img_bgr = decode_image_to_cv2(image_bytes)
        h_orig, w_orig = img_bgr.shape[:2]
        if w_orig < 16 or h_orig < 16:
            return None
        letterboxed, ratio, (dw, dh) = _letterbox(img_bgr, new_shape=imgsz)
        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0)
        input_name = YOLO_MODEL.get_inputs()[0].name
        output_name = YOLO_MODEL.get_outputs()[0].name
        preds = YOLO_MODEL.run([output_name], {input_name: blob})[0]
        detections = _postprocess(preds, conf_thresh, iou_thresh)
        for det in detections:
            bb = det["bbox"]
            bb["x1"] = max(0.0, min((bb["x1"] - dw) / ratio, float(w_orig)))
            bb["y1"] = max(0.0, min((bb["y1"] - dh) / ratio, float(h_orig)))
            bb["x2"] = max(0.0, min((bb["x2"] - dw) / ratio, float(w_orig)))
            bb["y2"] = max(0.0, min((bb["y2"] - dh) / ratio, float(h_orig)))
            det["centerX"] = (bb["x1"] + bb["x2"]) / 2.0
            det["centerY"] = (bb["y1"] + bb["y2"]) / 2.0
        return detections
    except Exception as e:
        logger.warning("YOLO detection failed: %s", e, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# YOLO-driven wall mask — extracts actual dark wall pixels from within
# each Wall/Curtain Wall detection bbox instead of crude filled rectangles
# ---------------------------------------------------------------------------

def build_wall_mask_from_yolo(gray: np.ndarray,
                               detections: List[Dict[str, Any]],
                               w: int, h: int) -> Optional[np.ndarray]:
    """Build a binary wall mask guided by YOLO wall detections only.

    Only horizontal and vertical structures are kept — diagonal lines
    and noise are excluded by design (Hough lines + YOLO guidance).
    """
    total_px = w * h

    # Step 1: Edge-based H/V wall lines only
    edges = cv2.Canny(gray, 30, 100, apertureSize=3)
    min_line_len = max(15, int(math.sqrt(total_px) * 0.02))
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=25, minLineLength=min_line_len, maxLineGap=20,
    )
    walls = np.zeros((h, w), dtype=np.uint8)
    if lines is not None:
        for line in lines:
            coords = np.asarray(line).reshape(-1)
            if coords.size < 4:
                continue
            x1, y1, x2, y2 = (int(coords[0]), int(coords[1]),
                                int(coords[2]), int(coords[3]))
            angle = math.degrees(math.atan2(abs(y2 - y1), abs(x2 - x1)))
            if not (angle < 8 or angle > 82):
                continue
            cv2.line(walls, (x1, y1), (x2, y2), 255, thickness=4)

    # Step 2: YOLO Wall/Curtain Wall detections → extract actual wall pixels
    wall_dets = [d for d in detections if d["class"] in WALL_CLASSES]
    for det in wall_dets:
        bb = det["bbox"]
        x1 = max(0, int(bb["x1"]))
        y1 = max(0, int(bb["y1"]))
        x2 = min(w, int(bb["x2"]))
        y2 = min(h, int(bb["y2"]))
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        roi = gray[y1:y2, x1:x2]
        local_mean = cv2.blur(roi, (15, 15))
        diff = cv2.subtract(local_mean, roi)
        _, roi_mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
        close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, close_k, iterations=1)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
        roi_h = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, h_kernel, iterations=1)
        roi_v = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, v_kernel, iterations=1)
        roi_mask = cv2.bitwise_or(roi_h, roi_v)
        walls[y1:y2, x1:x2] = cv2.bitwise_or(walls[y1:y2, x1:x2], roi_mask)

    # Step 3: Clean up
    if cv2.countNonZero(walls) < total_px * 0.005:
        return None

    dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    walls = cv2.dilate(walls, dilate_k, iterations=1)
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, close_k, iterations=2)

    return walls


# ---------------------------------------------------------------------------
# YOLO-driven room segmentation — the MAIN pipeline
# ---------------------------------------------------------------------------

def _remove_border_region(rooms_bin: np.ndarray) -> np.ndarray:
    h, w = rooms_bin.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    filled = rooms_bin.copy()
    seeds = [
        (0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1),
        (0, w // 2), (h - 1, w // 2), (h // 2, 0), (h // 2, w - 1),
    ]
    for (sy, sx) in seeds:
        if filled[sy, sx] == 255:
            cv2.floodFill(filled, flood_mask, (sx, sy), 128)
    result = rooms_bin.copy()
    result[filled == 128] = 0
    return result


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
                if abs(p2[1] - p1[1]) > 0.5:
                    snapped[(i + 1) % n][1] = p1[1]
                    changed = True
            elif angle > 90.0 - tol_deg:
                if abs(p2[0] - p1[0]) > 0.5:
                    snapped[(i + 1) % n][0] = p1[0]
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
            if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) < min_len:
                del pts[(i + 1) % len(pts)]
                changed = True
                break
    return np.array(pts, dtype=np.float64)


def _make_valid_simple_polygon(pts: np.ndarray) -> np.ndarray:
    from shapely.geometry import Polygon as ShapelyPolygon
    try:
        poly = ShapelyPolygon(pts)
        if poly.is_valid and poly.area > 0:
            return pts
        repaired = poly.buffer(0)
        if repaired.geom_type == "Polygon" and repaired.area > 0:
            coords = np.array(repaired.exterior.coords[:-1], dtype=np.int32)
            if len(coords) >= 4:
                return coords
        hull = cv2.convexHull(pts.astype(np.int32))
        return hull.reshape(-1, 2)
    except Exception:
        hull = cv2.convexHull(pts.astype(np.int32))
        return hull.reshape(-1, 2)


def _polygon_area(pts):
    n = len(pts)
    area = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _bbox_overlap_ratio(b1, b2) -> float:
    x_overlap = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
    y_overlap = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
    inter = x_overlap * y_overlap
    area1 = max((b1[2] - b1[0]) * (b1[3] - b1[1]), 1e-9)
    area2 = max((b2[2] - b2[0]) * (b2[3] - b2[1]), 1e-9)
    return inter / min(area1, area2)


def _remove_overlapping_rooms(rooms, threshold=0.5):
    if len(rooms) <= 1:
        return rooms
    rooms_sorted = sorted(rooms, key=lambda r: r.get("_px_area", 0), reverse=True)
    kept = []
    for room in rooms_sorted:
        bb = room.get("_bbox", (0, 0, 0, 0))
        dominated = False
        for kr in kept:
            if _bbox_overlap_ratio(bb, kr.get("_bbox", (0, 0, 0, 0))) > threshold:
                dominated = True
                break
        if not dominated:
            kept.append(room)
    return kept


def segment_rooms_from_yolo_walls(walls_mask: np.ndarray, gray: np.ndarray,
                                   w: int, h: int, px_to_meter: float,
                                   min_room_area_px: float = 0,
                                   dist_thresh: float = 0.35
                                   ) -> List[Dict[str, Any]]:
    """Segment rooms from a YOLO-derived wall mask using watershed."""
    if min_room_area_px <= 0:
        min_room_area_px = w * h * 0.008

    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    walls = cv2.morphologyEx(walls_mask, cv2.MORPH_CLOSE, close_k, iterations=2)

    rooms_bin = cv2.bitwise_not(walls)
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    rooms_bin = cv2.morphologyEx(rooms_bin, cv2.MORPH_OPEN, open_k, iterations=1)
    rooms_bin = _remove_border_region(rooms_bin)

    smooth_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    rooms_bin = cv2.morphologyEx(rooms_bin, cv2.MORPH_CLOSE, smooth_k, iterations=1)
    rooms_bin = cv2.morphologyEx(rooms_bin, cv2.MORPH_OPEN, smooth_k, iterations=1)

    dist = cv2.distanceTransform(rooms_bin, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    _, sure_fg = cv2.threshold(dist_norm, dist_thresh, 1.0, cv2.THRESH_BINARY)
    sure_fg = (sure_fg * 255).astype(np.uint8)

    num_markers, markers = cv2.connectedComponents(sure_fg)
    if num_markers <= 1:
        return []

    markers = markers + 1
    unknown = cv2.subtract(rooms_bin, sure_fg)
    markers[unknown == 255] = 0

    img_for_ws = cv2.cvtColor(rooms_bin, cv2.COLOR_GRAY2BGR)
    cv2.watershed(img_for_ws, markers)

    rooms = []
    for label in range(2, markers.max() + 1):
        mask = np.uint8(markers == label) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < min_room_area_px:
            continue

        rect = cv2.minAreaRect(cnt)
        (_, _), (rect_w, rect_h), _ = rect
        rect_area = rect_w * rect_h
        fill_ratio = area / rect_area if rect_area > 0 else 0

        if fill_ratio > 0.72:
            box = cv2.boxPoints(rect)
            approx = box.reshape(-1, 1, 2).astype(np.int32)
        else:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            escalate = 0.02
            while len(approx) > 10 and escalate < 0.08:
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
            snapped = _make_valid_simple_polygon(snapped)
            approx = snapped.reshape(-1, 1, 2)

        xs_px = [int(pt[0][0]) for pt in approx]
        ys_px = [int(pt[0][1]) for pt in approx]
        edge_margin = 3
        if (min(xs_px) <= edge_margin or min(ys_px) <= edge_margin or
                max(xs_px) >= w - edge_margin or max(ys_px) >= h - edge_margin):
            continue

        pts_m = [[(px - w / 2.0) / px_to_meter, (py - h / 2.0) / px_to_meter]
                  for px, py in [pt[0] for pt in approx]]
        xs = [p[0] for p in pts_m]
        ys = [p[1] for p in pts_m]
        bb_w = max(xs) - min(xs)
        bb_h = max(ys) - min(ys)
        if bb_w < 0.8 or bb_h < 0.8 or bb_w > 18 or bb_h > 18:
            continue
        min_side, max_side = min(bb_w, bb_h), max(bb_w, bb_h)
        if max_side > 0 and min_side / max_side < 0.15:
            continue
        img_w_m, img_h_m = w / px_to_meter, h / px_to_meter
        if bb_w > img_w_m * 0.85 and bb_h > img_h_m * 0.85:
            continue

        n = len(pts_m)
        walls_list = []
        for i in range(n):
            x1, y1 = pts_m[i]
            x2, y2 = pts_m[(i + 1) % n]
            walls_list.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

        rooms.append({
            "label": f"Room {len(rooms) + 1}",
            "dimensions": f"{bb_w:.1f}m x {bb_h:.1f}m",
            "centerX": sum(xs) / len(xs),
            "centerY": sum(ys) / len(ys),
            "elevationZ": 0.0,
            "isOpenSpace": False,
            "walls": walls_list,
            "area": round(_polygon_area(pts_m), 2),
            "_px_area": float(area),
            "_bbox": (min(xs_px), min(ys_px), max(xs_px), max(ys_px)),
        })

    rooms = _remove_overlapping_rooms(rooms, threshold=0.5)
    rooms.sort(key=lambda r: r.get("_px_area", 0), reverse=True)
    for r in rooms:
        r.pop("_px_area", None)
        r.pop("_bbox", None)
    return rooms


def get_yolo_room_labels(detections: List[Dict[str, Any]],
                         rooms: List[Dict[str, Any]],
                         w: int, h: int) -> List[Dict[str, Any]]:
    """Enrich room dicts with YOLO-detected object labels."""
    for room in rooms:
        if "walls" not in room or not room["walls"]:
            continue
        walls = room["walls"]
        xs = []
        ys = []
        for wall in walls:
            xs.extend([wall["x1"], wall["x2"]])
            ys.extend([wall["y1"], wall["y2"]])
        if not xs or not ys:
            continue
        room_cx = room.get("centerX", sum(xs) / len(xs))
        room_cy = room.get("centerY", sum(ys) / len(ys))
        half_w = (max(xs) - min(xs)) / 2.0
        half_h = (max(ys) - min(ys)) / 2.0
        room["detectedObjects"] = []
        for det in detections:
            if det["class"] in WALL_CLASSES:
                continue
            cx = det.get("centerX", 0)
            cy = det.get("centerY", 0)
            if abs(cx - room_cx) < max(half_w, 1.0) and \
               abs(cy - room_cy) < max(half_h, 1.0):
                room["detectedObjects"].append(det["class"])
    return rooms
