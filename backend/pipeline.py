import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from typing import List, Dict, Any


class BlueprintWatershedPipeline:
    def __init__(self, target_canvas_dimension: float = 14.0):
        self.target_dim = target_canvas_dimension

    def _polygon_to_walls(self, pts_m):
        walls = []
        n = len(pts_m)
        for i in range(n):
            x1, y1 = pts_m[i]
            x2, y2 = pts_m[(i + 1) % n]
            walls.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
        return walls

    def _watershed_from_mask(self, wall_mask: np.ndarray, img: np.ndarray,
                              w: int, h: int, px_to_meter: float) -> List[Dict[str, Any]]:
        rooms = []
        empty_space = cv2.bitwise_not(wall_mask)
        distance_map = cv2.distanceTransform(empty_space, cv2.DIST_L2, 5)

        min_dist_px = max(12, int(min(w, h) * 0.03))
        local_max_coords = peak_local_max(
            distance_map,
            min_distance=min_dist_px,
            labels=empty_space,
            threshold_abs=min_dist_px * 0.3,
            exclude_border=3
        )

        if len(local_max_coords) < 1:
            return rooms

        local_max_mask = np.zeros(distance_map.shape, dtype=bool)
        local_max_mask[tuple(local_max_coords.T)] = True

        markers, num_features = ndimage.label(local_max_mask)
        if num_features < 1:
            return rooms

        img_color = cv2.cvtColor(wall_mask, cv2.COLOR_GRAY2BGR)
        labels = cv2.watershed(img_color, np.int32(markers))

        for label_idx in range(1, num_features + 1):
            mask = np.zeros_like(img, dtype=np.uint8)
            mask[labels == label_idx] = 255

            mask_smooth = cv2.GaussianBlur(mask, (3, 3), 0)
            _, mask_thresh = cv2.threshold(mask_smooth, 127, 255, 0)
            room_contours, _ = cv2.findContours(mask_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not room_contours:
                continue

            cnt = room_contours[0]
            area_px = cv2.contourArea(cnt)
            if area_px < (w * h * 0.001) or area_px > (w * h * 0.85):
                continue

            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx) < 3:
                continue

            pts_m = []
            for point in approx:
                px, py = point[0]
                mx = (px - w / 2.0) / px_to_meter
                my = (py - h / 2.0) / px_to_meter
                pts_m.append([float(mx), float(my)])

            xs = [p[0] for p in pts_m]
            ys = [p[1] for p in pts_m]
            bb_w = max(xs) - min(xs)
            bb_h = max(ys) - min(ys)

            if bb_w < 0.15 or bb_h < 0.15 or bb_w > 18.0 or bb_h > 18.0:
                continue

            rooms.append({
                "label": f"Parsed Space {len(rooms) + 1}",
                "dimensions": f"{bb_w:.1f}m x {bb_h:.1f}m",
                "centerX": float(sum(xs) / len(xs)),
                "centerY": float(sum(ys) / len(ys)),
                "elevationZ": 0.0,
                "isOpenSpace": False,
                "walls": self._polygon_to_walls(pts_m),
                "area": round(bb_w * bb_h, 2)
            })

        return rooms

    def extract_orthogonal_layout(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_color is None:
            return []

        h, w = img_color.shape[:2]
        px_to_meter = float(h) / self.target_dim

        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
        _, _, value = cv2.split(hsv)
        enhanced_hsv = clahe.apply(value)

        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        blurred_hsv = cv2.GaussianBlur(enhanced_hsv, (5, 5), 0)
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        candidates = []

        # Strategy W1: Otsu threshold wall mask
        def w_otsu():
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel5, iterations=3)
        candidates.append(self._watershed_from_mask(w_otsu(), img_color, w, h, px_to_meter))

        # Strategy W2: Otsu on HSV value
        def w_otsu_hsv():
            _, binary = cv2.threshold(blurred_hsv, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel5, iterations=3)
        candidates.append(self._watershed_from_mask(w_otsu_hsv(), img_color, w, h, px_to_meter))

        # Strategy W3: Adaptive threshold
        def w_adaptive():
            binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 15, 4)
            return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel5, iterations=2)
        candidates.append(self._watershed_from_mask(w_adaptive(), img_color, w, h, px_to_meter))

        # Strategy W4: Canny edge wall mask
        def w_canny():
            med = np.median(blurred)
            lower = int(max(0, 0.3 * med))
            upper = int(min(255, 1.2 * med))
            edges = cv2.Canny(blurred, lower, upper, apertureSize=3)
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel5, iterations=2)
            return cv2.dilate(closed, kernel3, iterations=2)
        candidates.append(self._watershed_from_mask(w_canny(), img_color, w, h, px_to_meter))

        # Pick best candidate
        best_rooms = []
        best_score = -1
        for rooms in candidates:
            if not rooms:
                continue
            n = len(rooms)
            areas = [r["area"] for r in rooms]
            mean_area = sum(areas) / n
            score = min(n, 20) * 10
            if 0.5 < mean_area < 80:
                score += 20
            if n >= 2:
                score += 5
            if score > best_score:
                best_score = score
                best_rooms = rooms

        return best_rooms