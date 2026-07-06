import cv2
import numpy as np
import open3d as o3d
from scipy import ndimage
from skimage.feature import peak_local_max
from typing import List, Dict, Any

class BlueprintWatershedPipeline:
    def __init__(self, target_canvas_dimension: float = 14.0):
        self.target_dim = target_canvas_dimension

    def _generate_box_walls(self, cx: float, cy: float, width: float, height: float) -> List[Dict[str, float]]:
        x1, x2 = cx - (width / 2.0), cx + (width / 2.0)
        y1, y2 = cy - (height / 2.0), cy + (height / 2.0)
        return [
            {"x1": x1, "y1": y1, "x2": x2, "y2": y1},
            {"x1": x2, "y1": y1, "x2": x2, "y2": y2},
            {"x1": x2, "y1": y2, "x2": x1, "y2": y2},
            {"x1": x1, "y1": y2, "x2": x1, "y2": y1}
        ]

    def extract_orthogonal_layout(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return []

        h, w = img.shape
        rooms_output = []

        # Match FloorPlanCanvas.js exactly: planeHeight is fixed to target_dim
        # and planeWidth = target_dim * aspect, so pixel->meter conversion must
        # be based on image HEIGHT, not the max dimension.
        px_to_meter = float(h) / self.target_dim

        # 1. Binarize and invert (Walls = White, Empty Space = Black)
        _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

        # 2. Thicken the walls aggressively to swallow text, icons, and gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thick_walls = cv2.dilate(binary, kernel, iterations=3)

        # 3. Calculate Distance Transform on the EMPTY space
        empty_space = cv2.bitwise_not(thick_walls)
        distance_map = cv2.distanceTransform(empty_space, cv2.DIST_L2, 5)

        # 4. Find the absolute centers of the rooms (local maxima in the distance map).
        # min_distance is scaled to the image size (rather than a fixed 20px) and
        # a minimum distance-value threshold is enforced so small nooks created by
        # furniture icons, dashed lines, or a printed grid don't each spawn their
        # own basin — this was previously causing real blueprints to be shredded
        # into dozens of tiny fragments instead of whole rooms.
        min_dist_px = max(20, int(min(w, h) * 0.05))
        local_max_coords = peak_local_max(
            distance_map,
            min_distance=min_dist_px,
            labels=empty_space,
            threshold_abs=min_dist_px * 0.6
        )
        local_max_mask = np.zeros(distance_map.shape, dtype=bool)
        if len(local_max_coords) > 0:
            local_max_mask[tuple(local_max_coords.T)] = True

        # 5. Create markers for the Watershed algorithm
        markers, num_features = ndimage.label(local_max_mask)

        # 6. Apply Watershed to segment the image into distinct room basins
        img_color = cv2.cvtColor(thick_walls, cv2.COLOR_GRAY2BGR)
        labels = cv2.watershed(img_color, np.int32(markers))

        # 7. Extract the bounding geometries of each flooded room
        for label_idx in range(1, num_features + 1):  # Skip 0 (background) and -1 (borders)
            mask = np.zeros_like(img, dtype=np.uint8)
            mask[labels == label_idx] = 255

            x_px, y_px, w_px, h_px = cv2.boundingRect(mask)

            area = w_px * h_px
            if area < (w * h * 0.02) or area > (w * h * 0.85):
                continue

            cx = ((x_px + (w_px / 2.0)) - (w / 2.0)) / px_to_meter
            cy = ((y_px + (h_px / 2.0)) - (h / 2.0)) / px_to_meter
            rw = w_px / px_to_meter
            rh = h_px / px_to_meter

            # Discard fragments too small to plausibly be a real room (roughly
            # under 6 sq. meters / ~65 sq. ft) — this is what filters out
            # bathroom-fixture nooks and hallway slivers.
            if rw < 1.8 or rh < 1.8 or (rw * rh) < 5.5:
                continue

            if rw <= self.target_dim * (w / h) and rh <= self.target_dim:
                rooms_output.append({
                    "label": f"Parsed Space {len(rooms_output) + 1}",
                    "dimensions": f"{rw:.1f}m x {rh:.1f}m",
                    "centerX": float(cx),
                    "centerY": float(cy),
                    "elevationZ": 0.0,
                    "isOpenSpace": False,
                    "walls": self._generate_box_walls(cx, cy, rw, rh),
                    "area": round(rw * rh, 2)
                })

        # 8. Open3D Point Cloud Fallback
        if not rooms_output:
            try:
                y_indices, x_indices = np.where(thick_walls > 0)
                if len(x_indices) > 0:
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
                        "label": "Global Structural Layout (O3D)",
                        "dimensions": f"{rw:.1f}m x {rh:.1f}m",
                        "centerX": cx,
                        "centerY": cz,
                        "elevationZ": 0.0,
                        "isOpenSpace": False,
                        "walls": self._generate_box_walls(cx, cz, rw, rh),
                        "area": round(rw * rh, 2)
                    })
            except Exception:
                pass

        # 9. Guaranteed Emergency Fallback
        if not rooms_output:
             rooms_output.append({
                "label": "Default Zone",
                "dimensions": "6.0m x 6.0m",
                "centerX": 0.0,
                "centerY": 0.0,
                "elevationZ": 0.0,
                "isOpenSpace": False,
                "walls": self._generate_box_walls(0.0, 0.0, 6.0, 6.0),
                "area": 36.0
            })

        return rooms_output