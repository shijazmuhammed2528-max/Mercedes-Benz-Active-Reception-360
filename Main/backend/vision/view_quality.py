"""
View Quality Validator — Checks whether the vehicle is fully visible in frame.
"""
import logging

logger = logging.getLogger(__name__)


class ViewQualityValidator:
    """Check if a detected vehicle is fully visible with adequate quality."""

    def __init__(self, margin_ratio: float = 0.03, min_area_ratio: float = 0.05,
                 max_area_ratio: float = 0.95):
        self.margin_ratio = margin_ratio
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio

    def validate(self, bbox: list, frame_width: int, frame_height: int) -> dict:
        """
        Validate view quality for a detection.
        Returns: {passed, margin_ok, size_ok, aspect_ok, details}
        """
        if bbox is None:
            return {"passed": False, "reason": "No detection", "margin_ok": False,
                    "size_ok": False, "aspect_ok": False}

        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1

        margin_x = frame_width * self.margin_ratio
        margin_y = frame_height * self.margin_ratio

        # Check margins — vehicle should not touch frame edges
        left_ok = x1 >= margin_x
        top_ok = y1 >= margin_y
        right_ok = x2 <= (frame_width - margin_x)
        bottom_ok = y2 <= (frame_height - margin_y)
        margin_ok = left_ok and top_ok and right_ok and bottom_ok

        # Check size — vehicle should be a reasonable portion of the frame
        area_ratio = (bw * bh) / (frame_width * frame_height)
        size_ok = self.min_area_ratio <= area_ratio <= self.max_area_ratio

        # Check aspect ratio — should be plausible for a vehicle
        if bh > 0:
            aspect = bw / bh
        else:
            aspect = 0
        # Vehicles typically have aspect ratio 0.4 to 4.0 depending on angle
        aspect_ok = 0.3 <= aspect <= 5.0

        passed = margin_ok and size_ok and aspect_ok

        return {
            "passed": passed,
            "margin_ok": margin_ok,
            "size_ok": size_ok,
            "aspect_ok": aspect_ok,
            "area_ratio": round(area_ratio, 4),
            "aspect_ratio": round(aspect, 2),
            "margins": {
                "left": left_ok, "top": top_ok,
                "right": right_ok, "bottom": bottom_ok
            },
        }
