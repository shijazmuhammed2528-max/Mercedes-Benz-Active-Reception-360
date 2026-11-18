"""
Motion Detector — Determines if the vehicle is stationary.
Tracks bounding-box center and size across frames.
"""
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)


class MotionDetector:
    """Per-camera motion analyser."""

    def __init__(self, center_threshold: float = 15.0, size_threshold: float = 0.08,
                 stationary_duration: float = 2.0, fps: float = 24.0):
        self.center_threshold = center_threshold
        self.size_threshold = size_threshold
        self.stationary_duration = stationary_duration
        self.fps = fps

        self._prev_center = None
        self._prev_size = None
        self._stationary_start = None
        self._is_stationary = False
        self._state = "WAITING"  # WAITING | MOVING | STOPPING | STATIONARY
        self._stable_frames = 0

    def reset(self):
        self._prev_center = None
        self._prev_size = None
        self._stationary_start = None
        self._is_stationary = False
        self._state = "WAITING"
        self._stable_frames = 0

    def update(self, bbox: list | None) -> str:
        """
        Update with a new detection bbox [x1, y1, x2, y2] or None if no detection.
        Returns state: WAITING | MOVING | STOPPING | STATIONARY
        """
        if bbox is None:
            self._prev_center = None
            self._prev_size = None
            self._stationary_start = None
            self._is_stationary = False
            self._state = "WAITING"
            self._stable_frames = 0
            return self._state

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        center = np.array([cx, cy])
        size = np.array([w, h])

        if self._prev_center is None:
            self._prev_center = center
            self._prev_size = size
            self._state = "MOVING"
            return self._state

        # Calculate movement
        center_dist = np.linalg.norm(center - self._prev_center)
        if self._prev_size[0] > 0 and self._prev_size[1] > 0:
            size_change = max(
                abs(size[0] - self._prev_size[0]) / self._prev_size[0],
                abs(size[1] - self._prev_size[1]) / self._prev_size[1]
            )
        else:
            size_change = 0.0

        self._prev_center = center
        self._prev_size = size

        is_still = (center_dist < self.center_threshold and size_change < self.size_threshold)

        if is_still:
            self._stable_frames += 1
            if self._stationary_start is None:
                self._stationary_start = time.time()
                self._state = "STOPPING"
            else:
                elapsed = time.time() - self._stationary_start
                if elapsed >= self.stationary_duration:
                    self._is_stationary = True
                    self._state = "STATIONARY"
                else:
                    self._state = "STOPPING"
        else:
            self._stationary_start = None
            self._is_stationary = False
            self._stable_frames = 0
            self._state = "MOVING"

        return self._state

    @property
    def is_stationary(self) -> bool:
        return self._is_stationary

    @property
    def state(self) -> str:
        return self._state

    @property
    def stable_frames(self) -> int:
        return self._stable_frames
