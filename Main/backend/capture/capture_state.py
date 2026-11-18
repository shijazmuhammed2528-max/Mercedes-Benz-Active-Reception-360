"""
Capture State — Enum-like state machine for the capture process.
"""

STATES = {
    "INITIALIZING":     "Initializing system...",
    "MONITORING":       "Waiting for vehicle...",
    "VEHICLE_DETECTED": "Vehicle detected",
    "VEHICLE_MOVING":   "Vehicle is moving",
    "VEHICLE_STOPPING": "Vehicle is slowing down",
    "CHECKING_VIEWS":   "Checking all four views...",
    "CAPTURE_READY":    "All views ready — preparing capture",
    "CAPTURING":        "Capturing images...",
    "CAPTURED":         "Images captured successfully",
    "SAVED":            "Session saved to database",
    "ERROR":            "Error occurred",
    "STOPPED":          "Session stopped",
    "PAUSED":           "Session paused",
}


class CaptureState:
    def __init__(self):
        self._state = "INITIALIZING"
        self._message = STATES["INITIALIZING"]

    def set(self, state: str, message: str = None):
        if state in STATES:
            self._state = state
            self._message = message or STATES[state]

    @property
    def state(self) -> str:
        return self._state

    @property
    def message(self) -> str:
        return self._message

    def to_dict(self) -> dict:
        return {"state": self._state, "message": self._message}
