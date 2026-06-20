import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from src.frame_utils import preprocess_camera_frame


class LatestFrameGrabber:
    """
    Background thread that continuously reads from a camera and keeps only
    the most recent frame. Prevents buffer backlog when inference is slow.
    """

    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = preprocess_camera_frame(frame)
            with self._lock:
                self._latest = frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._latest is None:
                return False, None
            return True, self._latest.copy()

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
