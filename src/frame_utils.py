import cv2
import numpy as np

import src.config as config


def preprocess_camera_frame(frame: np.ndarray) -> np.ndarray:
    """Apply camera adjustments and optional downscale before inference."""
    frame = cv2.convertScaleAbs(
        frame,
        alpha=config.CAMERA_CONTRAST,
        beta=config.CAMERA_BRIGHTNESS,
    )

    max_width = config.CAMERA_MAX_WIDTH
    if max_width <= 0:
        return frame

    height, width = frame.shape[:2]
    if width <= max_width:
        return frame

    scale = max_width / width
    new_width = max_width
    new_height = max(1, int(height * scale))
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
