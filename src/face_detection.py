import sys

# MediaPipe imports optional audio tasks at package startup; vision-only here.
sys.modules.setdefault("sounddevice", None)

import cv2
import mediapipe as mp
import numpy as np
from typing import Any, Dict, List, Optional

import src.config as config


class SharedFaceDetector:
    """Single MediaPipe face detector shared across compliance and privacy stages."""

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        if not use_mock:
            self.face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.4,
            )
        else:
            self.face_detector = None

        self._frame_counter = 0
        self._last_cache: Dict[int, List[Dict[str, Any]]] = {}

    def detect_in_crop(self, person_crop_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run face detection on a person crop.
        Returns a list of dicts with crop-relative pixel bboxes and optional keypoints.
        """
        if self.face_detector is None or person_crop_bgr.size == 0:
            return []

        rgb_crop = cv2.cvtColor(person_crop_bgr, cv2.COLOR_BGR2RGB)
        detection_result = self.face_detector.process(rgb_crop)
        if not detection_result.detections:
            return []

        crop_h, crop_w = person_crop_bgr.shape[:2]
        faces: List[Dict[str, Any]] = []

        for detection in detection_result.detections:
            box = detection.location_data.relative_bounding_box
            face_xmin = max(0, int(box.xmin * crop_w))
            face_ymin = max(0, int(box.ymin * crop_h))
            face_xmax = min(crop_w, int((box.xmin + box.width) * crop_w))
            face_ymax = min(crop_h, int((box.ymin + box.height) * crop_h))

            if face_xmax <= face_xmin or face_ymax <= face_ymin:
                continue

            face_info: Dict[str, Any] = {
                "face_xmin": face_xmin,
                "face_ymin": face_ymin,
                "face_xmax": face_xmax,
                "face_ymax": face_ymax,
                "face_width": face_xmax - face_xmin,
                "face_height": face_ymax - face_ymin,
                "detection": detection,
            }
            faces.append(face_info)

        return faces

    def populate_frame_cache(self, frame_data) -> None:
        """Detect faces once per person and store on frame_data for reuse."""
        if self.use_mock or frame_data.raw_frame.size == 0:
            frame_data.extra_metadata["person_face_detections"] = {}
            return

        self._frame_counter += 1
        run_detection = (self._frame_counter - 1) % config.FACE_DETECT_INTERVAL == 0

        if not run_detection:
            frame_data.extra_metadata["person_face_detections"] = dict(self._last_cache)
            return

        h_img, w_img = frame_data.raw_frame.shape[:2]
        cache: Dict[int, List[Dict[str, Any]]] = {}

        for person in frame_data.persons:
            xmin, ymin, xmax, ymax = person.bbox
            pxmin = max(0, xmin)
            pymin = max(0, ymin)
            pxmax = min(w_img, xmax)
            pymax = min(h_img, ymax)
            person_crop = frame_data.raw_frame[pymin:pymax, pxmin:pxmax]
            detected = self.detect_in_crop(person_crop)
            if detected:
                cache[person.person_id] = detected
            elif person.person_id in self._last_cache:
                cache[person.person_id] = self._last_cache[person.person_id]

        self._last_cache = cache
        frame_data.extra_metadata["person_face_detections"] = cache

    @staticmethod
    def get_faces(frame_data, person_id: int) -> List[Dict[str, Any]]:
        cache = frame_data.extra_metadata.get("person_face_detections", {})
        return cache.get(person_id, [])
