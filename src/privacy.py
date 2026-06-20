import cv2
import sys

sys.modules.setdefault("sounddevice", None)

import numpy as np

from src.face_detection import SharedFaceDetector
from src.pipeline_types import FrameData, TrackedPerson
import src.config as config


class PrivacyAnonymizer:
    def __init__(self, use_mock: bool = False):
        """
        Provides anonymization by blurring faces and exposed sensitive areas.
        Reuses per-frame face detections populated by SharedFaceDetector.
        """
        self.use_mock = use_mock
        self.face_cache = {}

    def _blur_region(self, frame: np.ndarray, xmin: int, ymin: int, xmax: int, ymax: int):
        h, w = frame.shape[:2]
        xmin = max(0, min(xmin, w - 1))
        ymin = max(0, min(ymin, h - 1))
        xmax = max(0, min(xmax, w - 1))
        ymax = max(0, min(ymax, h - 1))

        if xmax <= xmin or ymax <= ymin:
            return

        roi = frame[ymin:ymax, xmin:xmax]
        kw, kh = config.PRIVACY_BLUR_KERNEL_SIZE
        kw = kw if kw % 2 == 1 else kw + 1
        kh = kh if kh % 2 == 1 else kh + 1

        blurred_roi = cv2.GaussianBlur(roi, (kw, kh), 0)
        frame[ymin:ymax, xmin:xmax] = blurred_roi

    def _to_relative_bbox(self, face_bbox, person_bbox):
        fxmin, fymin, fxmax, fymax = face_bbox
        pxmin, pymin, pxmax, pymax = person_bbox
        person_w = max(1, pxmax - pxmin)
        person_h = max(1, pymax - pymin)
        return np.array([
            (fxmin - pxmin) / person_w,
            (fymin - pymin) / person_h,
            (fxmax - pxmin) / person_w,
            (fymax - pymin) / person_h,
        ], dtype=np.float32)

    def _to_absolute_bbox(self, relative_bbox, person_bbox):
        pxmin, pymin, pxmax, pymax = person_bbox
        person_w = max(1, pxmax - pxmin)
        person_h = max(1, pymax - pymin)
        rxmin, rymin, rxmax, rymax = relative_bbox
        return [
            int(pxmin + rxmin * person_w),
            int(pymin + rymin * person_h),
            int(pxmin + rxmax * person_w),
            int(pymin + rymax * person_h),
        ]

    def _update_face_cache(self, person_id: int, relative_bbox):
        cached = self.face_cache.get(person_id)
        if cached is not None:
            alpha = float(config.PRIVACY_FACE_SMOOTHING_ALPHA)
            alpha = max(0.0, min(alpha, 1.0))
            relative_bbox = alpha * relative_bbox + (1.0 - alpha) * cached["bbox"]

        self.face_cache[person_id] = {
            "bbox": relative_bbox,
            "missing_frames": 0,
        }
        return relative_bbox

    def _get_cached_face(self, person_id: int):
        cached = self.face_cache.get(person_id)
        if cached is None:
            return None

        cached["missing_frames"] += 1
        if cached["missing_frames"] > config.PRIVACY_FACE_CACHE_FRAMES:
            del self.face_cache[person_id]
            return None
        return cached["bbox"]

    def _face_from_cache(self, frame_data: FrameData, person: TrackedPerson):
        """Convert cached crop-relative face detection to a relative person bbox."""
        faces = SharedFaceDetector.get_faces(frame_data, person.person_id)
        if not faces:
            return None

        xmin, ymin, xmax, ymax = person.bbox
        pxmin = max(0, xmin)
        pymin = max(0, ymin)

        best = max(faces, key=lambda face: face["face_width"] * face["face_height"])
        absolute_face_bbox = [
            pxmin + best["face_xmin"],
            pymin + best["face_ymin"],
            pxmin + best["face_xmax"],
            pymin + best["face_ymax"],
        ]
        return self._to_relative_bbox(absolute_face_bbox, person.bbox)

    def process(self, frame_data: FrameData) -> FrameData:
        frame = frame_data.processed_frame
        h_img, w_img = frame.shape[:2]
        active_person_ids = set()

        for person in frame_data.persons:
            xmin, ymin, xmax, ymax = person.bbox
            person_w = xmax - xmin
            person_h = ymax - ymin
            active_person_ids.add(person.person_id)

            if person_w <= 0 or person_h <= 0:
                continue

            relative_face_bbox = None

            if not self.use_mock:
                detected_relative_bbox = self._face_from_cache(frame_data, person)
                if detected_relative_bbox is not None:
                    relative_face_bbox = self._update_face_cache(
                        person.person_id,
                        detected_relative_bbox,
                    )

            if relative_face_bbox is None:
                relative_face_bbox = self._get_cached_face(person.person_id)

            if relative_face_bbox is not None:
                fxmin, fymin, fxmax, fymax = self._to_absolute_bbox(
                    relative_face_bbox,
                    person.bbox,
                )
                face_w = max(1, fxmax - fxmin)
                face_h = max(1, fymax - fymin)
                side_pad = int(face_w * config.PRIVACY_FACE_EXPAND_SIDE)
                top_pad = int(face_h * config.PRIVACY_FACE_EXPAND_TOP)
                bottom_pad = int(face_h * config.PRIVACY_FACE_EXPAND_BOTTOM)
                self._blur_region(
                    frame,
                    fxmin - side_pad,
                    fymin - top_pad,
                    fxmax + side_pad,
                    fymax + bottom_pad,
                )

            if person.metadata.get("has_exposed_tattoo", False):
                arm_ymin = ymin + int(person_h * 0.3)
                arm_ymax = ymin + int(person_h * 0.7)
                self._blur_region(frame, xmin, arm_ymin, xmax, arm_ymax)

        for person_id in list(self.face_cache):
            if person_id in active_person_ids:
                continue
            self.face_cache[person_id]["missing_frames"] += 1
            if self.face_cache[person_id]["missing_frames"] > config.PRIVACY_FACE_CACHE_FRAMES:
                del self.face_cache[person_id]

        return frame_data
