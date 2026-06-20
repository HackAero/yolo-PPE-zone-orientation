import cv2
import os
import sys

sys.modules.setdefault("sounddevice", None)

import numpy as np

from src.face_detection import SharedFaceDetector
from src.pipeline_types import FrameData, TrackedPerson
from src.tattoo import TattooDetector, build_tattoo_rois
import src.config as config


class PrivacyAnonymizer:
    def __init__(self, use_mock: bool = False):
        """
        Provides anonymization by blurring faces and exposed sensitive areas.
        Reuses per-frame face detections populated by SharedFaceDetector.
        """
        self.use_mock = use_mock
        self.face_cache = {}
        self._garfield_bgra = None
        if os.path.exists(config.GARFIELD_IMAGE_PATH):
            self._garfield_bgra = cv2.imread(config.GARFIELD_IMAGE_PATH, cv2.IMREAD_UNCHANGED)

        self.tattoo_detector = None
        if config.BLUR_TATOOS and not use_mock:
            try:
                print(f"[Privacy] Loading local tattoo model: {config.TATTOO_MODEL_PATH}")
                self.tattoo_detector = TattooDetector()
            except Exception as error:
                print(f"[Privacy] Tattoo model unavailable: {error}")

    def _blur_region(self, frame: np.ndarray, xmin: int, ymin: int, xmax: int, ymax: int):
        h, w = frame.shape[:2]
        xmin = max(0, min(xmin, w - 1))
        ymin = max(0, min(ymin, h - 1))
        xmax = max(0, min(xmax, w - 1))
        ymax = max(0, min(ymax, h - 1))

        if xmax <= xmin or ymax <= ymin:
            return False

        roi = frame[ymin:ymax, xmin:xmax]
        kw, kh = config.PRIVACY_BLUR_KERNEL_SIZE
        kw = kw if kw % 2 == 1 else kw + 1
        kh = kh if kh % 2 == 1 else kh + 1

        blurred_roi = cv2.GaussianBlur(roi, (kw, kh), 0)
        frame[ymin:ymax, xmin:xmax] = blurred_roi
        return True

    def _garfield_region(self, frame: np.ndarray, xmin: int, ymin: int, xmax: int, ymax: int):
        h, w = frame.shape[:2]
        xmin = max(0, min(xmin, w - 1))
        ymin = max(0, min(ymin, h - 1))
        xmax = max(0, min(xmax, w - 1))
        ymax = max(0, min(ymax, h - 1))

        if xmax <= xmin or ymax <= ymin:
            return

        # Blur first so any sticker gaps still hide the face.
        self._blur_region(frame, xmin, ymin, xmax, ymax)

        if self._garfield_bgra is None:
            return

        scale = float(config.GARFIELD_SCALE)
        cx = (xmin + xmax) // 2
        cy = (ymin + ymax) // 2
        rw = max(1, int((xmax - xmin) * scale))
        rh = max(1, int((ymax - ymin) * scale))
        sticker = cv2.resize(self._garfield_bgra, (rw, rh), interpolation=cv2.INTER_AREA)

        gxmin = cx - rw // 2
        gymin = cy - rh // 2
        gxmax = gxmin + rw
        gymax = gymin + rh

        dx1 = max(0, gxmin)
        dy1 = max(0, gymin)
        dx2 = min(w, gxmax)
        dy2 = min(h, gymax)
        if dx2 <= dx1 or dy2 <= dy1:
            return

        sx1 = dx1 - gxmin
        sy1 = dy1 - gymin
        sticker_crop = sticker[sy1 : sy1 + (dy2 - dy1), sx1 : sx1 + (dx2 - dx1)]
        roi = frame[dy1:dy2, dx1:dx2]

        if sticker_crop.ndim == 3 and sticker_crop.shape[2] == 4:
            alpha = sticker_crop[:, :, 3:4].astype(np.float32) / 255.0
            rgb = sticker_crop[:, :, :3].astype(np.float32)
            frame[dy1:dy2, dx1:dx2] = (
                alpha * rgb + (1.0 - alpha) * roi.astype(np.float32)
            ).astype(np.uint8)
        else:
            frame[dy1:dy2, dx1:dx2] = sticker_crop[:, :, :3]

    def _censor_region(self, frame: np.ndarray, xmin: int, ymin: int, xmax: int, ymax: int):
        if config.PRIVACY_CENSORSHIP_MODE == "garfield":
            self._garfield_region(frame, xmin, ymin, xmax, ymax)
        else:
            self._blur_region(frame, xmin, ymin, xmax, ymax)

    def _blur_masked_region(self, frame, bbox, mask):
        """Blur only mask-selected pixels inside a full-frame bounding box."""
        xmin, ymin, xmax, ymax = bbox
        h, w = frame.shape[:2]
        xmin = max(0, min(int(xmin), w))
        ymin = max(0, min(int(ymin), h))
        xmax = max(0, min(int(xmax), w))
        ymax = max(0, min(int(ymax), h))

        if xmax <= xmin or ymax <= ymin:
            return False

        roi = frame[ymin:ymax, xmin:xmax]
        if mask.shape != roi.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (roi.shape[1], roi.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        if not np.any(mask):
            return False

        kw, kh = config.PRIVACY_BLUR_KERNEL_SIZE
        kw = kw if kw % 2 == 1 else kw + 1
        kh = kh if kh % 2 == 1 else kh + 1
        blurred_roi = cv2.GaussianBlur(roi, (kw, kh), 0)
        roi[mask] = blurred_roi[mask]
        return True

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
        detection_frame = frame_data.raw_frame
        active_person_ids = set()
        tattoo_blur_regions = 0

        frame_data.extra_metadata["privacy_tattoo_blur_enabled"] = bool(config.BLUR_TATOOS)
        frame_data.extra_metadata["privacy_tattoo_detector_ready"] = bool(self.tattoo_detector is not None)

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
                self._censor_region(
                    frame,
                    fxmin - side_pad,
                    fymin - top_pad,
                    fxmax + side_pad,
                    fymax + bottom_pad,
                )
            # --- Local Tattoo Segmentation & Privacy Redaction ---
            if config.BLUR_TATOOS:
                run_tattoo = getattr(frame_data, "frame_index", 0) % 5 == 0

                for limb_roi in build_tattoo_rois(person, detection_frame.shape):
                    limb_xmin, limb_ymin, limb_xmax, limb_ymax = limb_roi["bbox"]
                    limb_key = f"tattoo_{limb_roi['side']}_{limb_roi['segment']}"

                    try:
                        if self.tattoo_detector is None:
                            raise RuntimeError("Tattoo detector is not initialized")

                        if run_tattoo and not self.use_mock:
                            limb_crop = detection_frame[
                                limb_ymin:limb_ymax,
                                limb_xmin:limb_xmax,
                            ]
                            tattoo_mask = self.tattoo_detector.detect_mask(limb_crop)
                            has_tattoo = bool(np.any(tattoo_mask))
                            person.metadata[limb_key] = has_tattoo
                            
                            if has_tattoo:
                                blurred = self._blur_masked_region(frame, limb_roi["bbox"], tattoo_mask)
                                if blurred:
                                    tattoo_blur_regions += 1
                        else:
                            has_tattoo = person.metadata.get(limb_key, False)
                            if has_tattoo:
                                if self._blur_region(frame, limb_xmin, limb_ymin, limb_xmax, limb_ymax):
                                    tattoo_blur_regions += 1

                    except Exception as error:
                        # Only print errors occasionally to avoid console spam in mock mode
                        if getattr(frame_data, "frame_index", 0) % 30 == 0:
                            print(f"[Privacy] Tattoo inference failed for worker {person.person_id}: {error}")
                        if config.TATTOO_FAIL_CLOSED:
                            if self._blur_region(frame, limb_xmin, limb_ymin, limb_xmax, limb_ymax):
                                tattoo_blur_regions += 1

        for person_id in list(self.face_cache):
            if person_id in active_person_ids:
                continue
            self.face_cache[person_id]["missing_frames"] += 1
            if self.face_cache[person_id]["missing_frames"] > config.PRIVACY_FACE_CACHE_FRAMES:
                del self.face_cache[person_id]

        frame_data.extra_metadata["privacy_tattoo_regions_blurred"] = int(tattoo_blur_regions)

        return frame_data
