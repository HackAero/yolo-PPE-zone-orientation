import cv2
import sys

# MediaPipe imports its optional audio tasks at package startup. This project
# only uses vision, so prevent sounddevice from initializing PortAudio and
# blocking camera startup on macOS.
sys.modules.setdefault("sounddevice", None)

import mediapipe as mp
import numpy as np
from src.types import FrameData
import src.config as config

class PrivacyAnonymizer:
    def __init__(self, use_mock: bool = False):
        """
        Provides anonymization by blurring faces and exposed sensitive areas.
        """
        self.use_mock = use_mock
        # Short-lived face locations keyed only by anonymous person ID. Bounding
        # boxes are stored relative to the person box, never as face images.
        self.face_cache = {}
        if not use_mock:
            print("[Privacy] Loading MediaPipe face detector.")
            self.face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5,
            )
        else:
            print("[Privacy] Running in MOCK mode.")
            self.face_detector = None

    def _blur_region(self, frame: np.ndarray, xmin: int, ymin: int, xmax: int, ymax: int):
        """
        Blurs a specific rectangular region in the frame.
        """
        h, w = frame.shape[:2]
        # Constrain coordinates to image frame boundaries
        xmin = max(0, min(xmin, w - 1))
        ymin = max(0, min(ymin, h - 1))
        xmax = max(0, min(xmax, w - 1))
        ymax = max(0, min(ymax, h - 1))
        
        if xmax <= xmin or ymax <= ymin:
            return
            
        roi = frame[ymin:ymax, xmin:xmax]
        # Gaussian blur requires odd numbers for kernel width/height
        kw, kh = config.PRIVACY_BLUR_KERNEL_SIZE
        kw = kw if kw % 2 == 1 else kw + 1
        kh = kh if kh % 2 == 1 else kh + 1
        
        blurred_roi = cv2.GaussianBlur(roi, (kw, kh), 0)
        frame[ymin:ymax, xmin:xmax] = blurred_roi

    def _to_relative_bbox(self, face_bbox, person_bbox):
        """Convert an absolute face box to person-relative coordinates."""
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
        """Project a person-relative face box into the current frame."""
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
        """Smooth a detected face location and reset its missing-frame count."""
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
        """Return a recent face location during a short detector dropout."""
        cached = self.face_cache.get(person_id)
        if cached is None:
            return None

        cached["missing_frames"] += 1
        if cached["missing_frames"] > config.PRIVACY_FACE_CACHE_FRAMES:
            del self.face_cache[person_id]
            return None
        return cached["bbox"]

    def process(self, frame_data: FrameData) -> FrameData:
        """
        Conveyor belt stage:
        Iterates over detected people, detects their faces, and blurs them on processed_frame.
        """
        # Note: In both mock and real mode, we run on frame_data.persons
        frame = frame_data.processed_frame
        detection_frame = frame_data.raw_frame
        h_img, w_img = frame.shape[:2]
        active_person_ids = set()

        for person in frame_data.persons:
            xmin, ymin, xmax, ymax = person.bbox
            person_w = xmax - xmin
            person_h = ymax - ymin
            active_person_ids.add(person.person_id)

            if person_w <= 0 or person_h <= 0:
                continue

            # --- Face Detection & Blurring ---
            relative_face_bbox = None
            
            if self.face_detector is not None and not self.use_mock:
                # Crop person region for face detection (makes it faster and less false-positive prone)
                pxmin = max(0, xmin)
                pymin = max(0, ymin)
                pxmax = min(w_img, xmax)
                pymax = min(h_img, ymax)
                
                # Detection uses the untouched frame, while anonymization is
                # applied only to processed_frame.
                person_crop = detection_frame[pymin:pymax, pxmin:pxmax]
                if person_crop.size > 0:
                    rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                    detection_result = self.face_detector.process(rgb_crop)
                    faces = []

                    if detection_result.detections:
                        crop_h, crop_w = person_crop.shape[:2]
                        for detection in detection_result.detections:
                            box = detection.location_data.relative_bounding_box

                            # MediaPipe returns normalized coordinates which may
                            # extend slightly outside the image at frame edges.
                            face_xmin = max(0, int(box.xmin * crop_w))
                            face_ymin = max(0, int(box.ymin * crop_h))
                            face_xmax = min(crop_w, int((box.xmin + box.width) * crop_w))
                            face_ymax = min(crop_h, int((box.ymin + box.height) * crop_h))
                            face_w = face_xmax - face_xmin
                            face_h = face_ymax - face_ymin

                            if face_w > 0 and face_h > 0:
                                faces.append((face_xmin, face_ymin, face_w, face_h))
                    
                    if len(faces) > 0:
                        # A person crop should normally contain one face. Choosing
                        # the largest candidate suppresses small background matches.
                        fx, fy, fw, fh = max(faces, key=lambda face: face[2] * face[3])
                        absolute_face_bbox = [
                            pxmin + fx,
                            pymin + fy,
                            pxmin + fx + fw,
                            pymin + fy + fh,
                        ]
                        detected_relative_bbox = self._to_relative_bbox(
                            absolute_face_bbox,
                            person.bbox,
                        )
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
            
            # --- Tattoos & Exposed Skin Blurring (Optional Hackathon Stage) ---
            # If someone has exposed arms where tattoos might be, apply a light blur.
            # We look at the middle height of the bounding box (arms region).
            # Tattoos can be toggled via config or metadata.
            if person.metadata.get("has_exposed_tattoo", False):
                # Blur arm region (heuristic: middle third of person height)
                arm_ymin = ymin + int(person_h * 0.3)
                arm_ymax = ymin + int(person_h * 0.7)
                self._blur_region(frame, xmin, arm_ymin, xmax, arm_ymax)

        # Age cache entries even when a tracked person briefly disappears.
        for person_id in list(self.face_cache):
            if person_id in active_person_ids:
                continue
            self.face_cache[person_id]["missing_frames"] += 1
            if self.face_cache[person_id]["missing_frames"] > config.PRIVACY_FACE_CACHE_FRAMES:
                del self.face_cache[person_id]

        return frame_data

if __name__ == "__main__":
    print("Testing PrivacyStage in Isolation...")
    import time
    
    # Create test data
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(dummy_frame, (200, 100, 300, 400), (255, 255, 255), -1) # dummy white box for person
    
    anonymizer = PrivacyAnonymizer(use_mock=True)
    person = TrackedPerson(person_id=1, bbox=[200, 100, 300, 400], confidence=0.9)
    
    data = FrameData(frame_index=0, timestamp=time.time(), raw_frame=dummy_frame, processed_frame=dummy_frame.copy(), persons=[person])
    out = anonymizer.process(data)
    print("Anonymization mock process finished.")
