import cv2
import numpy as np
import time
from typing import Dict, Tuple

from src.face_detection import SharedFaceDetector
from src.session_labels import worker_label
from src.pipeline_types import FrameData, TrackedPerson
import src.config as config


class PPEComplianceChecker:
    def __init__(self, use_mock: bool = False):
        """
        Applies heuristic PPE fallback and generates compliance violations/alerts.
        Primary PPE detection is done upstream by SharedPPEDetector in the engine.
        """
        self.use_mock = use_mock
        # Debounce map: (person_id, violation) -> last alert timestamp
        # Prevents the same violation from spamming the console / robot dispatcher
        # every frame. Keyed by (person_id, violation_name).
        self._alert_last_sent: Dict[Tuple[int, str], float] = {}

    def _detect_yellow_vest_info(self, frame: np.ndarray, bbox: list) -> bool:
        """
        Informational yellow-vest detector.
        Looks for strong yellow coverage in the torso area only.
        """
        xmin, ymin, xmax, ymax = bbox
        person_w = xmax - xmin
        person_h = ymax - ymin
        if person_w < 20 or person_h < 40:
            return False

        torso_ymin = ymin + int(person_h * 0.25)
        torso_ymax = ymin + int(person_h * 0.78)
        torso_xmin = xmin + int(person_w * 0.15)
        torso_xmax = xmax - int(person_w * 0.15)

        h_img, w_img = frame.shape[:2]
        torso_ymin = max(0, min(torso_ymin, h_img - 1))
        torso_ymax = max(0, min(torso_ymax, h_img - 1))
        torso_xmin = max(0, min(torso_xmin, w_img - 1))
        torso_xmax = max(0, min(torso_xmax, w_img - 1))

        torso_crop = frame[torso_ymin:torso_ymax, torso_xmin:torso_xmax]
        if torso_crop.size == 0:
            return False

        hsv = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, np.array([15, 70, 80]), np.array([40, 255, 255]))
        hi_vis_lime_mask = cv2.inRange(hsv, np.array([38, 65, 80]), np.array([55, 255, 255]))
        mask = cv2.bitwise_or(yellow_mask, hi_vis_lime_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        color_pixels = cv2.countNonZero(mask)
        total_pixels = torso_crop.shape[0] * torso_crop.shape[1]
        if total_pixels == 0:
            return False

        ratio = color_pixels / total_pixels
        if ratio < 0.16:
            return False

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False

        largest = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest)
        return contour_area >= 0.07 * total_pixels

    def _detect_helmet_heuristic(self, frame: np.ndarray, bbox: list) -> bool:
        """
        HSV color range heuristic for safety helmets.
        Limits detection to the top-center head region and requires a large,
        contiguous helmet-colored region to avoid false positives.
        """
        xmin, ymin, xmax, ymax = bbox
        person_w = xmax - xmin
        person_h = ymax - ymin

        head_ymin = ymin
        head_ymax = ymin + int(person_h * 0.15)
        head_xmin = xmin + int(person_w * 0.15)
        head_xmax = xmax - int(person_w * 0.15)

        h_img, w_img = frame.shape[:2]
        head_ymin = max(0, min(head_ymin, h_img - 1))
        head_ymax = max(0, min(head_ymax, h_img - 1))
        head_xmin = max(0, min(head_xmin, w_img - 1))
        head_xmax = max(0, min(head_xmax, w_img - 1))

        head_crop = frame[head_ymin:head_ymax, head_xmin:head_xmax]
        if head_crop.size == 0:
            return False

        hsv = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)

        colors = [
            (np.array([5, 80, 80]), np.array([35, 255, 255])),
            (np.array([85, 80, 80]), np.array([140, 255, 255])),
            (np.array([0, 0, 220]), np.array([180, 30, 255])),
            (np.array([0, 90, 90]), np.array([15, 255, 255])),
        ]

        mask = np.zeros(head_crop.shape[:2], dtype=np.uint8)
        for lower, upper in colors:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        color_pixels = cv2.countNonZero(mask)
        total_pixels = head_crop.shape[0] * head_crop.shape[1]
        if total_pixels == 0:
            return False

        ratio = color_pixels / total_pixels
        if ratio < 0.28:
            return False

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False

        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        if contour_area < 0.09 * total_pixels:
            return False

        x, y, w, h = cv2.boundingRect(largest_contour)
        if w < 0.4 * head_crop.shape[1] or h < 0.12 * head_crop.shape[0]:
            return False

        return True

    def _detect_glasses_heuristic(self, frame_data: FrameData, person: TrackedPerson) -> bool:
        """
        Uses cached MediaPipe face detections and eye-region edge contrast to detect glasses.
        """
        faces = SharedFaceDetector.get_faces(frame_data, person.person_id)
        if not faces:
            return False

        xmin, ymin, xmax, ymax = person.bbox
        h_img, w_img = frame_data.raw_frame.shape[:2]
        pxmin = max(0, xmin)
        pymin = max(0, ymin)
        pxmax = min(w_img, xmax)
        pymax = min(h_img, ymax)
        person_crop = frame_data.raw_frame[pymin:pymax, pxmin:pxmax]

        if person_crop.size == 0:
            return False

        crop_h, crop_w = person_crop.shape[:2]

        for face_info in faces:
            detection = face_info.get("detection")
            face_xmin = face_info["face_xmin"]
            face_ymin = face_info["face_ymin"]
            face_xmax = face_info["face_xmax"]
            face_ymax = face_info["face_ymax"]
            face_width = face_info["face_width"]
            face_height = face_info["face_height"]

            if face_width < 80 or face_height < 80:
                continue

            eye_top = face_ymin + int(face_height * 0.18)
            eye_bottom = face_ymin + int(face_height * 0.35)
            eye_left = face_xmin
            eye_right = face_xmax

            if detection is not None and detection.location_data.relative_keypoints:
                keypoints = detection.location_data.relative_keypoints
                if len(keypoints) >= 2:
                    left_eye = keypoints[0]
                    right_eye = keypoints[1]
                    eye_left = max(
                        face_xmin,
                        int(min(left_eye.x, right_eye.x) * crop_w) - int(face_width * 0.12),
                    )
                    eye_right = min(
                        face_xmax,
                        int(max(left_eye.x, right_eye.x) * crop_w) + int(face_width * 0.12),
                    )
                    eye_center_y = int(min(left_eye.y, right_eye.y) * crop_h)
                    eye_top = max(face_ymin, eye_center_y - int(face_height * 0.10))
                    eye_bottom = min(face_ymax, eye_center_y + int(face_height * 0.18))

            eye_top = max(face_ymin, eye_top)
            eye_bottom = min(face_ymax, eye_bottom)
            eye_left = max(face_xmin, eye_left)
            eye_right = min(face_xmax, eye_right)

            eye_region = person_crop[eye_top:eye_bottom, eye_left:eye_right]
            if eye_region.size == 0:
                continue

            gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            equalized = cv2.equalizeHist(gray)
            blurred = cv2.GaussianBlur(equalized, (5, 5), 0)

            edges = cv2.Canny(blurred, 50, 150)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

            h, w = edges.shape[:2]
            edge_pixels = cv2.countNonZero(edges)
            total_pixels = h * w
            if total_pixels == 0:
                continue

            edge_ratio = edge_pixels / total_pixels
            if edge_ratio < 0.03:
                continue

            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=20,
                minLineLength=max(18, int(w * 0.2)),
                maxLineGap=8,
            )

            horizontal_lines = 0
            vertical_lines = 0
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    dx = abs(x2 - x1)
                    dy = abs(y2 - y1)
                    if dx > dy * 2 and dx > int(w * 0.18) and max(y1, y2) < int(h * 0.75):
                        horizontal_lines += 1
                    elif dy > dx * 2 and dy > int(h * 0.2):
                        vertical_lines += 1

            if horizontal_lines >= 2:
                return True
            if horizontal_lines >= 1 and vertical_lines >= 1 and edge_ratio > 0.05:
                return True
            if edge_ratio > 0.12 and horizontal_lines >= 1:
                return True

        return False

    def process(self, frame_data: FrameData) -> FrameData:
        """
        Conveyor belt stage:
        Applies heuristic fallback for persons not matched by the shared PPE model,
        then generates violations and alerts.
        """
        run_heuristics = (
            self.use_mock
            or frame_data.frame_index % config.COMPLIANCE_HEURISTIC_INTERVAL == 0
        )
        ppe_model_available = bool(
            (frame_data.extra_metadata.get("ppe_debug") or {}).get("model_available", False)
        )

        for person in frame_data.persons:
            if self.use_mock:
                person.metadata["has_yellow_vest"] = bool(
                    person.metadata.get("has_yellow_vest", True)
                )
            else:
                if run_heuristics or "cached_yellow_vest" not in person.metadata:
                    person.metadata["cached_yellow_vest"] = self._detect_yellow_vest_info(
                        frame_data.raw_frame, person.bbox
                    )
                person.metadata["has_yellow_vest"] = bool(person.metadata["cached_yellow_vest"])

            if self.use_mock:
                if person.has_helmet is None:
                    person.has_helmet = True
                if person.has_glasses is None:
                    person.has_glasses = True
            else:
                # Helmet: only use color heuristic when the YOLO model did not run
                # (None). If the model explicitly found no helmet (False), trust it —
                # the head-region HSV heuristic false-positives on hair/skin/lighting.
                if person.has_helmet is None:
                    if run_heuristics:
                        person.has_helmet = self._detect_helmet_heuristic(
                            frame_data.raw_frame, person.bbox
                        )
                        person.metadata["cached_helmet"] = person.has_helmet
                    elif "cached_helmet" in person.metadata:
                        person.has_helmet = person.metadata["cached_helmet"]
                    else:
                        person.has_helmet = False
                elif person.has_helmet is False:
                    person.metadata["cached_helmet"] = False

                allow_glasses_heuristic = (
                    bool(getattr(config, "PPE_USE_GLASSES_HEURISTIC", False))
                    and not ppe_model_available
                )
                if not person.has_glasses and allow_glasses_heuristic:
                    if run_heuristics:
                        person.has_glasses = self._detect_glasses_heuristic(
                            frame_data, person
                        )
                        # Only cache freshly computed heuristic results
                        person.metadata["cached_glasses"] = person.has_glasses
                    elif "cached_glasses" in person.metadata:
                        person.has_glasses = person.metadata["cached_glasses"]

                # Always persist positives so PPE model can't reset a confirmed detection
                if person.has_helmet:
                    person.metadata["cached_helmet"] = True
                if person.has_glasses and allow_glasses_heuristic:
                    person.metadata["cached_glasses"] = True

            violations = []
            if not person.has_helmet:
                violations.append("Helmet")
            if not person.has_glasses:
                violations.append("Glasses")

            person.compliance_violations = violations

            now = frame_data.timestamp
            debounce = config.ALERT_DEBOUNCE_SECONDS

            for violation in violations:
                debounce_key = (person.person_id, violation)
                last_sent = self._alert_last_sent.get(debounce_key, 0.0)

                # Always attach the alert dict so downstream (dashboard) can see state
                alert = {
                    "type": "PPE_VIOLATION",
                    "severity": "Warning",
                    "message": f"{worker_label(person.person_id)} is missing a {violation.lower()}!",
                    "person_id": person.person_id,
                    "timestamp": now,
                    # Flag lets main.py / dispatcher know whether to surface this
                    "debounced": debounce > 0 and (now - last_sent) < debounce,
                }
                frame_data.alerts.append(alert)

                # Update the debounce timer for non-debounced alerts
                if not alert["debounced"]:
                    self._alert_last_sent[debounce_key] = now

        return frame_data


if __name__ == "__main__":
    print("Testing PPEComplianceChecker Stage in Isolation...")
    import time
    from src.pipeline_types import TrackedPerson

    checker = PPEComplianceChecker(use_mock=True)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    person = TrackedPerson(person_id=42, bbox=[100, 100, 200, 300], confidence=0.8)
    person.has_helmet = False

    data = FrameData(
        frame_index=0,
        timestamp=time.time(),
        raw_frame=dummy_frame,
        processed_frame=dummy_frame.copy(),
        persons=[person],
    )
    out = checker.process(data)
    print("Alerts triggered:")
    for alert in out.alerts:
        print(f" - {alert['message']}")
