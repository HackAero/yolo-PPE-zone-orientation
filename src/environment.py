import cv2
import numpy as np
import os
from typing import Dict, List, Tuple

from ultralytics import YOLO
from src.session_labels import worker_label
from src.pipeline_types import FrameData, TrackedPerson
import src.config as config


class EnvironmentBehaviorMonitor:
    def __init__(self, use_mock: bool = False):
        """
        Monitors environment safety (blur, smoke) and human behavior (fall detection).
        Uses YOLO Pose model for advanced checks and falls back to bounding box aspect ratio checks.
        """
        self.use_mock = use_mock
        self.fall_frame_count: Dict[int, int] = {}
        self._frame_counter = 0
        self._last_smoke_detected = False
        self._last_fire_detected = False
        self._last_person_pose: Dict[int, dict] = {}
        
        # Counters for temporal confirmation
        self.smoke_frame_count = 0  
        self.fire_frame_count = 0

        if not use_mock:
            self.pose_model = None
            self.smoke_fire_model = None
            try:
                print("[Environment] Initializing YOLO Pose Model...")
                self.pose_model = YOLO(config.POSE_MODEL_PATH)

                root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                smoke_fire_path = os.path.join(root_dir, "models/fire_smoke.pt")

                if os.path.exists(smoke_fire_path):
                    print(f"[Environment] Loading Fire/Smoke Model from: {smoke_fire_path}")
                    self.smoke_fire_model = YOLO(smoke_fire_path)
                else:
                    print(f"[Environment] Fire/Smoke model not found at: {smoke_fire_path}")
                    self.smoke_fire_model = None

            except Exception as exc:
                print(f"[Environment] Could not initialize YOLO Pose model: {exc}")
                self.pose_model = None
                self.smoke_fire_model = None
        else:
            print("[Environment] Running in MOCK mode.")
            self.pose_model = None
            self.smoke_fire_model = None

    def _yolo_kwargs(self) -> dict:
        return {
            "imgsz": config.YOLO_IMGSZ,
            "device": config.YOLO_DEVICE,
            "verbose": False,
        }


    def _run_smoke_detection(self, frame_data: FrameData) -> None:
        if self.use_mock:
            pass 

        run_smoke = (self._frame_counter - 1) % config.SMOKE_INFERENCE_INTERVAL == 0

        if (
            run_smoke
            and self.smoke_fire_model is not None
            and frame_data.raw_frame.size > 0
        ):
            sf_results = self.smoke_fire_model(frame_data.raw_frame, **self._yolo_kwargs())[0]
            
            # Reset current frame state
            current_smoke = False
            current_fire = False

            for box in sf_results.boxes:
                cls_id = int(box.cls[0])
                class_name = sf_results.names[cls_id]
                confidence = float(box.conf[0])
                
                if class_name == 'smoke' and confidence >= config.SMOKE_CONF_THRESHOLD :
                    current_smoke = True
                        
                elif class_name == 'fire' and confidence >= config.FIRE_CONF_THRESHOLD :
                    current_fire = True

            # --- TEMPORAL BUFFER LOGIC ---
            
            # 1. Handle Smoke Counter
            if current_smoke:
                self.smoke_frame_count += 1
            else:
                self.smoke_frame_count = 0 # Reset if it disappears
                
            # 2. Handle Fire Counter
            if current_fire:
                self.fire_frame_count += 1
            else:
                self.fire_frame_count = 0 # Reset if it disappears
                
            # 3. Trigger Alerts ONLY if thresholds are met
            if self.smoke_frame_count >= config.SMOKE_CONFIRMATION_FRAMES:
                frame_data.alerts.append({
                    "type": "ENVIRONMENT_ALERT",
                    "severity": "Critical",
                    "message": f"DANGER: Persistent SMOKE detected! ({self.smoke_frame_count} consecutive checks)",
                    "timestamp": frame_data.timestamp,
                })
                # Optional: prevent alert spam by resetting counter after alerting, 
                # or keep it high to send continuous alerts.
                self.smoke_frame_count = 0 
                
            if self.fire_frame_count >= config.FIRE_CONFIRMATION_FRAMES:
                frame_data.alerts.append({
                    "type": "ENVIRONMENT_ALERT",
                    "severity": "Critical",
                    "message": f"DANGER: Persistent FIRE detected! ({self.fire_frame_count} consecutive checks)",
                    "timestamp": frame_data.timestamp,
                })
                self.fire_frame_count = 0

            # Update class cache
            self._last_smoke_detected = current_smoke
            self._last_fire_detected = current_fire

        # Always update frame_data with the latest known state
        frame_data.is_smoke_detected = self._last_smoke_detected
        setattr(frame_data, 'is_fire_detected', self._last_fire_detected)
        
        
    def _collect_pose_crops(
        self, frame_data: FrameData
    ) -> Tuple[List[TrackedPerson], List[np.ndarray]]:
        persons: List[TrackedPerson] = []
        crops: List[np.ndarray] = []
        h_img, w_img = frame_data.raw_frame.shape[:2]

        for person in frame_data.persons:
            xmin, ymin, xmax, ymax = person.bbox
            pxmin = max(0, xmin)
            pymin = max(0, ymin)
            pxmax = min(w_img, xmax)
            pymax = min(h_img, ymax)
            crop = frame_data.raw_frame[pymin:pymax, pxmin:pxmax]
            if crop.size > 0:
                persons.append(person)
                crops.append(crop)

        return persons, crops

    def _apply_pose_result(self, person: TrackedPerson, pose_results) -> bool:
        is_fallen = False

        if pose_results.keypoints is not None and len(pose_results.keypoints.xy) > 0:
            kpts = pose_results.keypoints.xy[0].cpu().numpy()
            kpt_conf = (
                pose_results.keypoints.conf[0].cpu().numpy()
                if hasattr(pose_results.keypoints, "conf")
                else None
            )
            person.keypoints = kpts

            try:
                if kpt_conf is not None:
                    shoulder_conf = (kpt_conf[5] + kpt_conf[6]) / 2.0
                    hip_conf = (kpt_conf[11] + kpt_conf[12]) / 2.0

                    if (
                        shoulder_conf >= config.KEYPOINT_CONFIDENCE_THRESHOLD
                        and hip_conf >= config.KEYPOINT_CONFIDENCE_THRESHOLD
                    ):
                        mid_shoulder = (kpts[5] + kpts[6]) / 2.0
                        mid_hip = (kpts[11] + kpts[12]) / 2.0
                        dx = mid_hip[0] - mid_shoulder[0]
                        dy = mid_hip[1] - mid_shoulder[1]
                        if dy != 0:
                            angle = np.degrees(np.arctan(abs(dx) / abs(dy)))
                            if angle > config.FALL_ANGLE_THRESHOLD:
                                is_fallen = True
                else:
                    mid_shoulder = (kpts[5] + kpts[6]) / 2.0
                    mid_hip = (kpts[11] + kpts[12]) / 2.0
                    dx = mid_hip[0] - mid_shoulder[0]
                    dy = mid_hip[1] - mid_shoulder[1]
                    if dy != 0:
                        angle = np.degrees(np.arctan(abs(dx) / abs(dy)))
                        if angle > config.FALL_ANGLE_THRESHOLD:
                            is_fallen = True
            except Exception:
                pass

        return is_fallen

    def _run_pose_detection(self, frame_data: FrameData) -> None:
        run_pose = (self._frame_counter - 1) % config.POSE_INFERENCE_INTERVAL == 0

        if self.use_mock:
            for person in frame_data.persons:
                if person.is_fallen is None:
                    person.is_fallen = False
            return

        if run_pose and self.pose_model is not None:
            persons, crops = self._collect_pose_crops(frame_data)
            if crops:
                batch_results = self.pose_model(crops, **self._yolo_kwargs())
                if not isinstance(batch_results, list):
                    batch_results = [batch_results]

                for person, pose_result in zip(persons, batch_results):
                    is_fallen = self._apply_pose_result(person, pose_result)
                    self._last_person_pose[person.person_id] = {
                        "is_fallen_raw": is_fallen,
                        "keypoints": person.keypoints,
                    }
        else:
            for person in frame_data.persons:
                cached = self._last_person_pose.get(person.person_id)
                if cached is not None and cached.get("keypoints") is not None:
                    person.keypoints = cached["keypoints"]

        for person in frame_data.persons:
            xmin, ymin, xmax, ymax = person.bbox
            w = xmax - xmin
            h = ymax - ymin
            is_fallen = False

            cached = self._last_person_pose.get(person.person_id)
            if cached is not None:
                is_fallen = cached.get("is_fallen_raw", False)
            elif self.pose_model is None:
                aspect_ratio = w / max(1, h)
                if aspect_ratio > config.FALL_ASPECT_RATIO_THRESHOLD:
                    is_fallen = True

            person_id = person.person_id
            if is_fallen:
                self.fall_frame_count[person_id] = self.fall_frame_count.get(person_id, 0) + 1
            else:
                self.fall_frame_count[person_id] = 0

            person.is_fallen = (
                self.fall_frame_count[person_id] >= config.FALL_CONFIRMATION_FRAMES
            )

    def process(self, frame_data: FrameData) -> FrameData:
        self._frame_counter += 1

        if frame_data.raw_frame.size > 0:
            gray = cv2.cvtColor(frame_data.raw_frame, cv2.COLOR_BGR2GRAY)
            blur_val = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            frame_data.blur_score = blur_val

            if blur_val < config.BLUR_LAPLACIAN_THRESHOLD:
                frame_data.is_image_blurry = True
                alert = {
                    "type": "ENVIRONMENT_WARNING",
                    "severity": "Warning",
                    "message": (
                        f"Visual degradation detected! Image is blurry or obscured "
                        f"(Score: {blur_val:.1f})"
                    ),
                    "timestamp": frame_data.timestamp,
                }
                frame_data.alerts.append(alert)
        else:
            frame_data.blur_score = 50.0

        self._run_smoke_detection(frame_data)

        active_person_ids = {person.person_id for person in frame_data.persons}
        self.fall_frame_count = {
            pid: count
            for pid, count in self.fall_frame_count.items()
            if pid in active_person_ids
        }
        self._last_person_pose = {
            pid: data
            for pid, data in self._last_person_pose.items()
            if pid in active_person_ids
        }

        self._run_pose_detection(frame_data)

        for person in frame_data.persons:
            if person.is_fallen:
                alert = {
                    "type": "FALL_ALERT",
                    "severity": "Critical",
                    "message": (
                        f"CRITICAL: {worker_label(person.person_id)} is detected lying on the floor!"
                    ),
                    "person_id": person.person_id,
                    "timestamp": frame_data.timestamp,
                }
                frame_data.alerts.append(alert)

        return frame_data
