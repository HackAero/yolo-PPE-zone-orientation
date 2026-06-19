import cv2
import numpy as np
import os
from src.types import FrameData, TrackedPerson
import src.config as config

class EnvironmentBehaviorMonitor:
    def __init__(self, use_mock: bool = False):
        """
        Monitors environment safety (blur, smoke) and human behavior (fall detection).
        Uses YOLO Pose model for advanced checks and falls back to bounding box aspect ratio checks.
        """
        self.use_mock = use_mock
        # Fall confirmation buffer: tracks consecutive frames person appears fallen
        self.fall_frame_count = {}  # maps person_id -> count of consecutive fall frames
        
        if not use_mock:
            self.pose_model = None
            try:
                from ultralytics import YOLO
                pose_path = "yolov8n-pose.pt"
                # If the pose weight file is in root, load it
                if os.path.exists(pose_path):
                    print(f"[Environment] Loading YOLO Pose Model: {pose_path}")
                    self.pose_model = YOLO(pose_path)
                else:
                    print("[Environment] 'yolov8n-pose.pt' weights not found. Bounding box fallback active for fall detection.")
            except Exception as e:
                print(f"[Environment] Could not initialize YOLO Pose model: {e}")
                self.pose_model = None
        else:
            print("[Environment] Running in MOCK mode.")
            self.pose_model = None

    def process(self, frame_data: FrameData) -> FrameData:
        """
        Conveyor belt stage:
        1. Checks image quality (smoke/fog/smudges) using Laplacian variance.
        2. Detects falls (lying on floor) using skeleton pose angles or aspect ratios.
        """
        # --- 1. Image Quality / Blur Assessment ---
        # Don't compute Laplacian if the frame is mock and empty, or use placeholder values
        if frame_data.raw_frame.size > 0:
            gray = cv2.cvtColor(frame_data.raw_frame, cv2.COLOR_BGR2GRAY)
            # Variance of Laplacian measures edge high-frequency details. Low values = blurry/foggy/smoke.
            blur_val = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            frame_data.blur_score = blur_val
            
            # If camera is covered, lens is smudged, or heavy smoke blocks edges
            if blur_val < config.BLUR_LAPLACIAN_THRESHOLD:
                frame_data.is_image_blurry = True
                alert = {
                    "type": "ENVIRONMENT_WARNING",
                    "severity": "Warning",
                    "message": f"Visual degradation detected! Image is blurry or obscured (Score: {blur_val:.1f})",
                    "timestamp": frame_data.timestamp
                }
                frame_data.alerts.append(alert)
        else:
            frame_data.blur_score = 50.0

        # --- 2. Environment Smoke & Fire (Placeholder for YOLO custom class) ---
        if frame_data.is_smoke_detected:
            alert = {
                "type": "ENVIRONMENT_ALERT",
                "severity": "Critical",
                "message": "DANGER: Smoke / Fire fumes detected in visual feed!",
                "timestamp": frame_data.timestamp
            }
            frame_data.alerts.append(alert)

        # --- 3. Behavior / Fall Detection ---
        # Track active person IDs this frame
        active_person_ids = set(person.person_id for person in frame_data.persons)
        
        # Clean up tracking for people no longer in view (prevents memory leak)
        self.fall_frame_count = {pid: count for pid, count in self.fall_frame_count.items() if pid in active_person_ids}
        
        for person in frame_data.persons:
            if self.use_mock:
                # In mock mode, keep the simulated state if pre-populated.
                if person.is_fallen is None:
                    person.is_fallen = False
            else:
                # Real Mode Fall Analysis
                is_fallen = False
                xmin, ymin, xmax, ymax = person.bbox
                w = xmax - xmin
                h = ymax - ymin
                
                # Method A: YOLO Pose skeletal analysis (PRIMARY METHOD - more reliable)
                if self.pose_model is not None:
                    h_img, w_img = frame_data.raw_frame.shape[:2]
                    pxmin = max(0, xmin)
                    pymin = max(0, ymin)
                    pxmax = min(w_img, xmax)
                    pymax = min(h_img, ymax)
                    
                    crop = frame_data.raw_frame[pymin:pymax, pxmin:pxmax]
                    if crop.size > 0:
                        pose_results = self.pose_model(crop, verbose=False)[0]
                        if pose_results.keypoints is not None and len(pose_results.keypoints.xy) > 0:
                            # Extract coordinates and confidence of keypoints (17 points, shape [1, 17, 2])
                            kpts = pose_results.keypoints.xy[0].cpu().numpy()
                            kpt_conf = pose_results.keypoints.conf[0].cpu().numpy() if hasattr(pose_results.keypoints, 'conf') else None
                            person.keypoints = kpts
                            
                            # COCO Poses:
                            # Shoulder left/right: 5, 6
                            # Hip left/right: 11, 12
                            try:
                                # Only use keypoints if they have sufficient confidence
                                if kpt_conf is not None:
                                    shoulder_conf = (kpt_conf[5] + kpt_conf[6]) / 2.0
                                    hip_conf = (kpt_conf[11] + kpt_conf[12]) / 2.0
                                    
                                    # Check confidence threshold before angle calculation
                                    if shoulder_conf >= config.KEYPOINT_CONFIDENCE_THRESHOLD and hip_conf >= config.KEYPOINT_CONFIDENCE_THRESHOLD:
                                        mid_shoulder = (kpts[5] + kpts[6]) / 2.0
                                        mid_hip = (kpts[11] + kpts[12]) / 2.0
                                        
                                        dx = mid_hip[0] - mid_shoulder[0]
                                        dy = mid_hip[1] - mid_shoulder[1]
                                        
                                        # Angle relative to vertical axis (y-axis)
                                        if dy != 0:
                                            angle = np.degrees(np.arctan(abs(dx) / abs(dy)))
                                            if angle > config.FALL_ANGLE_THRESHOLD:
                                                is_fallen = True
                                else:
                                    # If confidence data unavailable, use keypoints anyway
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
                
                # Method B: Aspect Ratio Fallback (only if pose model not available or failed)
                if not is_fallen and self.pose_model is None:
                    aspect_ratio = w / max(1, h)
                    if aspect_ratio > config.FALL_ASPECT_RATIO_THRESHOLD:
                        is_fallen = True
                
                # Track consecutive fall frames for temporal confirmation
                person_id = person.person_id
                if is_fallen:
                    self.fall_frame_count[person_id] = self.fall_frame_count.get(person_id, 0) + 1
                else:
                    self.fall_frame_count[person_id] = 0
                
                # Only confirm fall if it persists for minimum consecutive frames
                person.is_fallen = self.fall_frame_count[person_id] >= config.FALL_CONFIRMATION_FRAMES

            # Raise critical fall alarms (only after confirmation buffer is satisfied)
            if person.is_fallen:
                alert = {
                    "type": "FALL_ALERT",
                    "severity": "Critical",
                    "message": f"CRITICAL: Worker ID {person.person_id} is detected lying on the floor!",
                    "person_id": person.person_id,
                    "timestamp": frame_data.timestamp
                }
                frame_data.alerts.append(alert)

        return frame_data

if __name__ == "__main__":
    print("Testing EnvironmentBehaviorMonitor Stage in Isolation...")
    import time
    
    monitor = EnvironmentBehaviorMonitor(use_mock=True)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    person = TrackedPerson(person_id=9, bbox=[50, 200, 300, 250], confidence=0.8) # Wide bbox = lying down
    person.is_fallen = True
    
    data = FrameData(
        frame_index=0, 
        timestamp=time.time(), 
        raw_frame=dummy_frame, 
        processed_frame=dummy_frame.copy(), 
        persons=[person],
        is_smoke_detected=True
    )
    
    out = monitor.process(data)
    print("Alerts triggered:")
    for a in out.alerts:
        print(f" - [{a['severity']}] {a['message']}")
