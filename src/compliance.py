import cv2
import numpy as np
import os
from src.pipeline_types import FrameData
import src.config as config

class PPEComplianceChecker:
    def __init__(self, use_mock: bool = False):
        """
        Detects if people are wearing PPE (Helmets, Glasses, etc.).
        Supports loading a custom YOLO model and falls back to a color-based heuristic if model is missing.
        """
        self.use_mock = use_mock
        if not use_mock and os.path.exists(config.PPE_MODEL_PATH) and config.PPE_MODEL_PATH != "yolov8n.pt":
            # If the user has placed a custom-trained model in models/
            from ultralytics import YOLO
            print(f"[Compliance] Loading custom PPE Model: {config.PPE_MODEL_PATH}")
            try:
                self.model = YOLO(config.PPE_MODEL_PATH)
            except Exception as e:
                print(f"[Compliance] Error loading custom model: {e}. Falling back to heuristics.")
                self.model = None
        else:
            if not use_mock:
                print("[Compliance] Custom model not configured or missing. Running with color-heuristic fallback.")
            self.model = None

    def _detect_helmet_heuristic(self, frame: np.ndarray, bbox: list) -> bool:
        """
        HSV color range heuristic for safety helmets (Yellow, Orange, Blue, White).
        Analyzes the head region (top 20% of the bounding box).
        """
        xmin, ymin, xmax, ymax = bbox
        person_w = xmax - xmin
        person_h = ymax - ymin
        
        # Isolate top 22% of bounding box
        head_ymin = ymin
        head_ymax = ymin + int(person_h * 0.22)
        
        # Center horizontally to reduce background noise
        offset_w = int(person_w * 0.2)
        head_xmin = xmin + offset_w
        head_xmax = xmax - offset_w
        
        h_img, w_img = frame.shape[:2]
        head_ymin = max(0, min(head_ymin, h_img - 1))
        head_ymax = max(0, min(head_ymax, h_img - 1))
        head_xmin = max(0, min(head_xmin, w_img - 1))
        head_xmax = max(0, min(head_xmax, w_img - 1))
        
        head_crop = frame[head_ymin:head_ymax, head_xmin:head_xmax]
        if head_crop.size == 0:
            return True  # Avoid false positives if cropping fails
            
        hsv = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
        
        # Yellow and Orange helmets
        lower_yellow = np.array([10, 60, 60])
        upper_yellow = np.array([35, 255, 255])
        
        # Blue helmets
        lower_blue = np.array([90, 60, 60])
        upper_blue = np.array([135, 255, 255])
        
        # White helmets (high brightness, low saturation)
        lower_white = np.array([0, 0, 190])
        upper_white = np.array([180, 45, 255])
        
        mask_y = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_b = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_w = cv2.inRange(hsv, lower_white, upper_white)
        
        combined_mask = mask_y | mask_b | mask_w
        color_pixels = cv2.countNonZero(combined_mask)
        total_pixels = head_crop.shape[0] * head_crop.shape[1]
        
        if total_pixels == 0:
            return True
            
        ratio = color_pixels / total_pixels
        # If more than 12% of the crop pixels match safety helmet colors, assume wearing
        return ratio > 0.12

    def process(self, frame_data: FrameData) -> FrameData:
        """
        Conveyor belt stage:
        Evaluates safety gear for each detected person.
        """
        for person in frame_data.persons:
            # Skip if mock values are already defined by mock engine
            if self.use_mock:
                if person.has_helmet is None:
                    person.has_helmet = True
                if person.has_glasses is None:
                    person.has_glasses = True
            else:
                # Real Mode
                if self.model is not None:
                    # Run inference on crop
                    xmin, ymin, xmax, ymax = person.bbox
                    h_img, w_img = frame_data.raw_frame.shape[:2]
                    pxmin = max(0, xmin)
                    pymin = max(0, ymin)
                    pxmax = min(w_img, xmax)
                    pymax = min(h_img, ymax)
                    
                    crop = frame_data.raw_frame[pymin:pymax, pxmin:pxmax]
                    if crop.size > 0:
                        results = self.model(crop, conf=config.PPE_CONF_THRESHOLD, verbose=False)[0]
                        
                        has_helmet = False
                        has_glasses = False
                        
                        # Match bounding box labels to detect helmet/glasses
                        for box in results.boxes:
                            cls_id = int(box.cls[0].cpu().item())
                            label = results.names[cls_id].lower()
                            
                            if "helmet" in label or "hardhat" in label or "hard-hat" in label:
                                if "no-" not in label and "no_helmet" not in label:
                                    has_helmet = True
                            if "glass" in label or "goggle" in label or "eyewear" in label:
                                if "no-" not in label and "no_glasses" not in label:
                                    has_glasses = True
                                    
                        person.has_helmet = has_helmet
                        person.has_glasses = has_glasses
                else:
                    # Heuristic Fallback Mode
                    person.has_helmet = self._detect_helmet_heuristic(frame_data.raw_frame, person.bbox)
                    # For safety glasses, we default to True in heuristic mode to avoid continuous false alarms,
                    # but allow developers to override it in their compliance branch.
                    person.has_glasses = True

            # Register violations and generate system alerts
            violations = []
            if not person.has_helmet:
                violations.append("Helmet")
            if not person.has_glasses:
                violations.append("Glasses")
                
            person.compliance_violations = violations
            
            # Generate pipeline level alerts
            for violation in violations:
                alert = {
                    "type": "PPE_VIOLATION",
                    "severity": "Warning",
                    "message": f"Worker ID {person.person_id} is missing a {violation.lower()}!",
                    "person_id": person.person_id,
                    "timestamp": frame_data.timestamp
                }
                frame_data.alerts.append(alert)

        return frame_data

if __name__ == "__main__":
    print("Testing PPEComplianceChecker Stage in Isolation...")
    import time
    
    checker = PPEComplianceChecker(use_mock=True)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    person = TrackedPerson(person_id=42, bbox=[100, 100, 200, 300], confidence=0.8)
    
    # Force compliance violation
    person.has_helmet = False
    
    data = FrameData(frame_index=0, timestamp=time.time(), raw_frame=dummy_frame, processed_frame=dummy_frame.copy(), persons=[person])
    out = checker.process(data)
    print("Alerts triggered:")
    for a in out.alerts:
        print(f" - {a['message']}")
