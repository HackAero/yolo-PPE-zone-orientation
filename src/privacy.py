import cv2
import numpy as np
from src.types import FrameData
import src.config as config

class PrivacyAnonymizer:
    def __init__(self, use_mock: bool = False):
        """
        Provides anonymization by blurring faces and exposed sensitive areas.
        """
        self.use_mock = use_mock
        if not use_mock:
            # Load default OpenCV Haar Cascade face detector (always bundled with opencv-python)
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            print(f"[Privacy] Loading face detector: {cascade_path}")
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                print("[Privacy] Warning: Haar face cascade could not load. Falling back to upper-body heuristic.")
                self.face_cascade = None
        else:
            print("[Privacy] Running in MOCK mode.")
            self.face_cascade = None

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

    def process(self, frame_data: FrameData) -> FrameData:
        """
        Conveyor belt stage:
        Iterates over detected people, detects their faces, and blurs them on processed_frame.
        """
        # Note: In both mock and real mode, we run on frame_data.persons
        frame = frame_data.processed_frame
        h_img, w_img = frame.shape[:2]

        for person in frame_data.persons:
            xmin, ymin, xmax, ymax = person.bbox
            person_w = xmax - xmin
            person_h = ymax - ymin
            
            # --- Face Detection & Blurring ---
            face_blurred = False
            
            if self.face_cascade is not None and not self.use_mock:
                # Crop person region for face detection (makes it faster and less false-positive prone)
                pxmin = max(0, xmin)
                pymin = max(0, ymin)
                pxmax = min(w_img, xmax)
                pymax = min(h_img, ymax)
                
                person_crop = frame[pymin:pymax, pxmin:pxmax]
                if person_crop.size > 0:
                    gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
                    # Detect faces in the crop
                    faces = self.face_cascade.detectMultiScale(
                        gray, 
                        scaleFactor=1.05, 
                        minNeighbors=4, 
                        minSize=(20, 20)
                    )
                    
                    for (fx, fy, fw, fh) in faces:
                        # Translate crop coordinates back to full image coordinates
                        abs_fx = pxmin + fx
                        abs_fy = pymin + fy
                        
                        # Apply padding
                        pad = config.PRIVACY_FACE_PADDING
                        self._blur_region(
                            frame,
                            abs_fx - pad,
                            abs_fy - pad,
                            abs_fx + fw + pad,
                            abs_fy + fh + pad
                        )
                        face_blurred = True

            # --- Heuristic Fallback ---
            # If face detector is unavailable, runs in mock mode, or failed to detect:
            # We blur the top 20% of the bounding box where the head is guaranteed to be.
            if not face_blurred:
                head_ymin = ymin
                head_ymax = ymin + int(person_h * 0.22)
                
                # Center the horizontal blur slightly to avoid background blurring
                offset_w = int(person_w * 0.15)
                head_xmin = xmin + offset_w
                head_xmax = xmax - offset_w
                
                self._blur_region(frame, head_xmin, head_ymin, head_xmax, head_ymax)
            
            # --- Tattoos & Exposed Skin Blurring (Optional Hackathon Stage) ---
            # If someone has exposed arms where tattoos might be, apply a light blur.
            # We look at the middle height of the bounding box (arms region).
            # Tattoos can be toggled via config or metadata.
            if person.metadata.get("has_exposed_tattoo", False):
                # Blur arm region (heuristic: middle third of person height)
                arm_ymin = ymin + int(person_h * 0.3)
                arm_ymax = ymin + int(person_h * 0.7)
                self._blur_region(frame, xmin, arm_ymin, xmax, arm_ymax)

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
