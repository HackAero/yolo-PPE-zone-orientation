from ultralytics import YOLO
import numpy as np
import cv2


class HelmetDetector:
    def __init__(self, model_path="yolov8n.pt"):
        """
        Detects safety helmets in frames.
        Use a trained helmet model path for best results.
        """
        self.model = YOLO(model_path)
        self.model_path = model_path

    def detect_helmets(self, frame):
        """
        Returns helmet boxes as [(x1, y1, x2, y2, confidence), ...]
        Works with trained PPE models that detect 'helmet' and 'no_helmet' classes.
        """
        results = self.model(frame, verbose=False)[0]

        helmet_boxes = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = self.model.names[cls_id]
            conf = float(box.conf[0])

            # Detect positive helmet detections (ignore no_helmet detections)
            if class_name.lower() in ["helmet", "hardhat", "hard_hat", "safety helmet"]:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                helmet_boxes.append((x1, y1, x2, y2, conf))

        return helmet_boxes


def helmet_inside_person(person_bbox, helmet_bbox):
    px1, py1, px2, py2 = person_bbox
    hx1, hy1, hx2, hy2, _ = helmet_bbox

    helmet_center_x = (hx1 + hx2) // 2
    helmet_center_y = (hy1 + hy2) // 2

    # Helmet should be inside person box
    inside_person = (
        px1 <= helmet_center_x <= px2 and
        py1 <= helmet_center_y <= py2
    )

    # Helmet should be near top 40% of body
    person_height = py2 - py1
    top_region_bottom = py1 + int(0.4 * person_height)

    in_head_region = py1 <= helmet_center_y <= top_region_bottom

    return inside_person and in_head_region


class GlassesDetector:
    def __init__(self, model_path="yolov8n.pt"):
        """
        Detects safety glasses/goggles in frames.
        Use a trained glasses model path for best results.
        """
        self.model = YOLO(model_path)
        self.model_path = model_path

    def detect_glasses(self, frame):
        """
        Returns goggles/glasses boxes as [(x1, y1, x2, y2, confidence), ...]
        Works with trained PPE models that detect 'goggles' and 'no_goggles' classes.
        """
        results = self.model(frame, verbose=False)[0]

        glasses_boxes = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = self.model.names[cls_id]
            conf = float(box.conf[0])

            # Detect positive goggles/glasses detections (ignore no_goggles)
            if class_name.lower() in ["glasses", "goggles", "safety glasses", "safety goggles", "eyewear"]:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                glasses_boxes.append((x1, y1, x2, y2, conf))

        return glasses_boxes


def glasses_inside_person(person_bbox, glasses_bbox):
    """
    Check if glasses are worn by the person.
    Glasses should be in the upper head region of the person bounding box.
    """
    px1, py1, px2, py2 = person_bbox
    gx1, gy1, gx2, gy2, _ = glasses_bbox

    glasses_center_x = (gx1 + gx2) // 2
    glasses_center_y = (gy1 + gy2) // 2

    # Glasses should be inside person box
    inside_person = (
        px1 <= glasses_center_x <= px2 and
        py1 <= glasses_center_y <= py2
    )

    # Glasses should be near top 35% of body (face/eye region)
    person_height = py2 - py1
    top_region_bottom = py1 + int(0.35 * person_height)

    in_face_region = py1 <= glasses_center_y <= top_region_bottom

    return inside_person and in_face_region