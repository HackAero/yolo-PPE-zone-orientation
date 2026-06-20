from ultralytics import YOLO

import src.config as config


def _normalize_label(label: str) -> str:
    return label.lower().replace("-", "_").strip()


class HelmetDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.model_path = model_path

    def detect_helmets(self, frame):
        results = self.model(frame, verbose=False)[0]
        helmet_boxes = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = _normalize_label(self.model.names[cls_id])
            conf = float(box.conf[0])
            if conf < config.PPE_CONF_THRESHOLD:
                continue
            if label == "helmet":
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                helmet_boxes.append((x1, y1, x2, y2, conf))

        return helmet_boxes


def helmet_inside_person(person_bbox, helmet_bbox):
    px1, py1, px2, py2 = person_bbox
    hx1, hy1, hx2, hy2, _ = helmet_bbox

    person_height = py2 - py1
    top_limit = py1 + int(person_height * 0.55)
    center_x = (hx1 + hx2) // 2
    center_y = (hy1 + hy2) // 2

    overlap_x1 = max(px1, hx1)
    overlap_y1 = max(py1, hy1)
    overlap_x2 = min(px2, hx2)
    overlap_y2 = min(py2, hy2)
    overlap_width = max(0, overlap_x2 - overlap_x1)
    overlap_height = max(0, overlap_y2 - overlap_y1)
    overlap_area = overlap_width * overlap_height
    helmet_area = max(1, (hx2 - hx1) * (hy2 - hy1))
    overlap_ratio = overlap_area / helmet_area

    return px1 <= center_x <= px2 and py1 <= center_y <= top_limit and overlap_ratio >= 0.05


class GlassesDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.model_path = model_path

    def detect_glasses(self, frame):
        results = self.model(frame, verbose=False)[0]
        glasses_boxes = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = _normalize_label(self.model.names[cls_id])
            conf = float(box.conf[0])
            if conf < config.PPE_CONF_THRESHOLD:
                continue
            if label == "goggles":
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                glasses_boxes.append((x1, y1, x2, y2, conf))

        return glasses_boxes


def glasses_inside_person(person_bbox, glasses_bbox):
    px1, py1, px2, py2 = person_bbox
    gx1, gy1, gx2, gy2, _ = glasses_bbox

    person_height = py2 - py1
    top_limit = py1 + int(person_height * 0.45)
    center_x = (gx1 + gx2) // 2
    center_y = (gy1 + gy2) // 2

    overlap_x1 = max(px1, gx1)
    overlap_y1 = max(py1, gy1)
    overlap_x2 = min(px2, gx2)
    overlap_y2 = min(py2, gy2)
    overlap_width = max(0, overlap_x2 - overlap_x1)
    overlap_height = max(0, overlap_y2 - overlap_y1)
    overlap_area = overlap_width * overlap_height
    glasses_area = max(1, (gx2 - gx1) * (gy2 - gy1))
    overlap_ratio = overlap_area / glasses_area

    return px1 <= center_x <= px2 and py1 <= center_y <= top_limit and overlap_ratio >= 0.05
