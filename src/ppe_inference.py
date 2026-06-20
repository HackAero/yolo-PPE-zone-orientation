import os
from dataclasses import dataclass, field
from typing import List, Tuple

import src.config as config
from src.ppe import glasses_inside_person, helmet_inside_person


Box = Tuple[int, int, int, int, float]


@dataclass
class PPEResult:
    helmet_boxes: List[Box] = field(default_factory=list)
    glasses_boxes: List[Box] = field(default_factory=list)
    model_available: bool = False


def _normalize_label(class_name: str) -> str:
    return class_name.lower().replace("-", " ").replace("_", " ").strip()


class SharedPPEDetector:
    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self.model = None

        if not use_mock and os.path.exists(config.PPE_MODEL_PATH):
            from ultralytics import YOLO

            print(f"[PPE] Loading shared PPE model: {config.PPE_MODEL_PATH}")
            try:
                self.model = YOLO(config.PPE_MODEL_PATH)
            except Exception as exc:
                print(f"[PPE] Error loading model: {exc}. Heuristics-only fallback.")
                self.model = None
        elif not use_mock:
            print("[PPE] Custom model not found. Heuristics-only fallback.")

    def detect_all(self, frame, persons=None) -> PPEResult:
        if self.model is None or frame.size == 0:
            return PPEResult(model_available=False)

        results = self.model(
            frame,
            conf=config.PPE_CONF_THRESHOLD,
            imgsz=config.YOLO_IMGSZ,
            device=config.YOLO_DEVICE,
            verbose=False,
        )[0]

        helmet_boxes: List[Box] = []
        glasses_boxes: List[Box] = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = _normalize_label(results.names[cls_id])
            conf = float(box.conf[0])
            if conf < config.PPE_CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            bbox = (x1, y1, x2, y2, conf)

            if label == "helmet":
                helmet_boxes.append(bbox)
                continue

            if label == "goggles":
                glasses_boxes.append(bbox)

        result = PPEResult(
            helmet_boxes=helmet_boxes,
            glasses_boxes=glasses_boxes,
            model_available=True,
        )
        if persons is not None:
            assign_ppe_to_persons(persons, result)
        return result


def assign_ppe_to_persons(persons, ppe_result: PPEResult) -> None:
    """Map full-frame PPE boxes to tracked persons."""
    for person in persons:
        person.has_helmet = any(
            helmet_inside_person(person.bbox, box) for box in ppe_result.helmet_boxes
        )
        person.has_glasses = any(
            glasses_inside_person(person.bbox, box) for box in ppe_result.glasses_boxes
        )
