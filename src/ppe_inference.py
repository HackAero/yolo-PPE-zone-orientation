import os
from dataclasses import dataclass, field
from typing import List, Tuple

import src.config as config
from src.ppe import glasses_inside_person, helmet_inside_person


Box = Tuple[int, int, int, int, float]
LabeledBox = Tuple[str, int, int, int, int, float]


@dataclass
class PPEResult:
    helmet_boxes: List[Box] = field(default_factory=list)
    glasses_boxes: List[Box] = field(default_factory=list)
    raw_detections: List[LabeledBox] = field(default_factory=list)
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

    def _run_model(self, frame):
        if self.model is None or frame.size == 0:
            return None

        imgsz = max(config.YOLO_IMGSZ, 640)
        return self.model(
            frame,
            conf=min(config.PPE_CONF_THRESHOLD, 0.12),
            imgsz=imgsz,
            device=config.YOLO_DEVICE,
            verbose=False,
        )[0]

    def _collect_boxes(self, results, x_offset=0, y_offset=0):
        helmet_boxes: List[Box] = []
        glasses_boxes: List[Box] = []
        raw_detections: List[LabeledBox] = []

        if results is None:
            return helmet_boxes, glasses_boxes, raw_detections

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = _normalize_label(results.names[cls_id])
            conf = float(box.conf[0])
            if conf < min(config.PPE_CONF_THRESHOLD, 0.12):
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            bbox = (x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset, conf)
            raw_detections.append((label, *bbox))

            if label == "helmet":
                helmet_boxes.append(bbox)
            elif label == "goggles":
                glasses_boxes.append(bbox)

        return helmet_boxes, glasses_boxes, raw_detections

    @staticmethod
    def _person_head_crop(frame, person_bbox):
        height, width = frame.shape[:2]
        px1, py1, px2, py2 = person_bbox

        person_width = max(1, px2 - px1)
        person_height = max(1, py2 - py1)

        x1 = max(0, int(px1 - person_width * 0.12))
        y1 = max(0, int(py1 - person_height * 0.12))
        x2 = min(width, int(px2 + person_width * 0.12))
        y2 = min(height, int(py1 + person_height * 0.70))

        if x2 <= x1 or y2 <= y1:
            return None, (0, 0)

        return frame[y1:y2, x1:x2], (x1, y1)

    def detect_all(self, frame, persons=None) -> PPEResult:
        if self.model is None or frame.size == 0:
            return PPEResult(model_available=False)

        helmet_boxes: List[Box] = []
        glasses_boxes: List[Box] = []
        raw_detections: List[LabeledBox] = []

        # Run once on the full frame, then again on each tracked person's head/upper body crop.
        # The crop pass helps with small PPE items like helmets and safety glasses.
        full_frame_results = self._run_model(frame)
        full_helmet_boxes, full_glasses_boxes, full_raw_detections = self._collect_boxes(full_frame_results)
        helmet_boxes.extend(full_helmet_boxes)
        glasses_boxes.extend(full_glasses_boxes)
        raw_detections.extend(full_raw_detections)

        if persons:
            for person in persons:
                crop, (x_offset, y_offset) = self._person_head_crop(frame, person.bbox)
                if crop is None or crop.size == 0:
                    continue

                crop_results = self._run_model(crop)
                crop_helmets, crop_glasses, crop_raw_detections = self._collect_boxes(crop_results, x_offset, y_offset)
                helmet_boxes.extend(crop_helmets)
                glasses_boxes.extend(crop_glasses)
                raw_detections.extend(crop_raw_detections)

        result = PPEResult(
            helmet_boxes=helmet_boxes,
            glasses_boxes=glasses_boxes,
            raw_detections=raw_detections,
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
