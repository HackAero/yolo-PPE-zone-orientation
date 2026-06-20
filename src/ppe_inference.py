"""
One shared PPE YOLO pass per frame (or every N frames in FAST_MODE).

Assigns helmet/goggles flags to tracked persons by box overlap. Compliance.py
handles vest color and heuristics when the model misses something.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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
        self._frame_counter = 0
        self._cached_result: Optional[PPEResult] = None

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

    def _run_model(self, frame_or_crops):
        if self.model is None:
            return None

        imgsz = getattr(config, "PPE_IMGSZ", config.YOLO_IMGSZ)
        return self.model(
            frame_or_crops,
            conf=min(config.PPE_CONF_THRESHOLD, 0.12),
            imgsz=imgsz,
            device=config.YOLO_DEVICE,
            verbose=False,
        )

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

    @staticmethod
    def _needs_crop_pass(person, frame_shape: Tuple[int, ...]) -> bool:
        """Crop refine only when full-frame pass likely missed small PPE."""
        frame_h = frame_shape[0]
        person_h = person.bbox[3] - person.bbox[1]
        if person_h < frame_h * 0.28:
            return True
        return not person.has_helmet or not person.has_glasses

    def _merge_boxes(
        self,
        helmet_boxes: List[Box],
        glasses_boxes: List[Box],
        raw_detections: List[LabeledBox],
        new_helmets: List[Box],
        new_glasses: List[Box],
        new_raw: List[LabeledBox],
    ) -> None:
        helmet_boxes.extend(new_helmets)
        glasses_boxes.extend(new_glasses)
        raw_detections.extend(new_raw)

    def detect_all(self, frame, persons=None) -> PPEResult:
        if self.model is None or frame.size == 0:
            return PPEResult(model_available=False)

        self._frame_counter += 1
        run_inference = (
            self._cached_result is None
            or (self._frame_counter - 1) % config.PPE_INFERENCE_INTERVAL == 0
        )

        if not run_inference and self._cached_result is not None:
            result = PPEResult(
                helmet_boxes=list(self._cached_result.helmet_boxes),
                glasses_boxes=list(self._cached_result.glasses_boxes),
                raw_detections=list(self._cached_result.raw_detections),
                model_available=True,
            )
            if persons is not None:
                assign_ppe_to_persons(persons, result)
            return result

        helmet_boxes: List[Box] = []
        glasses_boxes: List[Box] = []
        raw_detections: List[LabeledBox] = []

        full_outputs = self._run_model(frame)
        if full_outputs is not None:
            full_result = full_outputs[0] if isinstance(full_outputs, list) else full_outputs
            h, g, r = self._collect_boxes(full_result)
            self._merge_boxes(helmet_boxes, glasses_boxes, raw_detections, h, g, r)

        result = PPEResult(
            helmet_boxes=helmet_boxes,
            glasses_boxes=glasses_boxes,
            raw_detections=raw_detections,
            model_available=True,
        )
        if persons is not None:
            assign_ppe_to_persons(persons, result)

        if persons and getattr(config, "PPE_CROP_PASS", False):
            crop_targets = [
                p for p in persons if self._needs_crop_pass(p, frame.shape)
            ][: getattr(config, "PPE_CROP_MAX_PERSONS", 2)]

            crops = []
            offsets = []
            for person in crop_targets:
                crop, offset = self._person_head_crop(frame, person.bbox)
                if crop is not None and crop.size > 0:
                    crops.append(crop)
                    offsets.append(offset)

            if crops:
                crop_outputs = self._run_model(crops)
                if crop_outputs is not None:
                    for crop_result, (x_off, y_off) in zip(crop_outputs, offsets):
                        h, g, r = self._collect_boxes(crop_result, x_off, y_off)
                        self._merge_boxes(helmet_boxes, glasses_boxes, raw_detections, h, g, r)

                    result.helmet_boxes = helmet_boxes
                    result.glasses_boxes = glasses_boxes
                    result.raw_detections = raw_detections
                    assign_ppe_to_persons(persons, result)

        self._cached_result = PPEResult(
            helmet_boxes=list(result.helmet_boxes),
            glasses_boxes=list(result.glasses_boxes),
            raw_detections=list(result.raw_detections),
            model_available=True,
        )
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
