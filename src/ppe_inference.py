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


def _is_negative_helmet_label(label: str) -> bool:
    return "no helmet" in label or "no hardhat" in label or "no glasses" in label


def _is_positive_helmet_label(label: str) -> bool:
    return (
        "helmet" in label
        or "hard hat" in label
        or "safety helmet" in label
        or "safety hat" in label
    )


def _is_negative_glasses_label(label: str) -> bool:
    return "no glasses" in label or "no goggles" in label


def _is_positive_glasses_label(label: str) -> bool:
    return (
        "glass" in label
        or "goggle" in label
        or "eyewear" in label
        or "safety glasses" in label
        or "safety goggles" in label
    )


class SharedPPEDetector:
    """Single PPE YOLO model — one load, one full-frame inference per frame."""

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self.model = None
        self._frame_counter = 0
        self._last_result = PPEResult(model_available=False)
        self._last_person_ppe: dict = {}

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

        self._frame_counter += 1
        run_inference = (self._frame_counter - 1) % config.PPE_INFERENCE_INTERVAL == 0

        if not run_inference:
            if persons is not None:
                self._apply_cached_person_ppe(persons)
            return self._last_result

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
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            bbox = (x1, y1, x2, y2, conf)

            if _is_negative_helmet_label(label):
                continue
            if _is_positive_helmet_label(label):
                helmet_boxes.append(bbox)
                continue

            if _is_negative_glasses_label(label):
                continue
            if _is_positive_glasses_label(label):
                glasses_boxes.append(bbox)

        self._last_result = PPEResult(
            helmet_boxes=helmet_boxes,
            glasses_boxes=glasses_boxes,
            model_available=True,
        )
        if persons is not None:
            assign_ppe_to_persons(persons, self._last_result)
            self.remember_person_ppe(persons)
        return self._last_result

    def _apply_cached_person_ppe(self, persons) -> None:
        for person in persons:
            cached = self._last_person_ppe.get(person.person_id)
            if cached is None:
                continue
            person.has_helmet, person.has_glasses = cached

    def remember_person_ppe(self, persons) -> None:
        self._last_person_ppe = {
            person.person_id: (person.has_helmet, person.has_glasses)
            for person in persons
        }


def assign_ppe_to_persons(persons, ppe_result: PPEResult) -> None:
    """Map full-frame PPE boxes to tracked persons."""
    for person in persons:
        has_helmet = any(
            helmet_inside_person(person.bbox, box) for box in ppe_result.helmet_boxes
        )
        has_glasses = any(
            glasses_inside_person(person.bbox, box) for box in ppe_result.glasses_boxes
        )

        if has_helmet:
            person.has_helmet = True
        elif person.has_helmet is None:
            person.has_helmet = False

        if has_glasses:
            person.has_glasses = True
        elif person.has_glasses is None:
            person.has_glasses = False
