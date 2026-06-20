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


def _is_helmet_label(label: str) -> bool:
    key = _normalize_label(label)
    return (
        key in {
            "helmet",
            "hardhat",
            "hard hat",
            "safety helmet",
            "protective helmet",
        }
        or "helmet" in key
        or "hard hat" in key
        or "hardhat" in key
    )


def _is_glasses_label(label: str) -> bool:
    key = _normalize_label(label)
    return (
        key in {
            "goggles",
            "safety goggles",
            "safety glasses",
            "protective glasses",
            "protective eyewear",
            "clear safety glasses",
            "transparent safety glasses",
            "clear goggles",
            "transparent goggles",
            "glasses",
            "sunglasses",
            "sun glasses",
            "eyeglasses",
            "eye glasses",
            "spectacles",
        }
        or "safety goggle" in key
        or "safety glasses" in key
        or "protective eyewear" in key
        or "sunglass" in key
        or "eyeglass" in key
        or "spectacle" in key
        or "transparent" in key and ("goggle" in key or "glasses" in key or "eyewear" in key)
        or "clear" in key and ("goggle" in key or "glasses" in key or "eyewear" in key)
    )


class SharedPPEDetector:
    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self.model = None
        self._frame_counter = 0
        self._cached_result: Optional[PPEResult] = None

        if use_mock:
            return

        from ultralytics import YOLO

        prefer_local = not bool(getattr(config, "PPE_FORCE_PRETRAINED", False))

        # 1) Prefer custom local PPE model if configured.
        if prefer_local and os.path.exists(config.PPE_MODEL_PATH):
            print(f"[PPE] Loading local PPE model: {config.PPE_MODEL_PATH}")
            try:
                self.model = YOLO(config.PPE_MODEL_PATH)
            except Exception as exc:
                print(f"[PPE] Failed to load local model: {exc}")
                self.model = None

        # 2) Fallback to a pretrained YOLO-World model (downloaded by ultralytics).
        if self.model is None and getattr(config, "PPE_USE_PRETRAINED_WORLD", True):
            pretrained_name = getattr(config, "PPE_PRETRAINED_MODEL", "yolov8s-worldv2.pt")
            print(f"[PPE] Loading pretrained model fallback: {pretrained_name}")
            try:
                self.model = YOLO(pretrained_name)
                if hasattr(self.model, "set_classes"):
                    self.model.set_classes([
                        "helmet",
                        "safety helmet",
                        "hardhat",
                        "hard hat",
                        "safety goggles",
                        "safety glasses",
                        "protective eyewear",
                        "goggles",
                        "eyeglasses",
                        "sunglasses",
                        "spectacles",
                    ])
            except Exception as exc:
                print(f"[PPE] Failed to load pretrained fallback: {exc}")
                self.model = None

        # 3) Fall back to local model if pretrained path failed.
        if self.model is None and os.path.exists(config.PPE_MODEL_PATH):
            print(f"[PPE] Falling back to local PPE model: {config.PPE_MODEL_PATH}")
            try:
                self.model = YOLO(config.PPE_MODEL_PATH)
            except Exception as exc:
                print(f"[PPE] Failed to load local fallback model: {exc}")
                self.model = None

        if self.model is None:
            print("[PPE] No PPE model available. Using heuristics-only fallback.")

    def _run_model(self, frame_or_crops):
        if self.model is None:
            return None

        imgsz = getattr(config, "PPE_IMGSZ", config.YOLO_IMGSZ)
        conf = min(getattr(config, "PPE_CONF_THRESHOLD", 0.2), 0.20)
        return self.model(
            frame_or_crops,
            conf=conf,
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

        names = results.names if hasattr(results, "names") else {}
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = _normalize_label(str(names.get(cls_id, cls_id)))
            conf = float(box.conf[0])
            if conf < min(getattr(config, "PPE_CONF_THRESHOLD", 0.2), 0.20):
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            mapped = (x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset, conf)
            raw_detections.append((label, *mapped))

            if _is_helmet_label(label):
                helmet_boxes.append(mapped)
            elif _is_glasses_label(label):
                glasses_boxes.append(mapped)

        return helmet_boxes, glasses_boxes, raw_detections

    @staticmethod
    def _person_head_crop(frame, person_bbox):
        height, width = frame.shape[:2]
        px1, py1, px2, py2 = person_bbox

        person_width = max(1, px2 - px1)
        person_height = max(1, py2 - py1)

        x1 = max(0, int(px1 - person_width * 0.10))
        y1 = max(0, int(py1 - person_height * 0.12))
        x2 = min(width, int(px2 + person_width * 0.10))
        y2 = min(height, int(py1 + person_height * 0.68))

        if x2 <= x1 or y2 <= y1:
            return None, (0, 0)
        return frame[y1:y2, x1:x2], (x1, y1)

    @staticmethod
    def _needs_crop_pass(person, frame_shape: Tuple[int, ...]) -> bool:
        frame_h = frame_shape[0]
        person_h = person.bbox[3] - person.bbox[1]
        if person_h < frame_h * 0.30:
            return True
        return (person.has_helmet is not True) or (person.has_glasses is not True)

    def detect_all(self, frame, persons=None) -> PPEResult:
        if self.model is None or frame.size == 0:
            return PPEResult(model_available=False)

        self._frame_counter += 1
        run_inference = (
            self._cached_result is None
            or (self._frame_counter - 1) % config.PPE_INFERENCE_INTERVAL == 0
        )

        if not run_inference and self._cached_result is not None:
            cached = PPEResult(
                helmet_boxes=list(self._cached_result.helmet_boxes),
                glasses_boxes=list(self._cached_result.glasses_boxes),
                raw_detections=list(self._cached_result.raw_detections),
                model_available=True,
            )
            if persons is not None:
                assign_ppe_to_persons(persons, cached)
            return cached

        helmet_boxes: List[Box] = []
        glasses_boxes: List[Box] = []
        raw_detections: List[LabeledBox] = []

        full_outputs = self._run_model(frame)
        if full_outputs is not None:
            full_result = full_outputs[0] if isinstance(full_outputs, list) else full_outputs
            h, g, r = self._collect_boxes(full_result)
            helmet_boxes.extend(h)
            glasses_boxes.extend(g)
            raw_detections.extend(r)

        result = PPEResult(
            helmet_boxes=helmet_boxes,
            glasses_boxes=glasses_boxes,
            raw_detections=raw_detections,
            model_available=True,
        )

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
                        helmet_boxes.extend(h)
                        glasses_boxes.extend(g)
                        raw_detections.extend(r)

        result.helmet_boxes = helmet_boxes
        result.glasses_boxes = glasses_boxes
        result.raw_detections = raw_detections

        if persons is not None:
            assign_ppe_to_persons(persons, result)

        self._cached_result = PPEResult(
            helmet_boxes=list(result.helmet_boxes),
            glasses_boxes=list(result.glasses_boxes),
            raw_detections=list(result.raw_detections),
            model_available=True,
        )
        return result


def assign_ppe_to_persons(persons, ppe_result: PPEResult) -> None:
    """Precision-first mapping of PPE boxes to tracked persons with temporal smoothing."""

    helmet_conf = float(getattr(config, "PPE_HELMET_STRICT_CONF", 0.65))
    glasses_conf = float(getattr(config, "PPE_GLASSES_STRICT_CONF", 0.60))
    confirm_frames = max(1, int(getattr(config, "PPE_CONFIRM_FRAMES", 3)))
    clear_frames = max(1, int(getattr(config, "PPE_CLEAR_FRAMES", 5)))

    def _valid_helmet(person_bbox, box) -> bool:
        x1, y1, x2, y2, conf = box
        if conf < helmet_conf:
            return False
        if not helmet_inside_person(person_bbox, box):
            return False

        px1, py1, px2, py2 = person_bbox
        ph = max(1, py2 - py1)
        cy = (y1 + y2) / 2.0
        rel_y = (cy - py1) / ph
        return rel_y <= float(getattr(config, "PPE_HELMET_REL_Y_MAX", 0.55))

    def _valid_glasses(person_bbox, box, helmet_seen: bool) -> bool:
        x1, y1, x2, y2, conf = box
        conf_needed = glasses_conf
        if helmet_seen and bool(getattr(config, "PPE_GLASSES_HELMET_ASSIST", True)):
            conf_needed = max(0.40, glasses_conf - float(getattr(config, "PPE_GLASSES_HELMET_CONF_BONUS", 0.08)))

        if conf < conf_needed:
            return False
        if not glasses_inside_person(person_bbox, box):
            return False

        px1, py1, px2, py2 = person_bbox
        pw = max(1, px2 - px1)
        ph = max(1, py2 - py1)
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        cy = (y1 + y2) / 2.0
        cx = (x1 + x2) / 2.0
        rel_y = (cy - py1) / ph
        rel_x = (cx - px1) / pw

        # Keep glasses in upper-middle face area and reject tiny/noisy boxes.
        area_ratio = (bw * bh) / float(pw * ph)
        if area_ratio < float(getattr(config, "PPE_GLASSES_AREA_MIN", 0.0015)):
            return False
        if area_ratio > float(getattr(config, "PPE_GLASSES_AREA_MAX", 0.14)):
            return False
        if rel_x < float(getattr(config, "PPE_GLASSES_REL_X_MIN", 0.08)):
            return False
        if rel_x > float(getattr(config, "PPE_GLASSES_REL_X_MAX", 0.92)):
            return False
        return rel_y <= float(getattr(config, "PPE_GLASSES_REL_Y_MAX", 0.52))

    def _smooth(person, key: str, seen: bool) -> bool:
        hit_key = f"{key}_hits"
        miss_key = f"{key}_misses"
        stable_key = f"stable_{key}"

        hits = int(person.metadata.get(hit_key, 0))
        misses = int(person.metadata.get(miss_key, 0))
        stable = bool(person.metadata.get(stable_key, False))

        if seen:
            hits = min(confirm_frames, hits + 1)
            misses = 0
        else:
            misses = min(clear_frames, misses + 1)
            hits = max(0, hits - 1)

        if hits >= confirm_frames:
            stable = True
        elif misses >= clear_frames:
            stable = False

        person.metadata[hit_key] = hits
        person.metadata[miss_key] = misses
        person.metadata[stable_key] = stable
        return stable

    for person in persons:
        helmet_seen = any(_valid_helmet(person.bbox, b) for b in ppe_result.helmet_boxes)
        glasses_seen = any(_valid_glasses(person.bbox, b, helmet_seen) for b in ppe_result.glasses_boxes)

        person.has_helmet = _smooth(person, "helmet", helmet_seen)
        person.has_glasses = _smooth(person, "glasses", glasses_seen)
