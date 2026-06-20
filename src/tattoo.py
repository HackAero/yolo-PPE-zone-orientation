import os

import cv2
import numpy as np

import src.config as config


ARM_CHAINS = {
    "left": (5, 7, 9),
    "right": (6, 8, 10),
}

LEG_CHAINS = {
    "left": (11, 13, 15),
    "right": (12, 14, 16),
}


class TattooDetector:
    """Local YOLO segmentation wrapper for tattoo detection in arm crops."""

    def __init__(self, model_path=None, confidence=None):
        from ultralytics import YOLO

        self.model_path = model_path or config.TATTOO_MODEL_PATH
        self.confidence = (
            config.TATTOO_CONF_THRESHOLD
            if confidence is None
            else float(confidence)
        )

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Tattoo model not found: {self.model_path}")
        if os.path.getsize(self.model_path) == 0:
            raise ValueError(f"Tattoo model is empty: {self.model_path}")

        self.model = YOLO(self.model_path)

        # The downloaded checkpoint calls its only class "item". Inside this
        # privacy pipeline, class 0 represents a tattoo.
        names = getattr(getattr(self.model, "model", None), "names", None)
        if isinstance(names, dict) and len(names) == 1:
            self.model.model.names[0] = "tattoo"

    def detect_mask(self, arm_crop):
        """Return a boolean tattoo mask with the same dimensions as arm_crop."""
        if arm_crop is None or arm_crop.size == 0:
            raise ValueError("Cannot run tattoo detection on an empty arm crop")

        crop_h, crop_w = arm_crop.shape[:2]
        result = self.model.predict(
            source=arm_crop,
            conf=self.confidence,
            imgsz=config.TATTOO_INPUT_SIZE,
            retina_masks=True,
            verbose=False,
        )[0]

        combined_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        if result.masks is None:
            return combined_mask.astype(bool)

        for mask_probability in result.masks.data.cpu().numpy():
            if mask_probability.shape != (crop_h, crop_w):
                mask_probability = cv2.resize(
                    mask_probability,
                    (crop_w, crop_h),
                    interpolation=cv2.INTER_LINEAR,
                )
            combined_mask = np.maximum(
                combined_mask,
                (mask_probability >= config.TATTOO_MASK_THRESHOLD).astype(np.uint8),
            )

        # Smooth jagged masks and keep contiguous tattoo-like regions.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

        min_pixels = int(getattr(config, "TATTOO_MIN_MASK_PIXELS", 64))
        if int(combined_mask.sum()) < min_pixels:
            return np.zeros((crop_h, crop_w), dtype=bool)

        return combined_mask.astype(bool)


def _fallback_bbox_rois(person, frame_shape):
    """Fallback ROIs when keypoints are missing: upper arms + forearms + calves."""
    if person is None or person.bbox is None or len(person.bbox) != 4:
        return []

    frame_h, frame_w = frame_shape[:2]
    xmin, ymin, xmax, ymax = person.bbox
    xmin = max(0, min(int(xmin), frame_w - 1))
    xmax = max(0, min(int(xmax), frame_w - 1))
    ymin = max(0, min(int(ymin), frame_h - 1))
    ymax = max(0, min(int(ymax), frame_h - 1))
    if xmax <= xmin or ymax <= ymin:
        return []

    pw = xmax - xmin
    ph = ymax - ymin

    # Approximate bilateral limb strips within a standing person bbox.
    left_x1 = xmin + int(pw * 0.05)
    left_x2 = xmin + int(pw * 0.33)
    right_x1 = xmin + int(pw * 0.67)
    right_x2 = xmin + int(pw * 0.95)

    upper_y1 = ymin + int(ph * 0.22)
    upper_y2 = ymin + int(ph * 0.45)
    fore_y1 = ymin + int(ph * 0.45)
    fore_y2 = ymin + int(ph * 0.70)
    calf_y1 = ymin + int(ph * 0.70)
    calf_y2 = ymin + int(ph * 0.95)

    candidates = [
        ("left", "upper_arm", [left_x1, upper_y1, left_x2, upper_y2]),
        ("left", "forearm", [left_x1, fore_y1, left_x2, fore_y2]),
        ("right", "upper_arm", [right_x1, upper_y1, right_x2, upper_y2]),
        ("right", "forearm", [right_x1, fore_y1, right_x2, fore_y2]),
        ("left", "calf", [left_x1, calf_y1, left_x2, calf_y2]),
        ("right", "calf", [right_x1, calf_y1, right_x2, calf_y2]),
    ]

    rois = []
    for side, segment, bbox in candidates:
        x1, y1, x2, y2 = bbox
        if x2 > x1 and y2 > y1:
            rois.append({"side": side, "segment": segment, "bbox": bbox})
    return rois


def _segment_bbox(
    point1,
    point2,
    frame_shape,
    padding_ratio,
    min_padding,
):
    """Build an axis-aligned ROI around one limb segment."""
    frame_h, frame_w = frame_shape[:2]
    x1, y1 = point1
    x2, y2 = point2

    segment_length = np.linalg.norm(np.array([x2 - x1, y2 - y1]))
    padding = max(
        min_padding,
        int(segment_length * padding_ratio),
    )

    roi_xmin = max(0, int(min(x1, x2) - padding))
    roi_ymin = max(0, int(min(y1, y2) - padding))
    roi_xmax = min(frame_w, int(max(x1, x2) + padding))
    roi_ymax = min(frame_h, int(max(y1, y2) + padding))

    if roi_xmax <= roi_xmin or roi_ymax <= roi_ymin:
        return None
    return [roi_xmin, roi_ymin, roi_xmax, roi_ymax]


def _build_limb_rois(
    person,
    frame_shape,
    chains,
    segment_names,
    padding_ratio,
    min_padding,
):
    """Return reliable limb segment ROIs in full-frame coordinates."""
    if person.keypoints is None:
        return []

    keypoints = person.keypoints
    confidences = person.metadata.get("keypoint_confidence")
    threshold = config.KEYPOINT_CONFIDENCE_THRESHOLD

    def is_visible(index):
        if index >= len(keypoints):
            return False

        x, y = keypoints[index]
        if x <= 0 or y <= 0:
            return False

        if confidences is not None:
            if index >= len(confidences):
                return False
            return float(confidences[index]) >= threshold
        return True

    rois = []
    for side, joint_chain in chains.items():
        segments = (
            (segment_names[0], joint_chain[0], joint_chain[1]),
            (segment_names[1], joint_chain[1], joint_chain[2]),
        )

        for segment_name, start, end in segments:
            if not is_visible(start) or not is_visible(end):
                continue

            bbox = _segment_bbox(
                keypoints[start],
                keypoints[end],
                frame_shape,
                padding_ratio,
                min_padding,
            )
            if bbox is not None:
                rois.append({
                    "side": side,
                    "segment": segment_name,
                    "bbox": bbox,
                })

    return rois


def build_arm_rois(person, frame_shape):
    """Return upper-arm and forearm ROIs in full-frame coordinates."""
    return _build_limb_rois(
        person,
        frame_shape,
        ARM_CHAINS,
        ("upper_arm", "forearm"),
        config.ARM_ROI_PADDING_RATIO,
        config.ARM_ROI_MIN_PADDING,
    )


def build_leg_rois(person, frame_shape):
    """Return thigh and calf ROIs in full-frame coordinates."""
    return _build_limb_rois(
        person,
        frame_shape,
        LEG_CHAINS,
        ("thigh", "calf"),
        config.LEG_ROI_PADDING_RATIO,
        config.LEG_ROI_MIN_PADDING,
    )


def build_tattoo_rois(person, frame_shape):
    """Return every arm and leg ROI inspected by the tattoo detector."""
    rois = build_arm_rois(person, frame_shape) + build_leg_rois(person, frame_shape)
    if rois:
        return rois
    if bool(getattr(config, "TATTOO_FALLBACK_BBOX_ROIS", True)):
        return _fallback_bbox_rois(person, frame_shape)
    return []
