"""
Safety pipeline engine — runs each frame through a fixed sequence of stages.

Order matters: PPE and compliance run before zones (zones need helmet/glasses/vest
flags), environment and falls after tracking, privacy last so the displayed frame
is blurred. MQTT dispatch is triggered from zones and from confirmed env alerts.
"""
import cv2
import time
from typing import Generator, Optional

import src.config as config
from src.capture import LatestFrameGrabber
from src.compliance import PPEComplianceChecker
from src.environment import EnvironmentBehaviorMonitor
from src.face_detection import SharedFaceDetector
from src.frame_utils import preprocess_camera_frame
from src.dispatcher import RobotDispatcher
from src.mock_data import MockPipelineGenerator
from src.pipeline_types import FrameData
from src.ppe_inference import SharedPPEDetector
from src.privacy import PrivacyAnonymizer
from src.alert_filter import AlertVerificationFilter
from src.tracker import PersonTracker
from src.zone_map import ZoneMonitor


class SafetyPipelineEngine:
    def __init__(
        self,
        use_mock: bool = False,
        video_source: Optional[str] = None,
        zones_path: Optional[str] = None,
        camera_profile: Optional[str] = None,
    ):
        self.use_mock = use_mock
        self.video_source = video_source
        self.running = False

        resolved_zones = config.resolve_zones_path(
            profile=camera_profile,
            explicit_path=zones_path,
        )

        mode = "FAST" if config.FAST_MODE else "SMOOTH" if config.SMOOTH_MODE else "STANDARD"
        print(
            f"[PipelineEngine] Initializing pipeline. Mock mode: {use_mock}, mode: {mode}, "
            f"zones: {resolved_zones}"
        )

        self.face_detector = SharedFaceDetector(use_mock=use_mock)
        self.tracker_stage = PersonTracker(use_mock=use_mock)
        self.ppe_detector = SharedPPEDetector(use_mock=use_mock)
        self.compliance_stage = PPEComplianceChecker(use_mock=use_mock)
        self.environment_stage = EnvironmentBehaviorMonitor(use_mock=use_mock)
        self.privacy_stage = PrivacyAnonymizer(use_mock=use_mock)
        self.zone_monitor = ZoneMonitor(
            zones_path=resolved_zones,
            camera_profile=camera_profile or config.ZONES_PROFILE,
        )
        self.dispatcher = RobotDispatcher()
        self.alert_filter = AlertVerificationFilter()
        self._degraded_blur_streak = 0
        self._frame_grabber: Optional[LatestFrameGrabber] = None

        if self.use_mock:
            self.mock_generator = MockPipelineGenerator()
            self.cap = None
        else:
            self.mock_generator = None
            src = 0 if video_source is None else video_source
            print(f"[PipelineEngine] Opening video source: {src}")
            self.cap = cv2.VideoCapture(src)
            if self.cap.isOpened() and config.CAMERA_MAX_WIDTH > 0:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_MAX_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(config.CAMERA_MAX_WIDTH * 0.75))
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self.cap.isOpened():
                print(
                    f"[PipelineEngine] Error: Could not open video source {src}. "
                    "Falling back to MOCK mode."
                )
                self.use_mock = True
                self.mock_generator = MockPipelineGenerator()
                self.cap = None
            elif config.ASYNC_FRAME_GRAB:
                self._frame_grabber = LatestFrameGrabber(self.cap)

        if not self.use_mock:
            self._warmup_models()

    def _warmup_models(self) -> None:
        """Prime YOLO/MediaPipe with a tiny dummy frame to avoid first-frame spikes."""
        import numpy as np

        dummy = np.zeros((config.YOLO_IMGSZ, config.YOLO_IMGSZ, 3), dtype=np.uint8)
        try:
            self.tracker_stage.model(dummy, imgsz=config.YOLO_IMGSZ, device=config.YOLO_DEVICE, verbose=False)
        except Exception:
            pass
        try:
            if self.ppe_detector.model is not None:
                self.ppe_detector.model(
                    dummy,
                    imgsz=config.YOLO_IMGSZ,
                    device=config.YOLO_DEVICE,
                    verbose=False,
                )
        except Exception:
            pass

    def _ensure_processed_frame(self, frame_data: FrameData) -> None:
        if frame_data.processed_frame is None or frame_data.processed_frame.size == 0:
            frame_data.processed_frame = frame_data.raw_frame

    def process_frame(self, frame_data: FrameData) -> FrameData:
        stage_ms = {}
        frame_data.extra_metadata["stage_ms"] = stage_ms

        t0 = time.time()
        frame_data = self.tracker_stage.process(frame_data)
        stage_ms["tracker"] = int((time.time() - t0) * 1000)

        if not self.use_mock:
            t0 = time.time()
            ppe_result = self.ppe_detector.detect_all(
                frame_data.raw_frame,
                persons=frame_data.persons,
            )
            stage_ms["ppe_yolo"] = int((time.time() - t0) * 1000)
            stage_ms["ppe_assign"] = 0
            frame_data.extra_metadata["ppe_debug"] = {
                "helmet_boxes": ppe_result.helmet_boxes,
                "glasses_boxes": ppe_result.glasses_boxes,
                "raw_detections": ppe_result.raw_detections,
                "model_available": ppe_result.model_available,
            }

            if not ppe_result.model_available:
                for person in frame_data.persons:
                    person.has_helmet = None
                    person.has_glasses = None

            t0 = time.time()
            self.face_detector.populate_frame_cache(frame_data)
            stage_ms["face_detect"] = int((time.time() - t0) * 1000)
        else:
            stage_ms["ppe_yolo"] = 0
            stage_ms["ppe_assign"] = 0
            stage_ms["face_detect"] = 0

        t0 = time.time()
        frame_data = self.compliance_stage.process(frame_data)
        stage_ms["compliance_heuristics"] = int((time.time() - t0) * 1000)

        t0 = time.time()
        frame_data = self.zone_monitor.process(frame_data, dispatcher=None)
        stage_ms["zones"] = int((time.time() - t0) * 1000)

        t0 = time.time()
        frame_data = self.environment_stage.process(frame_data)
        stage_ms["environment"] = int((time.time() - t0) * 1000)

        self._ensure_processed_frame(frame_data)

        t0 = time.time()
        frame_data = self.privacy_stage.process(frame_data)
        stage_ms["privacy"] = int((time.time() - t0) * 1000)

        t0 = time.time()
        frame_data = self.alert_filter.apply(frame_data)
        stage_ms["alert_verify"] = int((time.time() - t0) * 1000)

        self._update_reliability_mode(frame_data)

        slowest = max(stage_ms, key=stage_ms.get) if stage_ms else "unknown"
        frame_data.extra_metadata["slowest_stage"] = slowest

        self._dispatch_confirmed_alerts(frame_data)
        
        return frame_data

    def _update_reliability_mode(self, frame_data: FrameData) -> None:
        if frame_data.is_image_blurry:
            self._degraded_blur_streak += 1
        else:
            self._degraded_blur_streak = max(0, self._degraded_blur_streak - 1)

        verifying_count = int(frame_data.extra_metadata.get("verifying_count", 0) or 0)
        limited_by_blur = self._degraded_blur_streak >= max(1, config.RELIABILITY_BLUR_FRAMES)
        limited_by_noise = verifying_count >= max(1, config.RELIABILITY_VERIFYING_THRESHOLD)
        is_limited = limited_by_blur or limited_by_noise

        reasons = []
        if limited_by_blur:
            reasons.append("blur")
        if limited_by_noise:
            reasons.append("low_confidence")

        frame_data.extra_metadata["reliability_mode"] = "LIMITED" if is_limited else "NORMAL"
        frame_data.extra_metadata["reliability_reason"] = ",".join(reasons) if reasons else "stable"

    def _dispatch_confirmed_alerts(self, frame_data: FrameData) -> None:
        is_limited = frame_data.extra_metadata.get("reliability_mode") == "LIMITED"
        suppress_warnings = bool(config.RELIABILITY_SUPPRESS_WARNING_DISPATCH)

        for alert in frame_data.alerts:
            alert_type = str(alert.get("type", ""))
            person_id_raw = alert.get("person_id", -1)
            try:
                person_id = int(person_id_raw)
            except (TypeError, ValueError):
                person_id = -1
            zone_id = alert.get("zone_id")

            if alert_type == "FALL_ALERT":
                self.dispatcher.send(alert_type="FALL_DETECTED", person_id=person_id, zone_id=zone_id)
                continue

            if alert_type == "RESTRICTED_ENTRY":
                self.dispatcher.send(alert_type="RESTRICTED_ENTRY", person_id=person_id, zone_id=zone_id)
                continue

            if alert_type in {"PPE_VIOLATION", "ZONE_PPE_VIOLATION"}:
                if is_limited and suppress_warnings:
                    continue
                msg = str(alert.get("message", "")).lower()
                if "helmet" in msg:
                    self.dispatcher.send(alert_type="NO_HELMET", person_id=person_id, zone_id=zone_id)
                elif "glass" in msg:
                    self.dispatcher.send(alert_type="NO_GLASSES", person_id=person_id, zone_id=zone_id)
                continue

            if alert_type == "ENVIRONMENT_ALERT":
                msg = str(alert.get("message", "")).upper()
                if "SMOKE" in msg:
                    self.dispatcher.send(alert_type="SMOKE_DETECTED", person_id=-1)
                elif "FIRE" in msg:
                    self.dispatcher.send(alert_type="FIRE_DETECTED", person_id=-1)

    
    def cycle_zone_layout(self) -> str:
        """Cycle: 3 zones -> full safe -> full work -> full restricted."""
        return self.zone_monitor.cycle_layout()

    def _read_camera_frame(self):
        if self._frame_grabber is not None:
            return self._frame_grabber.read()

        ret, frame = self.cap.read()
        if not ret:
            return False, None
        return True, preprocess_camera_frame(frame)

    def stream_frames(self) -> Generator[FrameData, None, None]:
        self.running = True
        frame_index = 0

        try:
            while self.running:
                start_time = time.time()

                if self.use_mock:
                    frame_data = self.mock_generator.next_frame()
                else:
                    ret, frame = self._read_camera_frame()
                    if not ret or frame is None:
                        # Async grabber may need a moment to deliver the first frame.
                        for _ in range(40):
                            time.sleep(0.05)
                            ret, frame = self._read_camera_frame()
                            if ret and frame is not None:
                                break

                    if not ret or frame is None:
                        print("[PipelineEngine] Video stream ended or failed to read frame.")
                        break

                    frame_data = FrameData(
                        frame_index=frame_index,
                        timestamp=time.time(),
                        raw_frame=frame,
                        processed_frame=frame,
                    )

                frame_data = self.process_frame(frame_data)

                latency = time.time() - start_time
                frame_data.extra_metadata["latency_ms"] = int(latency * 1000)
                frame_data.extra_metadata["fps"] = int(1.0 / latency) if latency > 0 else 0

                yield frame_data
                frame_index += 1

        finally:
            self.release()

    def release(self):
        self.running = False
        self.dispatcher.shutdown()

        if self._frame_grabber is not None:
            self._frame_grabber.stop()
            self._frame_grabber = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("[PipelineEngine] Video resource released.")


if __name__ == "__main__":
    print("Testing PipelineEngine with Mock Stream...")
    engine = SafetyPipelineEngine(use_mock=True)
    stream = engine.stream_frames()

    for _ in range(5):
        frame_data = next(stream)
        print(
            f"Frame {frame_data.frame_index} processed. "
            f"People: {frame_data.current_people_count}, "
            f"Alerts triggered: {len(frame_data.alerts)}, "
            f"FPS: {frame_data.extra_metadata['fps']}, "
            f"Stages: {frame_data.extra_metadata.get('stage_ms', {})}"
        )

    engine.release()
