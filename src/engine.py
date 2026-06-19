import cv2
import time
from typing import Generator, Optional
import src.config as config
from src.pipeline_types import FrameData
from src.tracker import PersonTracker
from src.compliance import PPEComplianceChecker
from src.environment import EnvironmentBehaviorMonitor
from src.privacy import PrivacyAnonymizer
from src.mock_data import MockPipelineGenerator
from src.ppe import HelmetDetector, helmet_inside_person


class SafetyPipelineEngine:
    def __init__(self, use_mock: bool = False, video_source: Optional[str] = None):
        self.use_mock = use_mock
        self.video_source = video_source
        self.running = False

        print(f"[PipelineEngine] Initializing pipeline. Mock mode: {use_mock}")

        self.tracker_stage = PersonTracker(use_mock=use_mock)
        self.compliance_stage = PPEComplianceChecker(use_mock=use_mock)
        self.environment_stage = EnvironmentBehaviorMonitor(use_mock=use_mock)
        self.privacy_stage = PrivacyAnonymizer(use_mock=use_mock)

        # Helmet detector.
        # For now this points to yolov8n.pt, which does NOT really detect helmets.
        # Later replace with: models/helmet_best.pt
        self.helmet_detector = HelmetDetector("yolov8n.pt")

        if self.use_mock:
            self.mock_generator = MockPipelineGenerator()
            self.cap = None
        else:
            self.mock_generator = None
            src = 0 if video_source is None else video_source
            print(f"[PipelineEngine] Opening video source: {src}")
            self.cap = cv2.VideoCapture(src)

            if not self.cap.isOpened():
                print(f"[PipelineEngine] Error: Could not open video source {src}. Falling back to MOCK mode.")
                self.use_mock = True
                self.mock_generator = MockPipelineGenerator()
                self.cap = None

    def process_frame(self, frame_data: FrameData) -> FrameData:
        # Step 1: Detect and track people
        frame_data = self.tracker_stage.process(frame_data)

        # Step 2: Existing PPE compliance logic
        frame_data = self.compliance_stage.process(frame_data)

        # Step 3: Helmet check
        helmet_boxes = self.helmet_detector.detect_helmets(frame_data.raw_frame)

        for person in frame_data.persons:
            has_helmet = False

            for helmet_box in helmet_boxes:
                if helmet_inside_person(person.bbox, helmet_box):
                    has_helmet = True
                    break

            if not has_helmet:
                if "NO HELMET" not in person.compliance_violations:
                    person.compliance_violations.append("NO HELMET")

        # Step 4: Environment / fall detection
        frame_data = self.environment_stage.process(frame_data)

        # Step 5: Privacy redaction last
        frame_data = self.privacy_stage.process(frame_data)

        return frame_data

    def stream_frames(self) -> Generator[FrameData, None, None]:
        self.running = True
        frame_index = 0

        try:
            while self.running:
                start_time = time.time()

                if self.use_mock:
                    frame_data = self.mock_generator.next_frame()
                else:
                    ret, frame = self.cap.read()

                    if not ret:
                        print("[PipelineEngine] Video stream ended or failed to read frame.")
                        break
                        
                    # Camera Pre-processing (Brightness & Contrast)
                    frame = cv2.convertScaleAbs(
                        frame, 
                        alpha=config.CAMERA_CONTRAST, 
                        beta=config.CAMERA_BRIGHTNESS
                    )
                        
                    # Create default FrameData
                    frame_data = FrameData(
                        frame_index=frame_index,
                        timestamp=time.time(),
                        raw_frame=frame,
                        processed_frame=frame.copy()
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
            f"FPS: {frame_data.extra_metadata['fps']}"
        )

    engine.release()