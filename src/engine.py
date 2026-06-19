import cv2
import time
from typing import Generator, Optional
from src.types import FrameData
from src.tracker import PersonTracker
from src.compliance import PPEComplianceChecker
from src.environment import EnvironmentBehaviorMonitor
from src.privacy import PrivacyAnonymizer
from src.mock_data import MockPipelineGenerator

class SafetyPipelineEngine:
    def __init__(self, use_mock: bool = False, video_source: Optional[str] = None):
        """
        Coordinates the pipeline execution. Manages stage sequencing,
        frame ingestion, and timing metrics.
        """
        self.use_mock = use_mock
        self.video_source = video_source
        self.running = False
        
        print(f"[PipelineEngine] Initializing pipeline. Mock mode: {use_mock}")
        
        # Initialize pipeline stages (modular conveyor belt)
        self.tracker_stage = PersonTracker(use_mock=use_mock)
        self.compliance_stage = PPEComplianceChecker(use_mock=use_mock)
        self.environment_stage = EnvironmentBehaviorMonitor(use_mock=use_mock)
        self.privacy_stage = PrivacyAnonymizer(use_mock=use_mock)
        
        # Initialize sources
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
        """
        Conveyor Belt Processing Sequence.
        Each module processes the frame data in-place.
        """
        # Step 1: Detect & track people (creates TrackedPerson objects)
        frame_data = self.tracker_stage.process(frame_data)
        
        # Step 2: Check safety compliance (helmet, glasses)
        frame_data = self.compliance_stage.process(frame_data)
        
        # Step 3: Check environment factors (blur, smoke) and human posture (fall detection)
        frame_data = self.environment_stage.process(frame_data)
        
        # Step 4: Apply privacy filters (blur faces/tattoos on processed_frame)
        # Note: Privacy runs last to guarantee all sensitive data is redacted before display
        frame_data = self.privacy_stage.process(frame_data)
        
        return frame_data

    def stream_frames(self) -> Generator[FrameData, None, None]:
        """
        Generator yielding fully processed FrameData packages from webcam or mock.
        """
        self.running = True
        frame_index = 0
        
        try:
            while self.running:
                start_time = time.time()
                
                if self.use_mock:
                    # In mock mode, raw frame and base persons are generated in mock_data.py
                    frame_data = self.mock_generator.next_frame()
                else:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("[PipelineEngine] Video stream ended or failed to read frame.")
                        break
                        
                    # Create default FrameData
                    frame_data = FrameData(
                        frame_index=frame_index,
                        timestamp=time.time(),
                        raw_frame=frame,
                        processed_frame=frame.copy()
                    )
                
                # Push through conveyor belt
                frame_data = self.process_frame(frame_data)
                
                # Add processing latency information
                latency = time.time() - start_time
                frame_data.extra_metadata["latency_ms"] = int(latency * 1000)
                frame_data.extra_metadata["fps"] = int(1.0 / latency) if latency > 0 else 0
                
                yield frame_data
                frame_index += 1
                
        finally:
            self.release()

    def release(self):
        """
        Clean up video capture objects.
        """
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("[PipelineEngine] Video resource released.")

if __name__ == "__main__":
    print("Testing PipelineEngine with Mock Stream...")
    engine = SafetyPipelineEngine(use_mock=True)
    stream = engine.stream_frames()
    
    # Process 5 mock frames as a smoke test
    for _ in range(5):
        frame_data = next(stream)
        print(f"Frame {frame_data.frame_index} processed. "
              f"People: {frame_data.current_people_count}, "
              f"Alerts triggered: {len(frame_data.alerts)}, "
              f"FPS: {frame_data.extra_metadata['fps']}")
    
    engine.release()
