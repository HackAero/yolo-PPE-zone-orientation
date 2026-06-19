import cv2
import numpy as np
from typing import Dict, List, Optional
import supervision as sv
from ultralytics import YOLO
from src.types import FrameData, TrackedPerson
from src.config import YOLO_MODEL_PATH, PERSON_CONF_THRESHOLD, REID_COSINE_SIMILARITY_THRESHOLD

class PersonTracker:
    def __init__(self, use_mock: bool = False):
        """
        Manages person detection, temporal tracking, and Re-Identification.
        """
        self.use_mock = use_mock
        if not use_mock:
            print(f"[Tracker] Initializing YOLO model: {YOLO_MODEL_PATH}")
            self.model = YOLO(YOLO_MODEL_PATH)
            self.tracker = sv.ByteTrack()
            
            # Re-ID database: maps persistent_id -> list of visual embeddings
            self.embedding_cache: Dict[int, List[np.ndarray]] = {}
            
            # Session maps: maps temporary byte_track ID to persistent_id
            self.track_id_to_persistent_id: Dict[int, int] = {}
            
            # Running counter for new unique people
            self.next_persistent_id = 1
        else:
            print("[Tracker] Running in MOCK mode.")
            self.model = None
            self.tracker = None
            self.next_persistent_id = 1

    def _extract_embedding(self, crop: np.ndarray) -> np.ndarray:
        """
        Extracts a color-histogram signature vector from the cropped image.
        This serves as a fast, light Re-ID model.
        In a real production pipeline, you'd replace this with a deep learning
        ReID embedding model (e.g. fast-reid, torchreid, or OSNet).
        """
        if crop.size == 0:
            return np.zeros(64, dtype=np.float32)
            
        # Convert crop to HSV to be more robust to minor lighting changes
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Calculate histograms for H, S, and V channels
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])
        
        # Concatenate and normalize
        hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()
        cv2.normalize(hist, hist)
        return hist

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Computes cosine similarity between two feature vectors.
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _match_reid(self, new_embedding: np.ndarray) -> Optional[int]:
        """
        Queries the embedding cache to find if this visual signature matches
        a previously seen person who left the screen.
        """
        best_match_id = None
        best_score = -1.0
        
        for pid, embeddings in self.embedding_cache.items():
            # Check against stored embeddings for this ID
            for cached_emb in embeddings:
                sim = self._cosine_similarity(new_embedding, cached_emb)
                if sim > best_score:
                    best_score = sim
                    best_match_id = pid
                    
        # Verify if similarity is above the configuration threshold
        if best_score >= REID_COSINE_SIMILARITY_THRESHOLD:
            return best_match_id
        return None

    def process(self, frame_data: FrameData) -> FrameData:
        """
        Conveyor belt stage:
        Takes frame_data, runs detection + tracking, and updates the list of TrackedPerson.
        """
        if self.use_mock:
            # Mock mode: The mock_data module will populate the people and totals.
            # We just ensure the engine's tracking counts match what's in the data.
            if len(frame_data.persons) > 0:
                frame_data.current_people_count = len(frame_data.persons)
                max_id = max(p.person_id for p in frame_data.persons)
                frame_data.total_unique_people = max(frame_data.total_unique_people, max_id)
            return frame_data

        # --- REAL DETECTION AND TRACKING PIPELINE ---
        frame = frame_data.raw_frame
        
        # Run inference
        results = self.model(frame, conf=PERSON_CONF_THRESHOLD, verbose=False)[0]
        
        # Convert results to supervision format
        detections = sv.Detections.from_ultralytics(results)
        
        # Class 0 in COCO is "person"
        detections = detections[detections.class_id == 0]
        
        # Update the temporal ByteTrack tracker
        detections = self.tracker.update_with_detections(detections)
        
        active_persons = []
        
        if detections.tracker_id is not None and len(detections.tracker_id) > 0:
            for i, track_id in enumerate(detections.tracker_id):
                bbox = detections.xyxy[i].astype(int).tolist()  # [xmin, ymin, xmax, ymax]
                conf = float(detections.confidence[i])
                
                # Get person crop
                xmin, ymin, xmax, ymax = bbox
                # Ensure coordinates are within frame bounds
                h_max, w_max = frame.shape[:2]
                xmin = max(0, min(xmin, w_max - 1))
                ymin = max(0, min(ymin, h_max - 1))
                xmax = max(0, min(xmax, w_max - 1))
                ymax = max(0, min(ymax, h_max - 1))
                
                crop = frame[ymin:ymax, xmin:xmax]
                embedding = self._extract_embedding(crop)
                
                # Resolve ByteTrack ID to persistent ID using ReID cache
                if track_id in self.track_id_to_persistent_id:
                    # Person is continuously tracked in session
                    pid = self.track_id_to_persistent_id[track_id]
                    # Update their visual signature cache
                    self.embedding_cache[pid].append(embedding)
                    if len(self.embedding_cache[pid]) > 15:
                        self.embedding_cache[pid].pop(0)
                else:
                    # New track ID detected. Try to match it to a past persistent_id (re-entry check)
                    matched_pid = self._match_reid(embedding)
                    if matched_pid is not None:
                        # Re-identified! Bind the tracker ID to this persistent ID
                        pid = matched_pid
                        self.track_id_to_persistent_id[track_id] = pid
                        self.embedding_cache[pid].append(embedding)
                    else:
                        # New unique person
                        pid = self.next_persistent_id
                        self.next_persistent_id += 1
                        self.track_id_to_persistent_id[track_id] = pid
                        self.embedding_cache[pid] = [embedding]
                
                person = TrackedPerson(
                    person_id=pid,
                    bbox=bbox,
                    confidence=conf,
                    embedding=embedding
                )
                active_persons.append(person)
        
        frame_data.persons = active_persons
        frame_data.current_people_count = len(active_persons)
        frame_data.total_unique_people = self.next_persistent_id - 1
        
        return frame_data

if __name__ == "__main__":
    # Test script for isolated testing
    print("Testing TrackerStage in Isolation...")
    import time
    
    # Initialize mock tracker
    tracker = PersonTracker(use_mock=True)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    data = FrameData(frame_index=0, timestamp=time.time(), raw_frame=dummy_frame, processed_frame=dummy_frame.copy())
    
    # Run test
    out = tracker.process(data)
    print(f"Mock test completed. People count: {out.current_people_count}, Unique total: {out.total_unique_people}")
