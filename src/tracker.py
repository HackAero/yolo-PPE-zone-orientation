import cv2
import numpy as np
from collections import deque
from typing import Deque, Dict, List, Optional, Set
import supervision as sv
from ultralytics import YOLO
from src.pipeline_types import FrameData, TrackedPerson
import src.config as config

# Try importing PyTorch for advanced Re-ID features
try:
    import torch
    import torchvision.transforms as T
    import torchvision.models as models
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class PersonTracker:
    def __init__(self, use_mock: bool = False):
        """
        Manages person detection, temporal tracking, and Re-Identification.
        Uses a pre-trained CNN (MobileNetV3) for semantic Re-ID embeddings.
        Falls back to spatial HSV color histograms if PyTorch is unavailable or fails.
        """
        self.use_mock = use_mock
        self._frame_counter = 0
        self._track_stable_frames: Dict[int, int] = {}
        self._last_embeddings: Dict[int, np.ndarray] = {}
        self._last_persons_snapshot: List[TrackedPerson] = []
        if not use_mock:
            print(f"[Tracker] Initializing YOLO model: {config.YOLO_MODEL_PATH}")
            self.model = YOLO(config.YOLO_MODEL_PATH)
            self.tracker = sv.ByteTrack()
            
            # Re-ID database: maps persistent_id -> deque of visual embeddings (max 50, auto-evicts oldest)
            self.embedding_cache: Dict[int, Deque[np.ndarray]] = {}
            
            # Session maps: maps temporary byte_track ID to persistent_id
            self.track_id_to_persistent_id: Dict[int, int] = {}
            
            # Running counter for new unique people (used only for ID assignment)
            self.next_persistent_id = 1
            
            # Ground-truth set of all unique person IDs ever confirmed — used for cumulative count
            self.confirmed_ids: Set[int] = set()
            
            # Initialize MobileNetV3 CNN Re-ID model
            self.reid_model = None
            if HAS_TORCH:
                try:
                    print("[Tracker] Loading MobileNetV3 Re-ID feature extractor (ImageNet weights)...")
                    # Load model
                    full_model = models.mobilenet_v3_small(weights='DEFAULT')
                    # Use only the feature extractor layers
                    self.reid_model = full_model.features
                    self.reid_model.eval()
                    
                    # Select GPU if available, CPU is very fast for single crops
                    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    self.reid_model.to(self.device)
                    
                    # Define ImageNet standard preprocessing transforms
                    self.transform = T.Compose([
                        T.ToPILImage(),
                        T.Resize((224, 224)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    print(f"[Tracker] Re-ID CNN loaded successfully on device: {self.device}")
                except Exception as e:
                    print(f"[Tracker] Error initializing CNN model: {e}. Falling back to Spatial Color Histograms.")
                    self.reid_model = None
            else:
                print("[Tracker] PyTorch/Torchvision unavailable. Falling back to Spatial Color Histograms.")
        else:
            print("[Tracker] Running in MOCK mode.")
            self.model = None
            self.tracker = None
            self.next_persistent_id = 1

    def _extract_embedding(self, crop: np.ndarray) -> np.ndarray:
        """
        Extracts a feature descriptor vector from the crop.
        Combines Semantic CNN features (shape/texture) with Spatial HSV features (color).
        """
        dim_cnn = 576 if self.reid_model is not None else 0
        dim_hsv = 192
        
        if crop.size == 0:
            return np.zeros(dim_cnn + dim_hsv, dtype=np.float32)

        # --- Method A: Deep Learning CNN Embeddings (Shape/Texture) ---
        vector_cnn = None
        if self.reid_model is not None:
            try:
                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                tensor = self.transform(rgb_crop).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    features = self.reid_model(tensor)
                    pooled = torch.nn.functional.adaptive_avg_pool2d(features, 1)
                    vector_cnn = pooled.flatten().cpu().numpy()
                norm = np.linalg.norm(vector_cnn)
                if norm > 0:
                    vector_cnn = vector_cnn / norm
            except Exception:
                vector_cnn = np.zeros(dim_cnn, dtype=np.float32)
        else:
            vector_cnn = np.array([], dtype=np.float32)

        # --- Method B: Spatial HSV Histogram (Color) ---
        h, w = crop.shape[:2]
        xmin_c = int(w * 0.2)
        xmax_c = int(w * 0.8)
        center_crop = crop[:, xmin_c:max(xmin_c + 1, xmax_c)]
        
        hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
        ch, cw = hsv.shape[:2]
        
        top_split = int(ch * 0.3)
        bottom_split = int(ch * 0.7)
        
        zones = [
            hsv[0:top_split, :],
            hsv[top_split:bottom_split, :],
            hsv[bottom_split:ch, :]
        ]
        
        hist_parts = []
        for zone in zones:
            if zone.size == 0:
                hist_parts.append(np.zeros(64, dtype=np.float32))
                continue
            hist_h = cv2.calcHist([zone], [0], None, [32], [0, 180])
            hist_s = cv2.calcHist([zone], [1], None, [16], [0, 256])
            hist_v = cv2.calcHist([zone], [2], None, [16], [0, 256])
            
            zone_hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()
            norm = np.linalg.norm(zone_hist)
            if norm > 0:
                zone_hist = zone_hist / norm
            hist_parts.append(zone_hist)
            
        vector_hsv = np.concatenate(hist_parts)
        full_norm = np.linalg.norm(vector_hsv)
        if full_norm > 0:
            vector_hsv = vector_hsv / full_norm
            
        # --- Unify ---
        if vector_cnn.size > 0:
            unified = np.concatenate([vector_cnn, vector_hsv])
            unified_norm = np.linalg.norm(unified)
            if unified_norm > 0:
                unified = unified / unified_norm
            return unified
        else:
            return vector_hsv

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Computes cosine similarity between two feature vectors.
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _match_reid(self, new_embedding: np.ndarray, exclude_ids: Set[int]) -> Optional[int]:
        """
        Queries the embedding cache to find if this visual signature matches
        a previously seen person who is currently NOT visible in the frame.
        
        Uses top-3 mean scoring per candidate to reduce noise from outlier
        embeddings — a single lucky match is no longer enough to claim a re-ID.
        """
        best_match_id = None
        best_score = -1.0
        
        threshold = config.REID_COSINE_SIMILARITY_THRESHOLD
        
        for pid, embeddings in self.embedding_cache.items():
            # Skip matching if this person ID is currently visible in the room
            if pid in exclude_ids:
                continue
            
            # Compute cosine similarity against every stored embedding for this person
            scores = [self._cosine_similarity(new_embedding, cached_emb) for cached_emb in embeddings]
            
            # Rank scores and average the top-3 to smooth out noise
            top_scores = sorted(scores, reverse=True)[:3]
            candidate_score = sum(top_scores) / len(top_scores)
            
            if candidate_score > best_score:
                best_score = candidate_score
                best_match_id = pid
                    
        if best_score >= threshold:
            return best_match_id
        return None

    def process(self, frame_data: FrameData) -> FrameData:
        """
        Conveyor belt stage:
        Takes frame_data, runs detection + tracking, and updates the list of TrackedPerson.
        """
        if self.use_mock:
            if len(frame_data.persons) > 0:
                frame_data.current_people_count = len(frame_data.persons)
                max_id = max(p.person_id for p in frame_data.persons)
                frame_data.total_unique_people = max(frame_data.total_unique_people, max_id)
            return frame_data

        # --- REAL DETECTION AND TRACKING PIPELINE ---
        frame = frame_data.raw_frame

        self._frame_counter += 1
        run_detection = (self._frame_counter - 1) % config.PERSON_DETECT_INTERVAL == 0

        if not run_detection and self._last_persons_snapshot:
            frame_data.persons = [
                TrackedPerson(
                    person_id=person.person_id,
                    bbox=list(person.bbox),
                    confidence=person.confidence,
                    has_helmet=person.has_helmet,
                    has_glasses=person.has_glasses,
                    compliance_violations=list(person.compliance_violations),
                    is_fallen=person.is_fallen,
                    keypoints=person.keypoints,
                    embedding=person.embedding,
                    reid_matched=person.reid_matched,
                    metadata=dict(person.metadata),
                )
                for person in self._last_persons_snapshot
            ]
            frame_data.current_people_count = len(frame_data.persons)
            frame_data.total_unique_people = len(self.confirmed_ids)
            return frame_data
        
        # Run inference
        results = self.model(
            frame,
            conf=config.PERSON_CONF_THRESHOLD,
            imgsz=config.YOLO_IMGSZ,
            device=config.YOLO_DEVICE,
            verbose=False,
        )[0]
        
        # Convert results to supervision format
        detections = sv.Detections.from_ultralytics(results)
        
        # Class 0 in COCO is "person"
        detections = detections[detections.class_id == 0]
        
        # Update the temporal ByteTrack tracker
        detections = self.tracker.update_with_detections(detections)
        
        active_persons = []
        
        # 1. Identify which persistent IDs are ALREADY mapped in this frame.
        # These are marked as active to prevent other tracks from matching them.
        active_persistent_ids: Set[int] = set()
        if detections.tracker_id is not None and len(detections.tracker_id) > 0:
            for track_id in detections.tracker_id:
                if track_id in self.track_id_to_persistent_id:
                    active_persistent_ids.add(self.track_id_to_persistent_id[track_id])
        
        # 2. Match and resolve track IDs
        if detections.tracker_id is not None and len(detections.tracker_id) > 0:
            for i, track_id in enumerate(detections.tracker_id):
                bbox = detections.xyxy[i].astype(int).tolist()  # [xmin, ymin, xmax, ymax]
                conf = float(detections.confidence[i])
                
                # Get person crop
                xmin, ymin, xmax, ymax = bbox
                h_max, w_max = frame.shape[:2]
                xmin = max(0, min(xmin, w_max - 1))
                ymin = max(0, min(ymin, h_max - 1))
                xmax = max(0, min(xmax, w_max - 1))
                ymax = max(0, min(ymax, h_max - 1))
                
                crop = frame[ymin:ymax, xmin:xmax]

                if track_id in self.track_id_to_persistent_id:
                    pid = self.track_id_to_persistent_id[track_id]
                    stable_frames = self._track_stable_frames.get(track_id, 0) + 1
                    self._track_stable_frames[track_id] = stable_frames

                    skip_reid = (
                        stable_frames >= config.REID_STABLE_TRACK_FRAMES
                        and (self._frame_counter - 1) % config.REID_INFERENCE_INTERVAL != 0
                        and pid in self._last_embeddings
                    )

                    if skip_reid:
                        embedding = self._last_embeddings[pid]
                    else:
                        embedding = self._extract_embedding(crop)
                        self._last_embeddings[pid] = embedding
                else:
                    embedding = self._extract_embedding(crop)

                # Resolve ByteTrack ID to persistent ID using ReID cache
                if track_id in self.track_id_to_persistent_id:
                    pid = self.track_id_to_persistent_id[track_id]
                    
                    # Diverse Caching: Only add new signature if it's visually distinct (> 0.92 similar means too identical)
                    # The deque has maxlen=50 and evicts the oldest entry automatically — no manual pop needed
                    is_distinct = True
                    for cached_emb in self.embedding_cache[pid]:
                        if self._cosine_similarity(embedding, cached_emb) > 0.92:
                            is_distinct = False
                            break
                            
                    if is_distinct:
                        self.embedding_cache[pid].append(embedding)
                            
                else:
                    # New track ID detected. Try to match it to a past persistent_id (re-entry check)
                    # We pass active_persistent_ids to avoid collisions with active workers
                    matched_pid = self._match_reid(embedding, active_persistent_ids)
                    if matched_pid is not None:
                        # Re-identified! Bind the tracker ID to this persistent ID
                        pid = matched_pid
                        self.track_id_to_persistent_id[track_id] = pid
                        
                        is_distinct = True
                        for cached_emb in self.embedding_cache[pid]:
                            if self._cosine_similarity(embedding, cached_emb) > 0.92:
                                is_distinct = False
                                break
                        if is_distinct:
                            self.embedding_cache[pid].append(embedding)  # deque auto-evicts at maxlen=50
                                
                        active_persistent_ids.add(pid)
                        self.confirmed_ids.add(pid)
                        self._last_embeddings[pid] = embedding
                    else:
                        pid = self.next_persistent_id
                        self.next_persistent_id += 1
                        self.track_id_to_persistent_id[track_id] = pid
                        self.embedding_cache[pid] = deque([embedding], maxlen=50)
                        active_persistent_ids.add(pid)
                        self.confirmed_ids.add(pid)
                        self._track_stable_frames[track_id] = 0
                        self._last_embeddings[pid] = embedding
                
                person = TrackedPerson(
                    person_id=pid,
                    bbox=bbox,
                    confidence=conf,
                    embedding=embedding
                )
                active_persons.append(person)
        
        frame_data.persons = active_persons
        frame_data.current_people_count = len(active_persons)
        frame_data.total_unique_people = len(self.confirmed_ids)
        self._last_persons_snapshot = active_persons
        
        return frame_data

if __name__ == "__main__":
    print("Testing TrackerStage in Isolation...")
    import time
    
    # Initialize mock tracker
    tracker = PersonTracker(use_mock=True)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    data = FrameData(frame_index=0, timestamp=time.time(), raw_frame=dummy_frame, processed_frame=dummy_frame.copy())
    
    # Run test
    out = tracker.process(data)
    print(f"Mock test completed. People count: {out.current_people_count}, Unique total: {out.total_unique_people}")
