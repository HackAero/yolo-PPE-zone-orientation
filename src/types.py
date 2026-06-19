from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np

@dataclass
class TrackedPerson:
    """
    Data structure representing a single tracked person in a frame.
    Each field is populated or modified by different pipeline components.
    """
    person_id: int
    bbox: List[int]  # Bounding box in pixels: [xmin, ymin, xmax, ymax]
    confidence: float
    
    # --- PPE Compliance (Person 4) ---
    has_helmet: Optional[bool] = None
    has_glasses: Optional[bool] = None
    compliance_violations: List[str] = field(default_factory=list)
    
    # --- Behavior & Pose (Person 5) ---
    is_fallen: Optional[bool] = None
    keypoints: Optional[np.ndarray] = None  # Pose keypoints (e.g., COCO keypoints array)
    
    # --- Re-Identification & Tracking (Person 2) ---
    embedding: Optional[np.ndarray] = None  # Crop visual embedding vector
    reid_matched: bool = False
    
    # Custom extensible metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameData:
    """
    Data structure carrying the state of the current video frame as it
    flows down the pipeline conveyor belt.
    """
    frame_index: int
    timestamp: float
    raw_frame: np.ndarray          # The original unchanged input frame (read-only)
    processed_frame: np.ndarray    # The frame being modified (e.g., blurred, annotated)
    
    # List of persons detected/tracked in this frame
    persons: List[TrackedPerson] = field(default_factory=list)
    
    # --- Environmental Conditions (Person 5) ---
    blur_score: float = 0.0            # Variance of Laplacian score
    is_image_blurry: bool = False      # Bad quality (smudge, out of focus)
    is_smoke_detected: bool = False    # Fire or smoke detected
    
    # --- Integration Counters & Flags (Person 1) ---
    current_people_count: int = 0
    total_unique_people: int = 0
    
    # Lists of active system alerts triggered during this frame
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    
    # Extensible field for passing raw model outputs or arbitrary data
    extra_metadata: Dict[str, Any] = field(default_factory=dict)
