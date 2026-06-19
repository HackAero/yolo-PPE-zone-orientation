import os

# --- Model Configurations ---
# Paths to model weights (put your customized/pre-trained models in the models/ folder)
YOLO_MODEL_PATH = "yolov8s.pt"  # Upgraded to small model for better detection
PPE_MODEL_PATH = "models/ppe_model.pt"   # Custom safety gear model (e.g. helmets/glasses)

# --- Camera Pre-processing ---
CAMERA_CONTRAST = 1.2   # 1.0 is default, >1.0 increases contrast
CAMERA_BRIGHTNESS = 20  # 0 is default, >0 increases brightness

# --- Confidence & Detection Thresholds ---
PERSON_CONF_THRESHOLD = 0.25
PPE_CONF_THRESHOLD = 0.5

# --- Privacy Settings (Person 3) ---
PRIVACY_BLUR_KERNEL_SIZE = (51, 51)  # Higher values = stronger blur
PRIVACY_FACE_PADDING = 10           # Bounding box padding in pixels for faces
BLUR_TATOOS = False                 # Toggle if tattoo/skin segmentation is active

# --- Quality & Safety Settings (Person 5) ---
BLUR_LAPLACIAN_THRESHOLD = 20.0     # Under this value, image is flagged as blurry/smudged
SMOKE_CONF_THRESHOLD = 0.5          # Threshold for environmental smoke detection
FALL_ANGLE_THRESHOLD = 50           # Angle (degrees) of spine relative to vertical (e.g. > 60 = horizontal/lying)
FALL_CONFIRMATION_FRAMES = 2        # Number of consecutive frames required to confirm a fall
KEYPOINT_CONFIDENCE_THRESHOLD = 0.35 # Minimum confidence for pose keypoints to be used
FALL_ASPECT_RATIO_THRESHOLD = 1.75   # Width/height ratio threshold for aspect ratio fallback detection

# --- Re-Identification settings (Person 2) ---
REID_COSINE_SIMILARITY_THRESHOLD = 0.88  # Extremely strict match to tell identical uniforms apart

# --- Directory Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
