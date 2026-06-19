import os

# --- Model Configurations ---
# Paths to model weights (put your customized/pre-trained models in the models/ folder)
YOLO_MODEL_PATH = "yolov8n.pt"  # Person detection, tracking & poses
PPE_MODEL_PATH = "yolov8n.pt"   # Custom safety gear model (e.g. helmets/glasses)

# --- Confidence & Detection Thresholds ---
PERSON_CONF_THRESHOLD = 0.4
PPE_CONF_THRESHOLD = 0.5

# --- Privacy Settings (Person 3) ---
PRIVACY_BLUR_KERNEL_SIZE = (57, 57)  # Higher values = stronger blur
PRIVACY_FACE_EXPAND_TOP = 0.35      # Include bangs and the top of the head
PRIVACY_FACE_EXPAND_SIDE = 0.12     # Include hair beside both sides of the face
PRIVACY_FACE_EXPAND_BOTTOM = 0.10   # Include the lower edge of the face
PRIVACY_FACE_CACHE_FRAMES = 8       # Keep the last face location through short detection gaps
PRIVACY_FACE_SMOOTHING_ALPHA = 0.7  # Weight of the latest face detection in EMA smoothing
BLUR_TATOOS = False                 # Toggle if tattoo/skin segmentation is active

# --- Quality & Safety Settings (Person 5) ---
BLUR_LAPLACIAN_THRESHOLD = 20.0     # Under this value, image is flagged as blurry/smudged
SMOKE_CONF_THRESHOLD = 0.5          # Threshold for environmental smoke detection
FALL_ANGLE_THRESHOLD = 60           # Angle (degrees) of spine relative to vertical (e.g. > 60 = horizontal/lying)

# --- Re-Identification settings (Person 2) ---
REID_COSINE_SIMILARITY_THRESHOLD = 0.7  # Above this, it's considered the same person

# --- Directory Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
