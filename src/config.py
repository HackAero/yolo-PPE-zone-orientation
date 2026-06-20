import os

# --- Performance / Inference ---
FAST_MODE = False
# Balanced live-feed tuning: smoother CPU playback with modest accuracy tradeoffs.
SMOOTH_MODE = True

YOLO_DEVICE = "cpu"

if FAST_MODE:
    YOLO_IMGSZ = 320
    YOLO_MODEL_PATH = "yolov8n.pt"
    CAMERA_MAX_WIDTH = 480
    PPE_INFERENCE_INTERVAL = 3
    PERSON_DETECT_INTERVAL = 2
    FACE_DETECT_INTERVAL = 4
    POSE_INFERENCE_INTERVAL = 4
    SMOKE_INFERENCE_INTERVAL = 6
    COMPLIANCE_HEURISTIC_INTERVAL = 3
    ASYNC_FRAME_GRAB = True
elif SMOOTH_MODE:
    YOLO_IMGSZ = 384
    YOLO_MODEL_PATH = "yolov8s.pt"
    CAMERA_MAX_WIDTH = 640
    PPE_INFERENCE_INTERVAL = 2
    PERSON_DETECT_INTERVAL = 2
    FACE_DETECT_INTERVAL = 3
    POSE_INFERENCE_INTERVAL = 3
    SMOKE_INFERENCE_INTERVAL = 5
    COMPLIANCE_HEURISTIC_INTERVAL = 2
    ASYNC_FRAME_GRAB = True
else:
    YOLO_IMGSZ = 416
    YOLO_MODEL_PATH = "yolov8s.pt"
    CAMERA_MAX_WIDTH = 0
    PPE_INFERENCE_INTERVAL = 1
    PERSON_DETECT_INTERVAL = 1
    FACE_DETECT_INTERVAL = 1
    POSE_INFERENCE_INTERVAL = 2
    SMOKE_INFERENCE_INTERVAL = 3
    COMPLIANCE_HEURISTIC_INTERVAL = 1
    ASYNC_FRAME_GRAB = False

REID_INFERENCE_INTERVAL = 2
REID_STABLE_TRACK_FRAMES = 5

# --- Model Configurations ---
PPE_MODEL_PATH = "models/ppe_model.pt"
POSE_MODEL_PATH = "yolov8n-pose.pt"

# --- Camera Pre-processing ---
CAMERA_CONTRAST = 1.2   # 1.0 is default, >1.0 increases contrast
CAMERA_BRIGHTNESS = 20  # 0 is default, >0 increases brightness

# --- Confidence & Detection Thresholds ---
PERSON_CONF_THRESHOLD = 0.25
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
SMOKE_CONF_THRESHOLD = 0.68          # Threshold for environmental smoke detection
FIRE_CONF_THRESHOLD = 0.6           # Threshold for environmental fire detection
SMOKE_CONFIRMATION_FRAMES = 4       # Threshold for consecutive smoke alerts to be sure  
FIRE_CONFIRMATION_FRAMES = 3        # Threshold for consecutive fire alerts to be sure  

FALL_ANGLE_THRESHOLD = 50           # Angle (degrees) of spine relative to vertical (e.g. > 60 = horizontal/lying)
FALL_CONFIRMATION_FRAMES = 2        # Number of consecutive frames required to confirm a fall
KEYPOINT_CONFIDENCE_THRESHOLD = 0.35 # Minimum confidence for pose keypoints to be used
FALL_ASPECT_RATIO_THRESHOLD = 1.75   # Width/height ratio threshold for aspect ratio fallback detection

# --- Re-Identification settings (Person 2) ---
REID_COSINE_SIMILARITY_THRESHOLD = 0.78  # Balanced threshold: strict enough to separate similar uniforms, loose enough to match re-entries

# --- Directory Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# --- Robot Dispatch ---
# Identity
TEAM_ID = "MTU"    # Your team name, shown in the robot dashboard
ZONE_ID = "A3"     # Your assigned zone, robot navigates here on dispatch

# Backend selector: "mqtt" | "http" | "console"
# For the hackathon demo: use "mqtt" — open robot_dashboard.html on any laptop
# to see signals arrive in real-time. No server needed.
DISPATCH_BACKEND = "mqtt"

# Cooldown: seconds before the same person can trigger another dispatch.
# Prevents the robot from being spammed for a single fall event.
DISPATCH_COOLDOWN_SECONDS = 15

# MQTT backend (used when DISPATCH_BACKEND = "mqtt")
# Default: HiveMQ free public broker — works instantly with no account.
DISPATCH_MQTT_BROKER = "broker.hivemq.com"
DISPATCH_MQTT_PORT = 8884          # 8884 = WebSocket + TLS (wss://)
DISPATCH_MQTT_TOPIC = "hackatum/robot/dispatch"
DISPATCH_MQTT_USE_WS = True        # Use WebSocket transport (required for browser dashboard)
DISPATCH_MQTT_USE_TLS = True       # TLS on port 8884

# HTTP backend (used when DISPATCH_BACKEND = "http")
DISPATCH_HTTP_URL = "http://robot-host:5000/dispatch"
