import os
from typing import Optional
import platform

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- Performance / Inference ---
# FAST_MODE: tuned for laptop CPU + webcam (skip frames, smaller YOLO input).
# Flip SMOOTH_MODE on if you have a GPU and want heavier models every frame.
FAST_MODE = True
SMOOTH_MODE = False

YOLO_DEVICE = "cpu"

if FAST_MODE:
    YOLO_IMGSZ = 320
    YOLO_MODEL_PATH = "yolov8n.pt"
    CAMERA_MAX_WIDTH = 480
    # --- FAST_MODE Overrides ---
    PPE_INFERENCE_INTERVAL = 6
    PPE_CROP_PASS = True
    PPE_CROP_MAX_PERSONS = 4
    PERSON_DETECT_INTERVAL = 3
    FACE_DETECT_INTERVAL = 2
    POSE_INFERENCE_INTERVAL = 4
    SMOKE_INFERENCE_INTERVAL = 4
    COMPLIANCE_HEURISTIC_INTERVAL = 2
    ASYNC_FRAME_GRAB = True
elif SMOOTH_MODE:
    YOLO_IMGSZ = 384
    YOLO_MODEL_PATH = "yolov8s.pt"
    CAMERA_MAX_WIDTH = 640
    PPE_IMGSZ = 384
    PPE_INFERENCE_INTERVAL = 5
    PPE_CROP_PASS = True
    PPE_CROP_MAX_PERSONS = 2
    PERSON_DETECT_INTERVAL = 2
    FACE_DETECT_INTERVAL = 3
    POSE_INFERENCE_INTERVAL = 5
    SMOKE_INFERENCE_INTERVAL = 10
    COMPLIANCE_HEURISTIC_INTERVAL = 2
    ASYNC_FRAME_GRAB = True
else:
    YOLO_IMGSZ = 416
    YOLO_MODEL_PATH = "yolov8s.pt"
    CAMERA_MAX_WIDTH = 0
    PPE_IMGSZ = 416
    PPE_INFERENCE_INTERVAL = 1
    PPE_CROP_PASS = True
    PPE_CROP_MAX_PERSONS = 3
    PERSON_DETECT_INTERVAL = 1
    FACE_DETECT_INTERVAL = 1
    POSE_INFERENCE_INTERVAL = 2
    SMOKE_INFERENCE_INTERVAL = 3
    COMPLIANCE_HEURISTIC_INTERVAL = 1
    ASYNC_FRAME_GRAB = False

REID_INFERENCE_INTERVAL = 4
REID_STABLE_TRACK_FRAMES = 5
REID_USE_CNN = not FAST_MODE

# --- Model Configurations ---
PPE_MODEL_PATH = "models/ppe_model.pt"
PPE_FORCE_PRETRAINED = True
PPE_USE_PRETRAINED_WORLD = True
PPE_PRETRAINED_MODEL = "yolov8s-worldv2.pt"
POSE_MODEL_PATH = "yolov8n-pose.pt"

# --- Camera Pre-processing ---
CAMERA_CONTRAST = 1.0   # 1.0 is default, >1.0 increases contrast
CAMERA_BRIGHTNESS = 0  # 0 is default, >0 increases brightness

# --- Confidence & Detection Thresholds ---
PERSON_CONF_THRESHOLD = 0.45
PPE_CONF_THRESHOLD = 0.2

# --- PPE Precision Tuning (pretrained-friendly) ---
# Higher thresholds reduce false positive PPE detections.
PPE_HELMET_AREA_MAX = 0.45

PPE_GLASSES_STRICT_CONF = 0.25
PPE_GLASSES_HELMET_ASSIST = True
PPE_GLASSES_HELMET_CONF_BONUS = 0.08
PPE_HELMET_REL_Y_MAX = 0.55
PPE_GLASSES_REL_Y_MAX = 0.52
PPE_GLASSES_REL_X_MIN = 0.08
PPE_GLASSES_REL_X_MAX = 0.92
PPE_GLASSES_AREA_MIN = 0.0015
PPE_GLASSES_AREA_MAX = 0.14
# Require repeated evidence before marking PPE as present.
PPE_CONFIRM_FRAMES = 1
# ==============================================================================
# "WOW FACTOR" HACKATHON NOTIFICATIONS
# ==============================================================================
ENABLE_TTS_SIREN = True
ENABLE_TWILIO_SMS = os.getenv("ENABLE_TWILIO_SMS", "False").lower() == "true"
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "")
TWILIO_TO_NUMBER = os.getenv("TWILIO_TO_NUMBER", "")

# ==============================================================================
# PIPELINE STAGES & SCHEDULING
# ==============================================================================
# Require repeated misses before clearing PPE presence.
PPE_CLEAR_FRAMES = 4
PPE_USE_GLASSES_HEURISTIC = False

# --- Privacy Settings (Person 3) ---
PRIVACY_CENSORSHIP_MODE = "blur"    # "blur" or "garfield" (toggle in UI / press g in main.py)
GARFIELD_SCALE = 1.45             # Sticker size vs face box (blur still runs underneath)
PRIVACY_BLUR_KERNEL_SIZE = (57, 57)  # Higher values = stronger blur
PRIVACY_FACE_EXPAND_TOP = 0.35      # Include bangs and the top of the head
PRIVACY_FACE_EXPAND_SIDE = 0.12     # Include hair beside both sides of the face
PRIVACY_FACE_EXPAND_BOTTOM = 0.10   # Include the lower edge of the face
PRIVACY_FACE_CACHE_FRAMES = 8       # Keep the last face location through short detection gaps
PRIVACY_FACE_SMOOTHING_ALPHA = 0.7  # Weight of the latest face detection in EMA smoothing
BLUR_TATOOS = True                  # Prototype: blur pose-derived arm regions
TATTOO_MODEL_PATH = "models/tattoo_model.pt"
TATTOO_CONF_THRESHOLD = 0.10        # Hypersensitive for hackathon blackout marker demos
TATTOO_MASK_THRESHOLD = 0.35        # Segmentation mask binarization threshold
TATTOO_MIN_MASK_PIXELS = 64         # Ignore tiny masks that are usually detector noise
TATTOO_INPUT_SIZE = 640             # YOLO tattoo inference resolution
TATTOO_FAIL_CLOSED = True           # Blur full ROI if tattoo inference/mask fails
TATTOO_FALLBACK_BBOX_ROIS = True    # Use bbox-based limb ROIs when pose keypoints are missing
ARM_ROI_PADDING_RATIO = 0.15        # Tighter padding for cleaner arm crops
ARM_ROI_MIN_PADDING = 12            # Minimum arm ROI padding in pixels
LEG_ROI_PADDING_RATIO = 0.15        # Padding relative to each leg segment length
LEG_ROI_MIN_PADDING = 12            # Minimum leg ROI padding in pixels

# --- Quality & Safety Settings (Person 5) ---
BLUR_LAPLACIAN_THRESHOLD = 20.0     # Under this value, image is flagged as blurry/smudged
SMOKE_CONF_THRESHOLD = 0.45          # Threshold for environmental smoke detection
FIRE_CONF_THRESHOLD = 0.45           # Threshold for environmental fire detection
SMOKE_CONFIRMATION_FRAMES = 1       # Threshold for consecutive smoke alerts to be sure  
FIRE_CONFIRMATION_FRAMES = 1        # Threshold for consecutive fire alerts to be sure  

FALL_ANGLE_THRESHOLD = 50           # Angle (degrees) of spine relative to vertical (e.g. > 60 = horizontal/lying)
FALL_CONFIRMATION_FRAMES = 2        # Number of consecutive frames required to confirm a fall
KEYPOINT_CONFIDENCE_THRESHOLD = 0.35 # Minimum confidence for pose keypoints to be used
FALL_ASPECT_RATIO_THRESHOLD = 1.75   # Width/height ratio threshold for aspect ratio fallback detection

# --- Re-Identification settings (Person 2) ---
REID_COSINE_SIMILARITY_THRESHOLD = 0.72  # Balanced threshold: strict enough to separate similar uniforms, loose enough to match re-entries

# --- Directory Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
GARFIELD_IMAGE_PATH = os.path.join(IMAGES_DIR, "garfield.png")

# --- Zone map (B-lite) ---
ZONES_ENABLED = True
ZONES_DIR = os.path.join(BASE_DIR, "zones")
# Profile: "monitor" (fixed cam / mock) or "rover" (on-robot camera). Override via CLI or ZONES_PROFILE env.
ZONES_PROFILE = os.environ.get("ZONES_PROFILE", "monitor")
ZONES_CONFIG_PATH = os.path.join(ZONES_DIR, f"{ZONES_PROFILE}.json")
# Legacy fallback if profile file missing
ZONES_LEGACY_PATH = os.path.join(BASE_DIR, "zones.json")


def resolve_zones_path(profile: Optional[str] = None, explicit_path: Optional[str] = None) -> str:
    """Pick zone JSON: explicit --zones-file wins, then profile, then legacy zones.json."""
    if explicit_path:
        return explicit_path
    name = profile or ZONES_PROFILE
    candidate = os.path.join(ZONES_DIR, f"{name}.json")
    if os.path.exists(candidate):
        return candidate
    if os.path.exists(ZONES_LEGACY_PATH):
        return ZONES_LEGACY_PATH
    return candidate

# --- Alert Debouncing ---
# Minimum seconds between repeated console prints / robot dispatches for the
# same person + violation type. Eliminates the per-frame spam you see in the
# terminal. Set to 0 to disable debouncing (show every alert).
ALERT_DEBOUNCE_SECONDS = 5

# --- False Alarm Filter ---
# Alerts must persist for N frames before they become confirmed incidents.
ALERT_VERIFY_CRITICAL_FRAMES = 1
ALERT_VERIFY_WARNING_FRAMES = 2
# Reset verifier state if an alert signature disappears longer than this.
ALERT_VERIFY_STALE_SECONDS = 1.5

# --- Reliability Degradation ---
# If vision quality/noise is poor for sustained frames, switch to LIMITED mode.
RELIABILITY_BLUR_FRAMES = 3
RELIABILITY_VERIFYING_THRESHOLD = 4
# In LIMITED mode, suppress warning-level robot dispatch to reduce noisy actions.
RELIABILITY_SUPPRESS_WARNING_DISPATCH = True

# --- Tracker Maintenance ---
# How often (in detection frames) to prune stale ByteTrack IDs from internal
# dicts. ByteTrack IDs are monotonically increasing so old ones accumulate.
TRACKER_PRUNE_INTERVAL = 500

# --- Runtime Responsiveness ---
# Store every Nth frame in replay pre-buffer to reduce copy overhead.
REPLAY_BUFFER_STRIDE = 10
# Limit repeated alert prints to avoid terminal I/O stalls.
ALERT_LOG_MIN_INTERVAL_SECONDS = 1.0

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
DISPATCH_MQTT_STATUS_TOPIC = "hackatum/robot/status"
DISPATCH_MQTT_USE_WS = True        # Use WebSocket transport (required for browser dashboard)
DISPATCH_MQTT_USE_TLS = True       # TLS on port 8884

# HTTP backend (used when DISPATCH_BACKEND = "http")
DISPATCH_HTTP_URL = "http://robot-host:5000/dispatch"
