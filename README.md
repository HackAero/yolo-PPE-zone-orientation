# MTU Smart Safety Monitoring System

A real-time, multi-stage Computer Vision pipeline built to enhance warehouse and industrial safety. The system detects workers, checks PPE compliance (helmets/glasses), monitors for environmental hazards (smoke/fire/fog), detects falls, and automatically anonymizes faces for privacy—all running in a modular conveyor-belt pipeline.

## Features

- **Person Detection & Re-ID**: Tracks unique workers across frames using YOLO + ByteTrack, with CNN-based Re-Identification (Re-ID).
- **PPE Compliance**: Detects safety helmets and glasses using a custom YOLO model, with a color-heuristic fallback.
- **Fall Detection**: Analyzes human pose (shoulders/hips) using YOLO-Pose to detect workers lying on the floor.
- **Environmental Safety**: Monitors camera quality (blur/smudges) and detects smoke/fire using a dedicated YOLO model.
- **Privacy Anonymization**: Automatically blurs faces and optionally blurs exposed tattoos.
- **Mock Simulation Engine**: Generates synthetic warehouse scenes and workers—perfect for testing without a camera or ML models.
- **Interactive Dashboard**: Real-time monitoring UI built with **Streamlit**.
- **Live Video Support**: Works with USB webcams, IP cameras, or local video files.
- **Zone-aware compliance (B-lite)**: Image-space zones (`zones.json`) with per-zone PPE rules, restricted-area entry alerts, and real `zone_id` on MQTT dispatch.

## System Architecture

The pipeline processes every frame through four sequential stages:

1. **Tracker Stage** (`tracker.py`): Detects people and assigns persistent IDs.
2. **Compliance Stage** (`compliance.py`): Checks helmets and glasses.
3. **Zone Stage** (`zone_map.py`): Assigns workers to camera-space zones and applies zone-specific rules.
4. **Environment Stage** (`environment.py`): Checks image quality, smoke/fire, and detects falls.
5. **Privacy Stage** (`privacy.py`): Anonymizes faces/tattoos on the processed frame.

The pipeline runs up to **4 independent YOLO models** per frame:
- `yolov8s.pt` → People
- `yolov8n-pose.pt` → Human pose (fall detection)
- `models/ppe_model.pt` (or heuristic fallback) → Helmets & Glasses
- `fire_smoke.pt` → Smoke & Fire



## Getting Started

### 1. Prerequisites
- Python 3.8+
- pip


### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run

Live webcam (default camera):
```bash
python main.py
```

Simulation (no camera or models needed):
```bash
python main.py --mock
```

Zones use the **full camera frame** automatically (no calibration step). Side-mounted camera: left = safe, center = work, right = restricted. Zone is chosen from the person's bbox center.

```bash
python main.py --source 0
```

Custom zone file:
```bash
python main.py --zones-file zones/my_calibrated.json
```

Streamlit dashboard:
```bash
streamlit run dashboard.py
```

## Privacy & security

This system watches a real work area, so we kept a few things in mind:

**Processing stays on the device.** Video runs through the pipeline locally. We are not sending a live feed to a cloud API as part of the default setup.

**Faces are blurred before display.** The privacy stage runs on the output frame, so what you see in the OpenCV window or dashboard is anonymized. Detection still uses the raw frame in memory earlier in the pipeline — that is normal for this kind of system, but nothing is recorded unless you explicitly save it yourself (e.g. with `save_camera.py`).

**Worker labels are session-only.** The UI shows opaque names like `Worker-A3F91C`, not employee IDs. They are only for counting and alerts during a run; we do not link them to HR records.

**We try not to commit sensitive files.** `.gitignore` excludes camera snapshots, video dumps, and `.env` files so test footage does not end up in the repo by accident.

For a production deployment at a site like MTU you would still need proper access control on the dashboard, retention policies, and a formal privacy review — but for the hackathon demo, the goal is: process locally, blur on screen, minimize what gets stored.