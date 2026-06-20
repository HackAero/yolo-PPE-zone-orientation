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

## System Architecture

The pipeline processes every frame through four sequential stages:

1. **Tracker Stage** (`tracker.py`): Detects people and assigns persistent IDs.
2. **Compliance Stage** (`compliance.py`): Checks helmets and glasses.
3. **Environment Stage** (`environment.py`): Checks image quality, smoke/fire, and detects falls.
4. **Privacy Stage** (`privacy.py`): Anonymizes faces/tattoos on the processed frame.

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