# MTU Smart Safety Monitoring

Real-time computer vision for an industrial floor: count people, check PPE, spot falls and fire, blur faces, and ping a rover when something goes wrong.

Built for a hackathon around **MTU Aero Engines** — factory and MRO environments where safety rules depend on *where* you are, not just *whether* you're wearing gear.

---

## The problem we picked

Fixed cameras (or a rover cam) see a live floor. Supervisors need to know:

- How many people are in the area (without double-counting when someone walks out and back in)
- Whether PPE matches the zone — helmet in the work floor, nothing required in the break area, full kit near restricted machinery
- Falls, smoke/fire, and camera blind spots
- That workers on screen are not identifiable in the UI

We are five people, no SLAM stack, one afternoon for hardware. So we aimed for something that **works live**, tells a clear story, and could grow into a real MTU deployment later.

---

## What you see in the demo

| Piece | What it does |
|-------|----------------|
| **OpenCV window** (`main.py`) | Live feed with zone bands, worker labels, PPE status, hazards |
| **Streamlit** (`dashboard.py`) | Same pipeline in the browser + supervisor replay clips |
| **Robot dashboard** (`robot_dashboard.html`) | MQTT alert board + fake rover ETA (no install, just open the file) |
| **Robot sim** (`robot_sim.py`) | Pretends to drive to the incident when an alert fires |

Press **`w`** in the video window to cycle zone layouts (3-band → all safe → all work → all restricted). Useful when judges ask “what if the whole room is a work zone?”

**Backup:** record a short screen capture of mock + dashboard + robot sim. Live demo is better; video saves you if Wi‑Fi or the webcam dies.

---

## Design decisions (why it looks like this)

### Zones without a floor plan (“B-lite”)

We wanted *different rules in different areas* but had no robot pose, no LiDAR, no calibration time on site.

**Choice:** draw zones as vertical bands on the camera image (left = safe, center = work, right = restricted). A person’s bbox center picks the zone. Rules live in `zones/monitor.json` — easy to edit without recompiling.

**Trade-off:** zones move with the camera, not with the building. Good enough for a side-mounted demo cam; a real install would add homography or SLAM later. The same JSON and `zone_id` fields are ready for that.

### One pipeline, several models, one CPU

The engine runs up to four YOLO passes (people, PPE, pose, fire/smoke) plus MediaPipe for faces. On a laptop CPU that is a lot.

**Choice:** `FAST_MODE` in `config.py` — smaller input size, skip frames on the heavy stages, one shared PPE model pass per N frames, async frame grab so inference does not stall the camera.

**Trade-off:** slightly choppier detections, much better FPS. `SMOOTH_MODE` exists if you have GPU headroom.

### PPE: model first, heuristics second

Custom `ppe_model.pt` for helmets and goggles. Yellow-vest check is HSV on the torso. Glasses fallback uses face landmarks + edge density.

**Gotcha we hit:** a color-based “helmet” heuristic on bare hair kept saying helmet OK. We now **trust the YOLO model when it says no helmet** and only run the color check when the model did not run at all.

### Privacy by default on output

Detection uses the raw frame in memory (normal for CV). **Display** goes through the privacy stage — faces blurred before `imshow` or Streamlit. Logs use `Worker-A3F91C` style labels (`session_labels.py`), not employee IDs.

Critical events can save a short replay to `data/replays/` (supervisor clip, not a full shift recording).

### Robot handoff over MQTT

No custom server for the hackathon. Pipeline publishes JSON to HiveMQ public broker; `robot_dashboard.html` and `robot_sim.py` subscribe in the browser / Python.

Alert types: `FALL_DETECTED`, `RESTRICTED_ENTRY`, `NO_HELMET`, `NO_GLASSES`, `NO_VEST`, `FIRE_DETECTED`, `SMOKE_DETECTED` — each with real `zone_id` from the zone stage, not a hardcoded bay name.

**Trade-off:** public broker is fine for a demo; production would use a private broker and auth.

---

## Architecture

```
Camera / mock
    │
    ▼
┌─────────────┐   ByteTrack + Re-ID histogram (unique headcount)
│  Tracker    │
└──────┬──────┘
       ▼
┌─────────────┐   Shared PPE YOLO (helmet / goggles)
│  PPE +      │   Compliance heuristics (vest, glasses fallback)
│  Compliance │
└──────┬──────┘
       ▼
┌─────────────┐   Zone rules + MQTT dispatch triggers
│  Zone map   │   press w to cycle layout
└──────┬──────┘
       ▼
┌─────────────┐   Fall, smoke/fire, blur/fog
│ Environment │
└──────┬──────┘
       ▼
┌─────────────┐   Face blur on output frame
│  Privacy    │
└──────┬──────┘
       ▼
  main.py / dashboard UI          MQTT ──► robot_dashboard.html
                                  └──► robot_sim.py
```

| Stage | File |
|-------|------|
| Orchestration | `src/engine.py` |
| People + Re-ID | `src/tracker.py` |
| PPE model | `src/ppe_inference.py` |
| Heuristics + alerts | `src/compliance.py` |
| Zones | `src/zone_map.py` |
| Falls / environment | `src/environment.py` |
| Blur | `src/privacy.py` |
| MQTT out | `src/dispatcher.py` |

---

## Quick start

**Prerequisites:** Python 3.8+, webcam optional

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Live camera

```bash
python main.py --source 0
```

Zones are automatic: left 20% safe, center work (helmet + glasses + vest), right restricted (entry alert). Person position = bbox center.

### No camera (mock warehouse)

```bash
python main.py --mock
```

### Full robot demo (three terminals + browser)

```bash
# 1 — fake rover
python robot_sim.py

# 2 — vision
python main.py --mock    # or --source 0

# 3 — open robot_dashboard.html in Chrome/Firefox
```

MQTT topics: `hackatum/robot/dispatch` (alerts), `hackatum/robot/status` (rover state).

### Streamlit

```bash
streamlit run dashboard.py
```

### Custom zones

```bash
python main.py --zones-file zones/monitor.json
python main.py --camera-profile rover   # same bands, rover cam profile name
```

---

## Config knobs worth knowing

All in `src/config.py`:

- `FAST_MODE` / `SMOOTH_MODE` — CPU vs quality
- `DISPATCH_BACKEND` — `mqtt` (default), `console`, or `http`
- PPE and fire thresholds — tuned during the hackathon on real webcam footage
- `ALERT_DEBOUNCE_SECONDS` — stops terminal/MQTT spam for the same violation

---

## Privacy & security (hackathon scope)

- Processing stays **on device**; default setup does not stream video to a cloud API.
- Faces are blurred on **output** frames only.
- Worker labels are **session-only** pseudonyms.
- `data/replays/`, `.env`, and camera dumps are gitignored.

A production MTU rollout would still need access control, retention policy, and a formal privacy review. This repo is a demonstrator, not a certified safety system.

---

## Honest limits

- Not certified for real safety decisions — demo / research quality.
- Vest detection is color-heuristic; works on high-vis yellow, not every uniform.
- Public MQTT broker — do not put sensitive data in payloads.
- Zone geometry is image-space; remount the camera and the bands move with it.

---

## If you only have two minutes (pitch outline)

1. **Problem** — industrial floors have zones with different PPE rules; one camera should enforce that and call for help.
2. **Trick** — conveyor pipeline + zone JSON + Re-ID counting + MQTT to a rover, all on CPU.
3. **Live** — walk into restricted → dashboard screams → robot sim drives; remove helmet in work zone → PPE alert; press `w` to show layout modes.
4. **Next** — homography to a floor plan, private MQTT, edge deploy on the rover itself.

---

*Hackatum · MTU Smart Safety Monitoring*
