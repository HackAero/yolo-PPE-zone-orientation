# MTU Smart Safety Monitoring

Real-time computer vision for an industrial floor. It counts personnel, checks PPE compliance, spots falls and fire, blurs faces and sensitive information, and pings an autonomous rover when something goes critically wrong. 

Built for **MTU Aero Engines** at Hackatum.

---

<img width="1918" height="1078" alt="image" src="https://github.com/user-attachments/assets/06d42a9c-23e6-4ba5-bcf2-e632cdfaaa20" />


## The problem

Traditional industrial safety monitoring usually relies on static, infrastructure-heavy setups or blanket rules that apply everywhere uniformly. MTU Aero Engines requires a more nuanced approach: safety rules depend heavily on *where* a worker is standing, not just *what* they are wearing.

In a live factory deployment, we will face several challenges:

- Accurately counting unique personnel on the floor without double-counting individuals as they move in and out of camera blind spots.
- Enforcing contextual PPE rules, so equipment matches the zone: helmet and glasses on the work floor, relaxed rules in safe areas, nobody in restricted bays
- Identifying critical events such as Falls, smoke and fire outbreaks.
- Ensuring compliance with strict European privacy regulations (GDPR/DSGVO) by preventing workers' identities from being exposed.

Additionally, we faced the hackathon constraint. We were a team of five with only a weekend to handle everything. We lacked the time and access required to map a full SLAM stack or execute complex on-site floor-plan calibrations. We needed a solution that was modular, well-performing and adaptable to any physical space.

---

## How we ended up here

Our path looked somewhat like this:

1. We started by building a **core pipeline**. Tracking people (YOLO + ByteTrack), checking PPE, watching for falls and fire, bluring faces on output, dispatching alerts over MQTT to a dummy rover and creating a browser dashboard. That had to run on a laptop CPU, so we built `SMOOTH_MODE` and `FAST_MODE` (smaller models, skip heavy frames, async camera grab).
2. The **spatial challenge**. MTU cares about *location* and we do too. Our first version used static vertical bands in `zones/*.json` (with the left side being safe, centre being the work zone and the right being restricted). It was simple enough, but it was glued to the camera, so if you remount the cam, the entire “factory” moves.
3. The **pragmatic leap**. We wanted to *show* different zone layouts live (all yellow, red/green splits, etc.) without a floor map. We introduced OpenCV ArUco marker detection via `zone_map.py`. We flash a fiducial marker and its ID picks which wall the rover or camera is facing; estimated distance picks “close” vs “far” layout. The layout locks when the marker leaves frame so the demo does not flicker much. While this approach is not production-ready, it provided a practical and effective way to communicate zone behavior on the exhibition floor.

<img width="1918" height="1078" alt="image" src="https://github.com/user-attachments/assets/cc02fb44-7fd5-46b6-87e4-2fe387d51bb0" />

4. **Testing it live** exposed critical edge cases. Smoke detection fired too often, so we slowed inference intervals and added multi-frame alert verification (`alert_filter.py`) plus a **LIMITED** HUD mode when the image is blurry or noisy (critical alerts still go out, but PPE spam to the robot is throttled). PPE heuristics falsely called bare hair “helmet”, so we now trust the YOLO model instead.

5. **Robot handoff without a server**. To demonstrate a complete industrial automation loop without managing a heavy backend server, we use public MQTT (HiveMQ). We send real-time alerts to a light `robot_dashboard.html` that can be opened in any browser and a `robot_sim.py` routine, that instantly dispatches commands to a future rover.

<img width="1918" height="962" alt="image" src="https://github.com/user-attachments/assets/02ef62f1-c875-4d65-8db0-caa8f06112bf" />
<img width="1825" height="968" alt="image" src="https://github.com/user-attachments/assets/8d6b58e1-a4a0-457a-a7f8-fba4ad7f9e2b" />


---

## Evaluation map

### Innovation & creativity (40%)


| What you care about     | What we did                                                                                                                                                  |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Originality**         | We implemented zone-dependent safety on a single camera feed tied to a rover dispatch loop.                                                                  |
| **Unique approach**     | We used ArUco markers to swap coloured zone layouts (green / yellow / red) without SLAM or a CAD floor plan.                                                 |
| **Unconventional tech** | We used public MQTT mapped to a static HTML dashboard, paired with a custom color-histogram Re-ID algorithm to maintain accurate headcounts entirely on CPU. |
| **WOW factor**          | Live colour bands react to a phone marker; restricted entry pings the robot dashboard; press `g` for a little Easter egg                                     |


**Zone colours on the feed**


| Colour     | Meaning                                                     |
| ---------- | ----------------------------------------------------------- |
| **Green**  | Safe areas, no PPE required                                 |
| **Yellow** | Work floor: helmet + glasses; vest checked but not required |
| **Red**    | Restricted entry alert!                                     |


Markers **0-3** = four walls of a demo cross layout. Distance < 2 m vs ≥ 2 m switches between two band layouts per wall. Person zone = bbox **foot point** inside the bands.

### Technical implementation (40%)


| Stage                   | File                   | Description                                                                                                                                    |
| ----------------------- | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Orchestration           | `src/engine.py`        | Coordinates data flow and state management across all pipeline stages.                                                                         |
| People + Re-ID          | `src/tracker.py`       | Combines YOLO object detection, ByteTrack tracking, and custom color-histogram matching to prevent duplicate counts on CPU.                    |
| PPE                     | `src/ppe_inference.py` | Runs specialized secondary models to identify safety gear.                                                                                     |
| Heuristics              | `src/compliance.py`    | Evaluates detected gear against the active spatial rules of the person's location.                                                             |
| Zones + ArUco           | `src/zone_map.py`      | Computes pose distance relative to markers to project and lock dynamic compliance boundaries.                                                  |
| Falls / smoke / blur    | `src/environment.py`   | Runs a YOLOv8-pose skeleton model to compute spine-to-vertical angles for robust fall detection, and tracks Laplacian variance for image blur. |
| Face blur / Garfield    | `src/privacy.py`       | Real-time output pixelation of faces to enforce GDPR/DSGVO compliance.                                                                         |
| Tattoo blur (prototype) | `src/tattoo.py`        | Prototype designed to detect and blur identifiable body art.                                                                                   |
| Alert verification      | `src/alert_filter.py`  | Implements temporal debouncing and multi-frame verification to eliminate false positives.                                                      |
| MQTT dispatch           | `src/dispatcher.py`    | Manages lightweight, non-blocking MQTT message payloads to remote endpoints.                                                                   |


---

## What is implemented


| Component             | Status                                         |
| --------------------- | ---------------------------------------------- |
| Live OpenCV UI        | Zone overlay, labels, hazards, replay clips    |
| Robot dashboard + sim | MQTT alerts and fake rover ETA                 |
| People + Re-ID        | YOLO + ByteTrack + histogram Re-ID             |
| PPE                   | YOLO helmet/goggles + HSV vest                 |
| Zones                 | ArUco-switched layouts; manual `w` cycle       |
| Environment           | Falls, smoke/fire, blur detection              |
| Alert hygiene         | Debounce, frame verification, LIMITED mode     |
| Privacy               | Output blur, tattoo prototype, Garfield toggle |
| MQTT                  | Fall, restricted entry, PPE, fire/smoke        |


**Not implemented:** calibrated floor geometry, printed marker rig, automated test suite, and certified safety sign-off. Static `zones/*.json` profiles remain in the repo for reference; the live demo path is hardcoded ArUco layouts in `src/zone_map.py`.

---

## Quick start

```bash
# Initialize and activate virtual environment
python -m venv .venv
source .venv/bin/activate
# Install dependencies
pip install -r requirements.txt
```

### ArUco markers (phones)

If you want to test marker-switched spatial layouts using your phone screen, generate the PNG targets using:

```bash
python generate_marker.py
```

Open the generated images located in `data/` on your phone screen at maximum brightness, making sure the outer white border remains visible.

### Live demo (recommended)

```bash
# Terminal 1: robot simulator
python robot_sim.py
# Terminal 2: vision pipeline (w = cycle layouts, g = Garfield/blur, q = quit)
python main.py --source 0
```
To open the dashboard on your browser, simply double-click on `robot_dashboard.html`.

MQTT topics: `hackatum/robot/dispatch`, `hackatum/robot/status`

---

## Config (`src/config.py`)

- `FAST_MODE` / `SMOOTH_MODE` — CPU vs quality
- `PPE_*`, `SMOKE_*`, `FALL_*` — thresholds and frame intervals
- `ALERT_DEBOUNCE_SECONDS`, `ALERT_VERIFY_*`, `RELIABILITY_*`
- `PRIVACY_CENSORSHIP_MODE`, `GARFIELD_SCALE` — blur vs Garfield (`images/garfield.png`)

---

## Privacy & limits

- **On-device processing.** All inference runs locally. Faces (and prototype tattoo regions) are censored on the **output** frame only. Internal tracking uses session pseudonyms (`Worker-A3F91C`), not employee IDs.
- **Not a certified safety system.** Hackathon demonstrator only — do not use for real go/no-go safety decisions or compliance sign-off.
- **Public MQTT.** Alerts publish to HiveMQ (`hackatum/robot/dispatch`). The robot dashboard is static HTML with no backend, which keeps our side simple, but the broker is shared and anyone can subscribe to the topic. Payloads contain only `team_id`, `zone_id`, `alert_type`, and pseudonymous labels — never video or raw biometrics.
- **Zone geometry is approximate.** ArUco marker ID (0–3) selects a demo wall layout; distance < 2 m vs ≥ 2 m switches between “close” and “far” band presets. Pose estimation assumes a 15 cm marker width and a synthetic camera matrix — accurate enough for a booth demo, not for production navigation. Zones are coloured rectangles in the image, not a calibrated MTU floor plan.
- **Detection heuristics.** Yellow-vest checks use HSV colour on the torso (informational in work zones, not a hard gate). Smoke/fire models can false-positive under glare or compression — raise `SMOKE_INFERENCE_INTERVAL` or `SMOKE_CONF_THRESHOLD` if needed. Re-ID headcounts rely on colour histograms; workers in identical uniforms may be merged or split incorrectly.
- **Degraded vision.** If Laplacian blur stays below `BLUR_LAPLACIAN_THRESHOLD` (default 20) for `RELIABILITY_BLUR_FRAMES` (3) consecutive frames, the HUD enters **LIMITED** mode: critical alerts (falls, restricted entry, fire/smoke) still dispatch; warning-level PPE robot pings may be suppressed.
- **Rate limits.** Zone/PPE console alerts debounce for `ALERT_DEBOUNCE_SECONDS` (5 s). Robot dispatches use `DISPATCH_COOLDOWN_SECONDS` (15 s) per person and alert type. Warning alerts need `ALERT_VERIFY_WARNING_FRAMES` (2) consecutive frames; critical alerts need `ALERT_VERIFY_CRITICAL_FRAMES` (1).

---

*Hackatum · MTU Smart Safety Monitoring*
