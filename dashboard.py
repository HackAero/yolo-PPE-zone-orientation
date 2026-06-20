import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
from collections import deque
from src.engine import SafetyPipelineEngine
from src.session_labels import worker_label
from main import render_annotations, SupervisorReplayRecorder, build_critical_event

# Set page config
st.set_page_config(
    page_title="MTU Safety Dashboard",
    page_icon="🏭",
    layout="wide"
)

# App Title & Description
st.title("🏭 MTU Smart Safety Monitoring System")
st.markdown("""
This dashboard monitors warehouse environments in real-time, enforcing safety compliance (helmets, eyewear), 
detecting industrial falls, ensuring privacy (blurring faces/tattoos), and monitoring environment visibility (smoke/blur).
""")

# Setup Sidebar Configurations
st.sidebar.header("⚙️ System Control Panel")

# Running Mode
run_mode = st.sidebar.radio(
    "Select Source Mode:",
    ("Simulated Mock Engine", "Live Camera / Video Stream")
)

# Advanced Controls
st.sidebar.divider()
st.sidebar.subheader("Safety Rules Thresholds")
conf_threshold = st.sidebar.slider("Person Detection Conf", 0.1, 1.0, 0.4, 0.05)
blur_intensity = st.sidebar.slider("Privacy Blur Intensity", 15, 101, 51, 2)  # Odd number
danger_window_minutes = st.sidebar.slider("Danger Histogram Window (min)", 1, 30, 5, 1)
censorship_mode = st.sidebar.radio(
    "Face censorship",
    ("Blur", "Garfield"),
    horizontal=True,
)
is_smoke_simulated = False

if run_mode == "Simulated Mock Engine":
    use_mock = True
    st.sidebar.info("Running in simulation mode. No webcam or model files required.")
else:
    use_mock = False
    source_input = st.sidebar.text_input("Camera Index / Video Path:", value="0")
    # Resolve source index
    if source_input.isdigit():
        video_source = int(source_input)
    else:
        video_source = source_input

# Start/Stop Session
st.sidebar.divider()
run_pipeline = st.sidebar.checkbox("🚀 Start Video Analysis", value=True)

# Layout Setup
col_video, col_analytics = st.columns([5, 3])

with col_video:
    st.subheader("🎥 Live Anonymized Room Feed")
    # Placeholder for the live frame stream
    frame_placeholder = st.empty()

with col_analytics:
    st.subheader("📊 Live Room Metrics")
    
    # Grid of Metrics
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        current_workers_metric = st.empty()
        safety_rating_metric = st.empty()
        fire_alarm_metric = st.empty()
        verification_points_metric = st.empty()
    with m_col2:
        unique_total_metric = st.empty()
        fps_metric = st.empty()
        tattoo_blur_metric = st.empty()
        danger_score_metric = st.empty()
        
    st.divider()
    st.subheader("📉 Danger Histogram")
    danger_histogram_placeholder = st.empty()
    danger_trend_placeholder = st.empty()
    with st.expander("Scoring Legend (How Danger Is Calculated)", expanded=False):
        st.markdown(
            """
**Alert Weights (Verification Points + Danger):**
- Environment alert (fire/smoke): +50
- Restricted entry: +40
- Fall alert: +35
- Zone PPE violation: +15
- PPE violation: +15
- Environment warning: +5

**Zone-Aware Person Risk (Per Frame):**
- Restricted zone base: +28
- Work floor base: +10
- Safe zone base: +2
- Restricted/no-entry presence bonus: +18
- Missing helmet base: +20
- Missing glasses base: +18
- Missing vest base: +4

**Zone Multipliers for Missing PPE:**
- Restricted zone: x2.2
- Work floor: x1.3
- Safe/other: x1.0

**Additional Effects:**
- Fall detected on person: +30
- Fully compliant worker in work floor: -4
- Active verifying alerts: up to +10

**Danger Bands:**
- 0-10 Safe
- 11-25 Low
- 26-45 Medium
- 46-70 High
- 71-100 Critical
            """
        )

    st.divider()
    st.subheader("⚠️ Active Safety Violations & Alerts")
    # Running alerts box
    alerts_placeholder = st.empty()
    
    st.divider()
    st.subheader("👷 Active Room Manifest")
    # Live workers compliance table
    table_placeholder = st.empty()

# Start the pipeline stream if checked
if run_pipeline:
    # Initialize Engine
    source = None if run_mode == "Simulated Mock Engine" else video_source
    engine = SafetyPipelineEngine(use_mock=use_mock, video_source=source)
    
    # Override configuration thresholds in running stages
    engine.tracker_stage.use_mock = use_mock
    engine.compliance_stage.use_mock = use_mock
    engine.environment_stage.use_mock = use_mock
    engine.privacy_stage.use_mock = use_mock
    
    stream = engine.stream_frames()
    replay = SupervisorReplayRecorder()
    
    # Store running alert history
    alert_history = []
    danger_history = deque(maxlen=3600)
    verification_points_total = 0

    alert_weights = {
        "ENVIRONMENT_ALERT": 50,
        "RESTRICTED_ENTRY": 40,
        "FALL_ALERT": 35,
        "ZONE_PPE_VIOLATION": 15,
        "PPE_VIOLATION": 15,
        "ENVIRONMENT_WARNING": 5,
    }

    zone_base_risk = {
        "restricted": 28,
        "work_floor": 10,
        "safe": 2,
    }

    zone_violation_risk = {
        "Helmet": 20,
        "Glasses": 18,
        "Vest": 4,
    }

    def compute_frame_scores(frame_data):
        verification_points = 0
        danger_points = 0

        for alert in frame_data.alerts:
            if alert.get("debounced", False):
                continue

            alert_type = str(alert.get("type", ""))
            severity = str(alert.get("severity", "Warning")).upper()
            weight = int(alert_weights.get(alert_type, 8 if severity == "WARNING" else 12))

            verification_points += weight
            danger_points += weight

        # Zone-aware person risk: staying in restricted area is intrinsically
        # more dangerous, and missing PPE there is penalized much more.
        for person in frame_data.persons:
            zone_id = str(person.metadata.get("zone_id", "safe"))
            zone_violations = list(person.metadata.get("zone_violations", []))
            zone_forbidden = bool(person.metadata.get("zone_forbidden", False))

            person_risk = int(zone_base_risk.get(zone_id, 4))

            if zone_forbidden or zone_id == "restricted":
                # No-entry area: high baseline risk just by presence.
                person_risk += 18

            for violation in zone_violations:
                v = str(violation)
                base = int(zone_violation_risk.get(v, 8))
                if zone_forbidden or zone_id == "restricted":
                    # Missing PPE in restricted zone must dominate histogram.
                    person_risk += int(base * 2.2)
                elif zone_id == "work_floor":
                    person_risk += int(base * 1.3)
                else:
                    person_risk += base

            if person.is_fallen:
                person_risk += 30

            # Reward compliance in non-restricted areas.
            if zone_id == "work_floor" and not zone_violations:
                person_risk = max(0, person_risk - 4)

            danger_points += person_risk
            verification_points += max(0, person_risk // 3)

        verifying_count = int(frame_data.extra_metadata.get("verifying_count", 0) or 0)
        if verifying_count > 0:
            danger_points += min(10, verifying_count * 2)

        danger_score = int(max(0, min(100, danger_points)))
        return verification_points, danger_score

    def render_danger_histogram(history, now_ts, window_minutes):
        window_seconds = int(window_minutes * 60)
        recent = [item for item in history if now_ts - item[0] <= window_seconds]

        if not recent:
            return None, None

        scores = [item[1] for item in recent]
        bins = {
            "Safe (0-10)": 0,
            "Low (11-25)": 0,
            "Medium (26-45)": 0,
            "High (46-70)": 0,
            "Critical (71-100)": 0,
        }
        for s in scores:
            if s <= 10:
                bins["Safe (0-10)"] += 1
            elif s <= 25:
                bins["Low (11-25)"] += 1
            elif s <= 45:
                bins["Medium (26-45)"] += 1
            elif s <= 70:
                bins["High (46-70)"] += 1
            else:
                bins["Critical (71-100)"] += 1

        hist_df = pd.DataFrame(
            {"Range": list(bins.keys()), "Frames": list(bins.values())}
        ).set_index("Range")

        trend_bucket = {}
        for ts, score, _vpts in recent:
            minute_bucket = int(ts // 60)
            trend_bucket.setdefault(minute_bucket, []).append(score)

        trend_rows = []
        for bucket in sorted(trend_bucket.keys()):
            avg_score = float(np.mean(trend_bucket[bucket]))
            label = time.strftime("%H:%M", time.localtime(bucket * 60))
            trend_rows.append((label, avg_score))

        trend_df = pd.DataFrame(trend_rows, columns=["Time", "DangerScore"]).set_index("Time")
        return hist_df, trend_df
    
    frame_count = 0
    try:
        for frame_data in stream:
            frame_count += 1
            if not frame_data or frame_data.processed_frame is None:
                continue
            # Check for user actions halting stream
            if not run_pipeline:
                break
            
            # Update parameters dynamically from sidebar sliders
            # Ensure blur kernel is odd
            kernel_size = blur_intensity if blur_intensity % 2 == 1 else blur_intensity + 1
            import src.config as config
            config.PRIVACY_BLUR_KERNEL_SIZE = (kernel_size, kernel_size)
            config.PERSON_CONF_THRESHOLD = conf_threshold
            config.PRIVACY_CENSORSHIP_MODE = (
                "garfield" if censorship_mode == "Garfield" else "blur"
            )
            
            # Redact and annotate the processed frame
            render_annotations(frame_data)

            # Critical event replay capture (same behavior as OpenCV main.py path)
            event = build_critical_event(frame_data)
            if event is not None:
                replay.trigger(frame_data.timestamp, event)

            replay.draw_overlay(frame_data.processed_frame, frame_data.timestamp)
            replay.add_frame(frame_data.timestamp, frame_data.processed_frame)
            
            # Streamlit needs RGB image format, cv2 operates on BGR
            rgb_frame = cv2.cvtColor(frame_data.processed_frame, cv2.COLOR_BGR2RGB)
            
            # Display image in Streamlit
            frame_placeholder.image(rgb_frame, channels="RGB", width="stretch")
            
            # Update metrics
            current_workers_metric.metric("Workers Present", frame_data.current_people_count)
            unique_total_metric.metric("Total Visitors Today", frame_data.total_unique_people)
            
            # Calculate safety score
            num_violations = 0
            num_fallen = 0
            for p in frame_data.persons:
                if len(p.compliance_violations) > 0:
                    num_violations += 1
                if p.is_fallen:
                    num_fallen += 1
            
            safe_workers = len(frame_data.persons) - num_violations - num_fallen
            safety_score = int((safe_workers / len(frame_data.persons)) * 100) if len(frame_data.persons) > 0 else 100
            
            safety_rating_metric.metric("Safety Rating", f"{safety_score}%", 
                                        delta=None if safety_score == 100 else f"-{100 - safety_score}% Compliance Deficit",
                                        delta_color="off" if safety_score == 100 else "inverse")
            
            fps = frame_data.extra_metadata.get("fps", 0)
            fps_metric.metric("Engine Performance", f"{fps} FPS", f"{frame_data.extra_metadata.get('latency_ms', 0)} ms/frame")

            fire_active = bool(frame_data.is_fire_detected or frame_data.is_smoke_detected)
            fire_alarm_metric.metric(
                "Fire Alarm",
                "ACTIVE" if fire_active else "Clear",
                delta="SMOKE/FIRE" if fire_active else "No hazard",
                delta_color="inverse" if fire_active else "off",
            )

            frame_verification_points, frame_danger_score = compute_frame_scores(frame_data)
            verification_points_total += frame_verification_points
            danger_history.append((float(frame_data.timestamp), frame_danger_score, frame_verification_points))

            verification_points_metric.metric(
                "Verification Points",
                f"{verification_points_total}",
                delta=f"+{frame_verification_points}/frame" if frame_verification_points > 0 else "0/frame",
                delta_color="off",
            )

            danger_label = "SAFE"
            if frame_danger_score > 70:
                danger_label = "CRITICAL"
            elif frame_danger_score > 45:
                danger_label = "HIGH"
            elif frame_danger_score > 25:
                danger_label = "MEDIUM"
            elif frame_danger_score > 10:
                danger_label = "LOW"

            danger_score_metric.metric(
                "Danger Score",
                f"{frame_danger_score}",
                delta=danger_label,
                delta_color="inverse" if frame_danger_score > 45 else "off",
            )

            if frame_count % 15 == 0:
                hist_df, trend_df = render_danger_histogram(
                    danger_history,
                    float(frame_data.timestamp),
                    danger_window_minutes,
                )
                if hist_df is not None:
                    danger_histogram_placeholder.bar_chart(hist_df)
                else:
                    danger_histogram_placeholder.info("Collecting danger data...")

                if trend_df is not None and not trend_df.empty:
                    danger_trend_placeholder.line_chart(trend_df)
                else:
                    danger_trend_placeholder.info("Collecting danger trend...")

            tattoo_enabled = bool(frame_data.extra_metadata.get("privacy_tattoo_blur_enabled", False))
            tattoo_ready = bool(frame_data.extra_metadata.get("privacy_tattoo_detector_ready", False))
            tattoo_regions = int(frame_data.extra_metadata.get("privacy_tattoo_regions_blurred", 0) or 0)
            tattoo_status = "OFF"
            if tattoo_enabled and tattoo_ready:
                tattoo_status = f"ON ({tattoo_regions})"
            elif tattoo_enabled and not tattoo_ready:
                tattoo_status = "ON (MODEL MISSING)"

            tattoo_blur_metric.metric(
                "Tattoo Blur",
                tattoo_status,
                delta=f"{tattoo_regions} regions/frame" if tattoo_enabled and tattoo_ready else None,
                delta_color="off",
            )
            
            # Parse alerts and maintain a small list of unique alerts
            for alert in frame_data.alerts:
                # Add to history if unique (avoid duplicates in list display)
                alert_sig = (alert["type"], alert.get("person_id", None), alert["message"])
                if alert_sig not in [x[0] for x in alert_history]:
                    alert_history.insert(0, (alert_sig, alert["severity"], alert["message"], time.strftime("%H:%M:%S")))
                    if len(alert_history) > 8:
                        alert_history.pop()
            
            # Render alerts in HTML list
            alerts_html = "<div style='height: 180px; overflow-y: scroll; border: 1px solid #444; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 13px;'>"
            if alert_history:
                for sig, severity, msg, timestamp in alert_history:
                    if severity == "Critical":
                        color = "#ff4d4d"
                        badge = "🚨 CRITICAL"
                    elif severity == "Warning":
                        color = "#ffa500"
                        badge = "⚠️ WARNING"
                    else:
                        color = "#1e90ff"
                        badge = "ℹ️ INFO"
                    alerts_html += f"<p style='color:{color}; margin: 4px 0;'>[{timestamp}] <b>{badge}</b>: {msg}</p>"
            else:
                alerts_html += "<p style='color:#888; font-style:italic; text-align:center; padding-top: 60px;'>No active alerts. Site secure.</p>"
            alerts_html += "</div>"
            alerts_placeholder.markdown(alerts_html, unsafe_allow_html=True)
            
            if frame_count % 15 == 0:
                # Render manifest table
                manifest_data = []
                for p in frame_data.persons:
                    status_symbol = "🔴 Lying Down" if p.is_fallen else "🟢 Standing"
                    helmet_symbol = "✅ Wearing" if p.has_helmet else "❌ Missing"
                    glasses_symbol = "✅ Wearing" if p.has_glasses else "❌ Missing"
                    
                    manifest_data.append({
                        "Worker ID": worker_label(p.person_id),
                        "Posture Status": status_symbol,
                        "Hard Hat": helmet_symbol,
                        "Safety Eyewear": glasses_symbol,
                        "Tattoo Privacy": "🟢 Active" if tattoo_enabled and tattoo_ready else ("🟠 Enabled (No Model)" if tattoo_enabled else "⚪ Off"),
                    })
                    
                if manifest_data:
                    df = pd.DataFrame(manifest_data)
                    # Display table without index
                    table_placeholder.dataframe(df, width="stretch", hide_index=True)
                else:
                    table_placeholder.info("No personnel currently registered in monitored area.")
                
            # Slow down simulation frame rate to look natural (~15 FPS max)
            if use_mock:
                time.sleep(0.04)
                
    except Exception as e:
        st.error(f"Error in pipeline stream: {e}")
        import traceback
        st.code(traceback.format_exc())
    finally:
        engine.release()
else:
    st.info("System is offline. Toggle 'Start Video Analysis' on the sidebar control panel.")