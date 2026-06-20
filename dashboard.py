import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
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
    with m_col2:
        unique_total_metric = st.empty()
        fps_metric = st.empty()
        tattoo_blur_metric = st.empty()
        
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
    
    try:
        for frame_data in stream:
            # Check for user actions halting stream
            if not run_pipeline:
                break
            
            # Update parameters dynamically from sidebar sliders
            # Ensure blur kernel is odd
            kernel_size = blur_intensity if blur_intensity % 2 == 1 else blur_intensity + 1
            import src.config as config
            config.PRIVACY_BLUR_KERNEL_SIZE = (kernel_size, kernel_size)
            config.PERSON_CONF_THRESHOLD = conf_threshold
            
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
            frame_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
            
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
                table_placeholder.dataframe(df, use_container_width=True, hide_index=True)
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