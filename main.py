import cv2
import argparse
import time
import sys
from src.engine import SafetyPipelineEngine
import src.config as config

def render_annotations(frame_data):
    """
    Renders annotations, safety statuses, and HUD info onto the processed_frame.
    """
    frame = frame_data.processed_frame
    h, w = frame.shape[:2]
    
    # 1. Draw Bounding Boxes and Labels for Tracked Persons
    for person in frame_data.persons:
        xmin, ymin, xmax, ymax = person.bbox
        
        # Color coding: Red = Fall (Critical), Orange = Unsafe if missing PPE, Green = Safe
        if person.is_fallen:
            color = (0, 0, 255)  # BGR Red
            status_text = "FALL DETECTED!"
        elif len(person.compliance_violations) > 0:
            color = (0, 140, 255)  # BGR Orange
            status_text = "UNSAFE"
        else:
            color = (0, 255, 0)  # BGR Green
            status_text = "SAFE"

        helmet_text = "Wears Helmet" if person.has_helmet else "No Helmet"
        glasses_text = "Wears Glasses" if person.has_glasses else "No Glasses"
        if person.has_helmet is None:
            helmet_text = "Helmet: Unknown"
        if person.has_glasses is None:
            glasses_text = "Glasses: Unknown"

        # Draw bounding box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        
        # Header tag background (three rows for labels)
        worker_text = f"Worker {person.person_id}"
        tag_text = f"{worker_text}"
        info_text = status_text
        details_text = f"{helmet_text} | {glasses_text}"
        (tw, th), _ = cv2.getTextSize(details_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        (iw, ih), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        (ww, wh), _ = cv2.getTextSize(worker_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        tag_width = max(tw, iw, ww)
        total_height = th + ih + wh + 10
        cv2.rectangle(frame, (xmin, ymin - total_height), (xmin + tag_width + 10, ymin), color, -1)
        
        # Draw header text
        text_color = (0, 0, 0) if color in [(0, 255, 0), (0, 140, 255)] else (255, 255, 255)
        y_offset = ymin - total_height + 16
        cv2.putText(
            frame,
            worker_text,
            (xmin + 5, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            text_color,
            1
        )
        cv2.putText(
            frame,
            details_text,
            (xmin + 5, y_offset + wh),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            text_color,
            1
        )
        cv2.putText(
            frame,
            info_text,
            (xmin + 5, y_offset + wh + th),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            text_color,
            1
        )
        
    # 2. Draw Top HUD Bar (ROI-only overlay, no full-frame copy)
    hud_h = 55
    roi = frame[0:hud_h, 0:w]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, roi, 0.35, 0, roi)
    
    fps = frame_data.extra_metadata.get("fps", 0)
    latency = frame_data.extra_metadata.get("latency_ms", 0)
    stage_ms = frame_data.extra_metadata.get("stage_ms", {})
    slowest = frame_data.extra_metadata.get("slowest_stage", "")
    slowest_ms = stage_ms.get(slowest, 0) if slowest else 0
    mode_label = "FAST" if config.FAST_MODE else "SMOOTH" if config.SMOOTH_MODE else "STD"
    
    # Title
    cv2.putText(frame, "MTU WORKER MONITORING SYSTEM", (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    # Counters
    stats_str = f"Live Count: {frame_data.current_people_count} | Cumulative Unique: {frame_data.total_unique_people}"
    cv2.putText(frame, stats_str, (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    
    # System stats
    perf_str = f"FPS: {fps} | {latency}ms | {mode_label}"
    cv2.putText(frame, perf_str, (w - 280, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 255, 100), 1)
    if slowest:
        stage_str = f"Slowest: {slowest} ({slowest_ms}ms)"
        cv2.putText(frame, stage_str, (w - 280, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 220, 180), 1)
    
    # 3. Draw Environmental Quality Overlays
    if frame_data.is_image_blurry:
        cv2.rectangle(frame, (w - 180, 28), (w - 15, 48), (0, 0, 255), -1)
        cv2.putText(frame, "BLURRY / FOG WARNING", (w - 170, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        
    if frame_data.is_smoke_detected:
        cv2.rectangle(frame, (w - 330, 28), (w - 190, 48), (0, 0, 255), -1)
        cv2.putText(frame, "FIRE/SMOKE HAZARD", (w - 320, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

def main():
    parser = argparse.ArgumentParser(description="Hackatum safety monitoring computer vision pipeline.")
    parser.add_argument("--mock", action="store_true", help="Run with simulated warehouse video and inputs")
    parser.add_argument("--source", type=str, default=None, help="Video source (e.g. '0' for webcam, or path to file)")
    args = parser.parse_args()
    
    # Initialize Engine
    source = args.source
    if source is not None and source.isdigit():
        source = int(source)
        
    engine = SafetyPipelineEngine(use_mock=args.mock, video_source=source)
    stream = engine.stream_frames()
    
    print("\n" + "="*50)
    print("MTU Pipeline Engine Active.")
    print("Press 'q' in the window to quit.")
    print("Consuming feed...")
    print("="*50 + "\n")
    
    try:
        for frame_data in stream:
            # Add drawing layer
            render_annotations(frame_data)
            
            # Print console logs for active alerts (just for terminal feedback)
            if frame_data.alerts:
                for alert in frame_data.alerts:
                    print(f"[{time.strftime('%H:%M:%S', time.localtime(alert['timestamp']))}] "
                          f"[{alert['severity']}] {alert['message']}")
            
            # Display image
            cv2.imshow("MTU Room Monitor", frame_data.processed_frame)
            
            # Read keyboard quit input
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nShutdown command received. Closing stream.")
                break
                
    except KeyboardInterrupt:
        print("\nShutdown via keyboard interrupt.")
    finally:
        engine.release()
        cv2.destroyAllWindows()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()
