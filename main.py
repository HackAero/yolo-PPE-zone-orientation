import cv2
import argparse
import time
import sys
from src.engine import SafetyPipelineEngine
from src.session_labels import worker_label
import src.config as config

def draw_corner_brackets(frame, xmin, ymin, xmax, ymax, color, thickness=2, length=15):
    # Top-left
    cv2.line(frame, (xmin, ymin), (xmin + length, ymin), color, thickness)
    cv2.line(frame, (xmin, ymin), (xmin, ymin + length), color, thickness)
    # Top-right
    cv2.line(frame, (xmax, ymin), (xmax - length, ymin), color, thickness)
    cv2.line(frame, (xmax, ymin), (xmax, ymin + length), color, thickness)
    # Bottom-left
    cv2.line(frame, (xmin, ymax), (xmin + length, ymax), color, thickness)
    cv2.line(frame, (xmin, ymax), (xmin, ymax - length), color, thickness)
    # Bottom-right
    cv2.line(frame, (xmax, ymax), (xmax - length, ymax), color, thickness)
    cv2.line(frame, (xmax, ymax), (xmax, ymax - length), color, thickness)

def render_annotations(frame_data):
    """
    Renders sleek, minimalist annotations, safety statuses, and HUD info onto the processed_frame.
    """
    frame = frame_data.processed_frame
    h, w = frame.shape[:2]
    
    # Optional overlay for translucent shapes
    overlay = frame.copy()
    
    post_blend_text = []

    ppe_debug = frame_data.extra_metadata.get("ppe_debug", {})
    helmet_boxes = ppe_debug.get("helmet_boxes", []) if isinstance(ppe_debug, dict) else []
    glasses_boxes = ppe_debug.get("glasses_boxes", []) if isinstance(ppe_debug, dict) else []
    raw_detections = ppe_debug.get("raw_detections", []) if isinstance(ppe_debug, dict) else []

    for x1, y1, x2, y2, conf in helmet_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"HELMET {conf:.2f}", (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    for x1, y1, x2, y2, conf in glasses_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, f"GOGGLES {conf:.2f}", (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    for item in raw_detections:
        if not isinstance(item, (list, tuple)) or len(item) != 6:
            continue
        label, x1, y1, x2, y2, conf = item
        label_text = str(label).upper()
        raw_color = (0, 255, 0) if label_text == "HELMET" else (255, 255, 0) if label_text == "GOGGLES" else (180, 180, 180)
        cv2.putText(
            frame,
            f"{label_text} RAW {conf:.2f}",
            (x1, min(h - 12, y2 + 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            raw_color,
            1,
        )

    if helmet_boxes or glasses_boxes:
        legend = "PPE DEBUG: green=helmet  cyan=goggles"
        (lw, lh), _ = cv2.getTextSize(legend, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (10, h - lh - 22), (10 + lw + 14, h - 10), (0, 0, 0), -1)
        cv2.putText(frame, legend, (17, h - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1)
        if raw_detections:
            raw_legend = f"RAW DETECTIONS: {len(raw_detections)}"
            (rw, rh), _ = cv2.getTextSize(raw_legend, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (10, h - lh - rh - 30), (10 + rw + 14, h - lh - 26), (0, 0, 0), -1)
            cv2.putText(frame, raw_legend, (17, h - lh - 33), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1)
    
    # 1. Draw Bounding Boxes and Labels for Tracked Persons
    for person in frame_data.persons:
        xmin, ymin, xmax, ymax = person.bbox
        has_yellow_vest = person.metadata.get("has_yellow_vest")
        if has_yellow_vest is True:
            vest_text = "VEST: YES"
        elif has_yellow_vest is False:
            vest_text = "VEST: NO"
        else:
            vest_text = "VEST: ?"
        
        is_safe = True
        status_text = f"ID {person.person_id} | SAFE | {vest_text}"
        color = (0, 200, 0)  # Green for safe
        
        if person.is_fallen:
            color = (0, 0, 255)  # Red
            status_text = f"ID {person.person_id} | FALL DETECTED | {vest_text}"
            is_safe = False
        elif len(person.compliance_violations) > 0:
            color = (0, 140, 255)  # Orange
            viols = []
            if "Helmet" in person.compliance_violations: viols.append("NO HELMET")
            if "Glasses" in person.compliance_violations: viols.append("NO GLASSES")
            status_text = f"ID {person.person_id} | UNSAFE: {', '.join(viols)} | {vest_text}"
            is_safe = False

        if is_safe:
            # Minimalist corner brackets
            draw_corner_brackets(frame, xmin, ymin, xmax, ymax, color, thickness=2, length=15)
            # Small ID pill
            (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2)
            cv2.rectangle(overlay, (xmin, ymin - th - 8), (xmin + tw + 10, ymin), (0, 0, 0), -1)
            post_blend_text.append((status_text, (xmin + 5, ymin - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1))
        else:
            # Danger: Full thin box + translucent fill
            cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), color, -1)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            
            # Danger ID pill
            (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2)
            cv2.rectangle(frame, (xmin, ymin - th - 8), (xmin + tw + 10, ymin), color, -1)
            post_blend_text.append((status_text, (xmin + 5, ymin - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1))
            
    # Apply alpha blend for danger boxes and safe ID pills
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    
    # HUD Overlay (For top metrics and hazards)
    hud_overlay = frame.copy()
    
    # 2. Draw Top Left Counters
    stats_str = f"LIVE: {frame_data.current_people_count}   TOTAL: {frame_data.total_unique_people}"
    (sw, sh), _ = cv2.getTextSize(stats_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(hud_overlay, (10, 10), (10 + sw + 20, 10 + sh + 16), (0, 0, 0), -1)
    post_blend_text.append((stats_str, (20, 10 + sh + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1))
    
    # 3. Draw Environmental Hazards (Top Right)
    hazard_x = w - 10
    if frame_data.is_smoke_detected:
        text = "FIRE / SMOKE"
        (hw, hh), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(hud_overlay, (hazard_x - hw - 20, 10), (hazard_x, 10 + hh + 16), (0, 0, 255), -1)
        post_blend_text.append((text, (hazard_x - hw - 10, 10 + hh + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1))
        hazard_x -= (hw + 30)
        
    if frame_data.is_image_blurry:
        text = "BLUR / FOG"
        (hw, hh), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(hud_overlay, (hazard_x - hw - 20, 10), (hazard_x, 10 + hh + 16), (0, 140, 255), -1)
        post_blend_text.append((text, (hazard_x - hw - 10, 10 + hh + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1))
        hazard_x -= (hw + 30)

    cv2.addWeighted(hud_overlay, 0.65, frame, 0.35, 0, frame)
    
    # Draw all crisp text over the blended overlays
    for text, pos, font, scale, color, thick in post_blend_text:
        cv2.putText(frame, text, pos, font, scale, color, thick)
        
    # 4. System stats (Bottom Right, muted, no background pill)
    fps = frame_data.extra_metadata.get("fps", 0)
    latency = frame_data.extra_metadata.get("latency_ms", 0)
    mode_label = "FAST" if config.FAST_MODE else "SMOOTH" if config.SMOOTH_MODE else "STD"
    perf_str = f"FPS: {fps} | Latency: {latency}ms | {mode_label}"
    
    (pw, ph), _ = cv2.getTextSize(perf_str, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.putText(frame, perf_str, (w - pw - 10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    
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
                    if not alert.get("debounced", False):
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
