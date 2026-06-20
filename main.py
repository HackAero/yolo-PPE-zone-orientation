import cv2
import argparse
import time
import sys
from src.engine import SafetyPipelineEngine
from src.zone_map import draw_zones_overlay
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

    draw_zones_overlay(frame, frame_data)
    
    # Optional overlay for translucent shapes
    overlay = frame.copy()
    
    post_blend_text = []
    danger_area_count = 0

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
        
        zone_id = person.metadata.get("zone_id", "safe")
        has_yellow_vest = person.metadata.get("has_yellow_vest")
        
        is_safe = True
        messages = []
        color = (0, 200, 0) # Green
        
        if person.is_fallen:
            messages.append("FALL DETECTED")
            is_safe = False
            color = (0, 0, 255)
            danger_area_count += 1
        elif zone_id == "restricted":
            messages.append("SOMEONE ENTERED THE RESTRICTED AREA")
            is_safe = False
            color = (0, 0, 255)
            danger_area_count += 1
        elif zone_id == "work_floor":
            viols = []
            if person.has_helmet is False:
                viols.append("NO HELMET")
            if person.has_glasses is False:
                viols.append("NO GLASSES")
            if has_yellow_vest is False:
                viols.append("NO VEST")
            
            if viols:
                messages.extend(viols)
                is_safe = False
                color = (0, 140, 255) # Orange
                danger_area_count += 1
        elif zone_id == "safe":
            # can wear nothing
            pass
            
        if is_safe:
            messages.append("SAFE")
            
        # Draw box
        if is_safe:
            draw_corner_brackets(frame, xmin, ymin, xmax, ymax, color, thickness=2, length=15)
        else:
            cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), color, -1)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            
        # Draw multiline minimalist text
        y_offset = ymin - 8
        # We'll draw them backwards so they stack upwards
        for msg in reversed(messages):
            # Use smaller font
            font_scale = 0.35
            thickness = 1
            (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # draw pill
            y_top = y_offset - th - 4
            cv2.rectangle(frame if not is_safe else overlay, (xmin, y_top), (xmin + tw + 6, y_offset + 2), color if not is_safe else (0, 0, 0), -1)
            
            # queue text
            text_color = (255, 255, 255)
            post_blend_text.append((msg, (xmin + 3, y_offset - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness))
            
            y_offset -= (th + 8)

        # Draw ID separately at the top
        msg_id = f"ID: {person.person_id}"
        (tw, th), _ = cv2.getTextSize(msg_id, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        y_top = y_offset - th - 4
        cv2.rectangle(frame if not is_safe else overlay, (xmin, y_top), (xmin + tw + 6, y_offset + 2), (0,0,0), -1)
        post_blend_text.append((msg_id, (xmin + 3, y_offset - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1))
            
    # Apply alpha blend for danger boxes and safe ID pills
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    
    # HUD Overlay (For top metrics and hazards)
    hud_overlay = frame.copy()

    if danger_area_count > 0:
        warning_text = f"SAFETY ALERT: {danger_area_count} WORKER(S) IN DANGER"
        (dw, dh), _ = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(hud_overlay, (10, 54), (min(w - 10, 10 + dw + 24), 54 + dh + 20), (0, 0, 255), -1)
        post_blend_text.append((warning_text, (22, 54 + dh + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2))
    
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
    window_name = "MTU Room Monitor"
    parser = argparse.ArgumentParser(description="Hackatum safety monitoring computer vision pipeline.")
    parser.add_argument("--mock", action="store_true", help="Run with simulated warehouse video and inputs")
    parser.add_argument("--source", type=str, default=None, help="Video source (e.g. '0' for webcam, or path to file)")
    parser.add_argument(
        "--camera-profile",
        choices=["monitor", "rover"],
        default="monitor",
        help="Zone layout for fixed monitor cam vs rover-mounted cam (default: monitor)",
    )
    parser.add_argument(
        "--zones-file",
        type=str,
        default=None,
        help="Custom zones JSON path (overrides --camera-profile)",
    )
    args = parser.parse_args()
    
    # Initialize Engine
    source = args.source
    if source is not None and source.isdigit():
        source = int(source)
        
    engine = SafetyPipelineEngine(
        use_mock=args.mock,
        video_source=source,
        camera_profile=args.camera_profile,
        zones_path=args.zones_file,
    )
    stream = engine.stream_frames()
    
    print("\n" + "="*50)
    print("MTU Pipeline Engine Active.")
    print("Press 'q' in the window to quit.")
    print("Consuming feed...")
    print("="*50 + "\n")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1800, 1350)
    
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
            display_frame = cv2.resize(
                frame_data.processed_frame,
                None,
                fx=2.0,
                fy=2.0,
                interpolation=cv2.INTER_LINEAR,
            )
            cv2.imshow(window_name, display_frame)
            
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
