import cv2
import argparse
import time
import sys
import os
import json
from datetime import datetime
from collections import deque
from src.engine import SafetyPipelineEngine
from src.session_labels import worker_label
from src.zone_map import draw_zones_overlay
import src.config as config


class SupervisorReplayRecorder:
    def __init__(self, output_dir="data/replays", pre_seconds=5.0, post_seconds=5.0, cooldown_seconds=15.0):
        self.output_dir = output_dir
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.cooldown_seconds = cooldown_seconds

        self._pre_buffer = deque(maxlen=1200)
        self._active_event = None
        self._last_trigger_ts = 0.0
        self._popup_until_ts = 0.0
        self._popup_lines = []
        self._saved_until_ts = 0.0
        self._saved_text = ""

        os.makedirs(self.output_dir, exist_ok=True)

    def add_frame(self, timestamp, frame):
        frame_copy = frame.copy()
        self._pre_buffer.append((timestamp, frame_copy))

        if self._active_event is None:
            return

        self._active_event["frames"].append(frame_copy)
        if timestamp >= self._active_event["end_ts"]:
            self._finalize_event(timestamp)

    def trigger(self, timestamp, event):
        if self._active_event is not None:
            return False
        if timestamp - self._last_trigger_ts < self.cooldown_seconds:
            return False

        pre_start = timestamp - self.pre_seconds
        pre_frames = [frame for ts, frame in self._pre_buffer if ts >= pre_start]

        self._active_event = {
            "event": event,
            "start_ts": timestamp,
            "end_ts": timestamp + self.post_seconds,
            "frames": pre_frames,
        }
        self._last_trigger_ts = timestamp

        self._popup_until_ts = timestamp + 4.0
        self._popup_lines = [
            f"CRITICAL EVENT: {event['type']}",
            event["summary"],
            f"ACTION: {event['action']} | CONF: {event['confidence']}",
        ]
        return True

    def _finalize_event(self, timestamp):
        if self._active_event is None:
            return

        event = self._active_event["event"]
        frames = self._active_event["frames"]
        self._active_event = None

        if not frames:
            return

        # Normalize replay frames so codecs receive consistent 8-bit BGR images.
        norm_frames = []
        base_h, base_w = frames[0].shape[:2]
        base_w = int(base_w) - (int(base_w) % 2)
        base_h = int(base_h) - (int(base_h) % 2)
        for frame in frames:
            if frame is None or frame.size == 0:
                continue
            out = frame
            if len(out.shape) == 2:
                out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
            elif out.shape[2] == 4:
                out = cv2.cvtColor(out, cv2.COLOR_BGRA2BGR)
            if out.dtype != "uint8":
                out = out.clip(0, 255).astype("uint8")
            if out.shape[1] != base_w or out.shape[0] != base_h:
                out = cv2.resize(out, (base_w, base_h), interpolation=cv2.INTER_LINEAR)
            else:
                out = out[:base_h, :base_w]
            norm_frames.append(out.copy())

        if not norm_frames:
            return

        height, width = norm_frames[0].shape[:2]
        duration = max(0.1, self.pre_seconds + self.post_seconds)
        fps = max(8.0, min(24.0, len(norm_frames) / duration))

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        event_slug = event["type"].lower().replace("/", "_").replace(" ", "_")
        base_name = f"event_{stamp}_{event_slug}"
        avi_path = os.path.join(self.output_dir, f"{base_name}.avi")
        mp4_path = os.path.join(self.output_dir, f"{base_name}.mp4")
        meta_path = os.path.join(self.output_dir, f"{base_name}.json")

        # Primary export: MJPG AVI is broadly decodable on macOS and avoids green-frame artifacts.
        avi_writer = cv2.VideoWriter(
            avi_path,
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (width, height),
        )

        if avi_writer.isOpened():
            for frame in norm_frames:
                avi_writer.write(frame)
            avi_writer.release()
            replay_path = avi_path
        else:
            replay_path = mp4_path

        # Secondary export: keep mp4 as optional compatibility artifact.
        mp4_writer = cv2.VideoWriter(
            mp4_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if mp4_writer.isOpened():
            for frame in norm_frames:
                mp4_writer.write(frame)
            mp4_writer.release()

        metadata = {
            "timestamp": timestamp,
            "event": event,
            "video_path": replay_path,
            "video_path_avi": avi_path,
            "video_path_mp4": mp4_path,
            "frame_count": len(norm_frames),
            "fps": fps,
            "pre_seconds": self.pre_seconds,
            "post_seconds": self.post_seconds,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        self._saved_until_ts = timestamp + 3.0
        self._saved_text = f"REPLAY SAVED: {os.path.basename(replay_path)}"

    def draw_overlay(self, frame, timestamp):
        if timestamp <= self._popup_until_ts and self._popup_lines:
            line_height = 24
            panel_h = 18 + line_height * len(self._popup_lines)
            panel_w = max(420, int(frame.shape[1] * 0.55))
            cv2.rectangle(frame, (12, 90), (12 + panel_w, 90 + panel_h), (0, 0, 255), -1)
            for i, text in enumerate(self._popup_lines):
                y = 90 + 24 + i * line_height
                scale = 0.6 if i == 0 else 0.48
                thick = 2 if i == 0 else 1
                cv2.putText(frame, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick)

        if timestamp <= self._saved_until_ts and self._saved_text:
            cv2.rectangle(frame, (12, 64), (620, 86), (0, 120, 0), -1)
            cv2.putText(frame, self._saved_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)


def build_critical_event(frame_data):
    event_time = datetime.fromtimestamp(frame_data.timestamp).strftime("%H:%M:%S")

    if frame_data.is_smoke_detected:
        return {
            "type": "FIRE/SMOKE",
            "summary": f"Smoke/fire detected at {event_time} in camera view",
            "where": "camera_view",
            "who": "environment",
            "confidence": "HIGH",
            "action": "DISPATCH ROVER / EVACUATE",
        }

    for person in frame_data.persons:
        if person.is_fallen:
            zone = person.metadata.get("zone_label", "unknown")
            return {
                "type": "FALL",
                "summary": f"Worker {person.person_id} fell in zone {zone} at {event_time}",
                "where": zone,
                "who": f"worker_{person.person_id}",
                "confidence": "HIGH",
                "action": "CHECK WORKER NOW",
            }

    for alert in frame_data.alerts:
        if alert.get("debounced", False):
            continue

        if alert.get("severity") == "Critical" or alert.get("type") == "RESTRICTED_ENTRY":
            person_id = alert.get("person_id", "unknown")
            zone_id = alert.get("zone_id", "unknown")
            return {
                "type": str(alert.get("type", "CRITICAL")).replace("_", " "),
                "summary": f"Worker {person_id} critical alert in zone {zone_id} at {event_time}",
                "where": zone_id,
                "who": f"worker_{person_id}",
                "confidence": "HIGH",
                "action": "DISPATCH ROVER",
            }

    return None

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
        color = (0, 200, 0)

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
            if person.has_helmet is not True:
                viols.append("NO HELMET")
            if person.has_glasses is not True:
                viols.append("NO GLASSES")
            if has_yellow_vest is False:
                viols.append("NO VEST")

            if viols:
                messages.extend(viols)
                is_safe = False
                color = (0, 140, 255)
                danger_area_count += 1
        elif zone_id == "safe":
            pass

        if is_safe:
            messages.append("SAFE")

        if is_safe:
            draw_corner_brackets(frame, xmin, ymin, xmax, ymax, color, thickness=2, length=15)
        else:
            cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), color, -1)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

        y_offset = ymin - 8
        for msg in reversed(messages):
            font_scale = 0.35
            thickness = 1
            (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            y_top = y_offset - th - 4
            cv2.rectangle(
                frame if not is_safe else overlay,
                (xmin, y_top),
                (xmin + tw + 6, y_offset + 2),
                color if not is_safe else (0, 0, 0),
                -1,
            )

            text_color = (255, 255, 255)
            post_blend_text.append(
                (msg, (xmin + 3, y_offset - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
            )

            y_offset -= (th + 8)

        msg_id = worker_label(person.person_id)
        (tw, th), _ = cv2.getTextSize(msg_id, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        y_top = y_offset - th - 4
        cv2.rectangle(
            frame if not is_safe else overlay,
            (xmin, y_top),
            (xmin + tw + 6, y_offset + 2),
            (0, 0, 0),
            -1,
        )
        post_blend_text.append(
            (msg_id, (xmin + 3, y_offset - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        )

    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    hud_overlay = frame.copy()

    if danger_area_count > 0:
        warning_text = f"SAFETY ALERT: {danger_area_count} WORKER(S) IN DANGER"
        (dw, dh), _ = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(hud_overlay, (10, 54), (min(w - 10, 10 + dw + 24), 54 + dh + 20), (0, 0, 255), -1)
        post_blend_text.append(
            (warning_text, (22, 54 + dh + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        )

    stats_str = f"LIVE: {frame_data.current_people_count}   TOTAL: {frame_data.total_unique_people}"
    (sw, sh), _ = cv2.getTextSize(stats_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(hud_overlay, (10, 10), (10 + sw + 20, 10 + sh + 16), (0, 0, 0), -1)
    post_blend_text.append((stats_str, (20, 10 + sh + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1))

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

    for text, pos, font, scale, color, thick in post_blend_text:
        cv2.putText(frame, text, pos, font, scale, color, thick)

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
    replay = SupervisorReplayRecorder()

    print("\n" + "=" * 50)
    print("MTU Pipeline Engine Active.")
    print("Press 'w' to cycle zone layout, 'q' to quit.")
    print("=" * 50 + "\n")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1800, 1350)

    try:
        for frame_data in stream:
            render_annotations(frame_data)

            event = build_critical_event(frame_data)
            if event is not None:
                replay.trigger(frame_data.timestamp, event)

            replay.draw_overlay(frame_data.processed_frame, frame_data.timestamp)
            replay.add_frame(frame_data.timestamp, frame_data.processed_frame)

            if frame_data.alerts:
                for alert in frame_data.alerts:
                    if not alert.get("debounced", False):
                        print(
                            f"[{time.strftime('%H:%M:%S', time.localtime(alert['timestamp']))}] "
                            f"[{alert['severity']}] {alert['message']}"
                        )

            display_frame = cv2.resize(
                frame_data.processed_frame,
                None,
                fx=2.0,
                fy=2.0,
                interpolation=cv2.INTER_LINEAR,
            )
            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\nShutdown command received. Closing stream.")
                break
            if key == ord("w"):
                engine.cycle_zone_layout()

    except KeyboardInterrupt:
        print("\nShutdown via keyboard interrupt.")
    finally:
        engine.release()
        cv2.destroyAllWindows()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
