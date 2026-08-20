import cv2
# pyrefly: ignore [missing-import]
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json
import time
import sys
import os
import threading
import http.server
import socketserver
import numpy as np

PORT = 5173

# Shared cross-thread global state
frame_lock = threading.Lock()
latest_jpg = None
current_state = "WAITING"
solve_time = 0.0
model_loaded = False

def count_regrips(inside_list):
    regrips = 0
    has_been_inside = False
    current_st = 'outside'
    for inside in inside_list:
        st = 'inside' if inside else 'outside'
        if st == 'inside':
            if current_st == 'outside' and has_been_inside:
                regrips += 1
            has_been_inside = True
            current_st = 'inside'
        else:
            current_st = 'outside'
    return regrips

def process_and_save_solve(frames, start_timestamp, stop_timestamp):
    if len(frames) < 5:
        return
    
    total_solve_time = stop_timestamp - start_timestamp
    
    # 1. Log raw events to hand_events.json
    try:
        with open('hand_events.json', 'w') as f:
            json.dump(frames, f, indent=2)
    except Exception as e:
        print(f"Error saving hand_events.json: {e}", file=sys.stderr)

    # 2. Extract telemetry coordinates
    n_frames = len(frames)
    dt_array = [0.0]
    left_centroids = [None] * n_frames
    right_centroids = [None] * n_frames
    
    for i in range(n_frames):
        if i > 0:
            dt_array.append((frames[i]["timestamp"] - frames[i-1]["timestamp"]) / 1000.0)
        
        lh = frames[i]["left_hand"]
        rh = frames[i]["right_hand"]
        
        if lh:
            xs = [pt[0] for pt in lh]
            ys = [pt[1] for pt in lh]
            zs = [pt[2] for pt in lh]
            left_centroids[i] = [sum(xs)/21, sum(ys)/21, sum(zs)/21]
            
        if rh:
            xs = [pt[0] for pt in rh]
            ys = [pt[1] for pt in rh]
            zs = [pt[2] for pt in rh]
            right_centroids[i] = [sum(xs)/21, sum(ys)/21, sum(zs)/21]

    # Calculate frame displacement distance
    left_distances = [0.0] * n_frames
    right_distances = [0.0] * n_frames
    
    for i in range(1, n_frames):
        c1_l, c2_l = left_centroids[i-1], left_centroids[i]
        if c1_l and c2_l:
            left_distances[i] = float(np.linalg.norm(np.array(c1_l) - np.array(c2_l)))
            
        c1_r, c2_r = right_centroids[i-1], right_centroids[i]
        if c1_r and c2_r:
            right_distances[i] = float(np.linalg.norm(np.array(c1_r) - np.array(c2_r)))

    # Calculate velocities
    left_vel = [0.0] * n_frames
    right_vel = [0.0] * n_frames
    for i in range(1, n_frames):
        dt = dt_array[i]
        if dt > 0:
            left_vel[i] = left_distances[i] / dt
            right_vel[i] = right_distances[i] / dt

    # Rolling average window 5 smoothing
    def rolling_mean(arr, window=5):
        res = []
        for i in range(len(arr)):
            start = max(0, i - window + 1)
            slice_arr = arr[start:i+1]
            res.append(sum(slice_arr) / len(slice_arr))
        return res

    left_vel_smooth = rolling_mean(left_vel)
    right_vel_smooth = rolling_mean(right_vel)

    # Classify raw states
    raw_states = []
    block_ids = []
    block_id = 0
    prev_state = ""
    for i in range(n_frames):
        comb_vel = max(left_vel_smooth[i], right_vel_smooth[i])
        state = 'paused' if comb_vel < 0.1 else 'turning'
        raw_states.append(state)
        if state != prev_state:
            block_id += 1
            prev_state = state
        block_ids.append(block_id)

    # Group block durations
    block_durs = {}
    for i in range(n_frames):
        bid = block_ids[i]
        block_durs[bid] = block_durs.get(bid, 0.0) + dt_array[i]

    # Convert small pauses < 250ms to turning
    filtered_states = []
    for i in range(n_frames):
        bid = block_ids[i]
        state = raw_states[i]
        if state == 'paused' and block_durs[bid] < 0.25:
            filtered_states.append('turning')
        else:
            filtered_states.append(state)

    # Re-group
    final_block_ids = []
    final_block_id = 0
    prev_final_state = ""
    for i in range(n_frames):
        state = filtered_states[i]
        if state != prev_final_state:
            final_block_id += 1
            prev_final_state = state
        final_block_ids.append(final_block_id)

    final_block_durs = {}
    final_block_states = {}
    for i in range(n_frames):
        bid = final_block_ids[i]
        final_block_durs[bid] = final_block_durs.get(bid, 0.0) + dt_array[i]
        final_block_states[bid] = filtered_states[i]

    pause_count = 0
    total_pause_duration = 0.0
    timeline = []

    for bid in sorted(final_block_durs.keys()):
        state = final_block_states[bid]
        dur = final_block_durs[bid]
        
        timeline.append({
            "state": state,
            "duration_seconds": float(round(dur, 3))
        })
        
        if state == 'paused':
            pause_count += 1
            total_pause_duration += dur

    # Calculate regrips
    cube_bbox = { "x_min": 0.35, "x_max": 0.65, "y_min": 0.35, "y_max": 0.65 }
    def is_inside(c):
        if not c:
            return False
        return (cube_bbox["x_min"] <= c[0] <= cube_bbox["x_max"] and
                cube_bbox["y_min"] <= c[1] <= cube_bbox["y_max"])

    left_inside = [is_inside(c) for c in left_centroids]
    right_inside = [is_inside(c) for c in right_centroids]

    left_regrips = count_regrips(left_inside)
    right_regrips = count_regrips(right_inside)

    # Left vs. Right hand usage shares
    left_cum_dist = sum(left_distances)
    right_cum_dist = sum(right_distances)
    total_cum_dist = left_cum_dist + right_cum_dist
    left_usage = (left_cum_dist / total_cum_dist) * 100 if total_cum_dist > 0 else 0.0
    right_usage = (right_cum_dist / total_cum_dist) * 100 if total_cum_dist > 0 else 0.0

    # Build results JSON
    results = {
        "total_solve_time_seconds": float(round(total_solve_time, 3)),
        "pause_count": pause_count,
        "total_pause_duration_seconds": float(round(total_pause_duration, 3)),
        "regrip_count": {
            "left_hand": left_regrips,
            "right_hand": right_regrips,
            "total": left_regrips + right_regrips
        },
        "hand_usage_percentage": {
            "left_hand": float(round(left_usage, 2)),
            "right_hand": float(round(right_usage, 2))
        },
        "rhythm_timeline": timeline,
        "settings_used": {
            "velocity_threshold": 0.1,
            "minimum_pause_duration_seconds": 0.25,
            "velocity_smoothing_window": 5,
            "cube_bbox": cube_bbox
        }
    }

    # Save results.json to root and public/assets
    for path in ['results.json', os.path.join('public', 'assets', 'results.json')]:
        try:
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Server API: Compiled results successfully saved to {path}")
        except Exception as e:
            print(f"Error saving compiled results to {path}: {e}", file=sys.stderr)

    # Save session state durations
    session_state = {
        "start_timestamp": int(start_timestamp * 1000),
        "stop_timestamp": int(stop_timestamp * 1000),
        "solve_time_seconds": float(round(total_solve_time, 3))
    }
    try:
        with open('session_state.json', 'w') as f:
            json.dump(session_state, f, indent=2)
        print("Server API: Compiled session_state saved successfully.")
    except Exception as e:
        print(f"Error saving session state: {e}", file=sys.stderr)

def camera_thread_func():
    global latest_jpg, current_state, solve_time, model_loaded
    
    # Initialize the Hand Landmarker Options
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    try:
        detector = vision.HandLandmarker.create_from_options(options)
        model_loaded = True
        print("Backend: MediaPipe HandLandmarker Initialized Successfully.")
    except Exception as e:
        print(f"Backend Error: Failed to initialize landmarker: {e}", file=sys.stderr)
        return

    # Open local webcam channel
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Backend Error: Could not open camera source.", file=sys.stderr)
        detector.close()
        return

    # Set frame sizing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # State Machine variables
    state = "WAITING"
    start_time = 0
    start_perf = 0
    countdown_start = 0
    stop_countdown_start = 0
    solve_frames = []

    # Virtual Stackmat pad zones (normalized x coordinates on mirrored display)
    pad_config = {
        "y_min": 0.65,
        "y_max": 0.90,
        "left_x_min": 0.12,
        "left_x_max": 0.38,
        "right_x_min": 0.62,
        "right_x_max": 0.88
    }

    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (9, 10), (10, 11), (11, 12),
        (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17)
    ]

    def get_centroid(landmarks):
        if not landmarks:
            return None
        x = sum(lm.x for lm in landmarks)
        y = sum(lm.y for lm in landmarks)
        z = sum(lm.z for lm in landmarks)
        return [x / 21, y / 21, z / 21]

    try:
        loop_start_time = time.perf_counter()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            h, w, _ = frame.shape
            current_time_ms = int((time.perf_counter() - loop_start_time) * 1000)

            # Convert frame color space from BGR to RGB (Task 2)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Run MediaPipe hand tracking (Task 2)
            result = detector.detect_for_video(mp_image, current_time_ms)

            left_hand_found = None
            right_hand_found = None
            left_centroid = None
            right_centroid = None

            if result.hand_landmarks and result.handedness:
                for hand_landmarks, handedness_info in zip(result.hand_landmarks, result.handedness):
                    if not handedness_info:
                        continue
                    label = handedness_info[0].category_name  # 'Left' or 'Right'
                    
                    if label == 'Left':
                        left_hand_found = hand_landmarks
                        left_centroid = get_centroid(hand_landmarks)
                    elif label == 'Right':
                        right_hand_found = hand_landmarks
                        right_centroid = get_centroid(hand_landmarks)

                    # Draw landmarks on raw camera feed before mirroring
                    color = (0, 255, 0) if label == 'Left' else (0, 255, 255)
                    # Skeleton lines
                    for start_idx, end_idx in HAND_CONNECTIONS:
                        if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                            p1 = hand_landmarks[start_idx]
                            p2 = hand_landmarks[end_idx]
                            cv2.line(frame, (int(p1.x * w), int(p1.y * h)), (int(p2.x * w), int(p2.y * h)), (230, 230, 230), 2)
                    # Joints
                    for lm in hand_landmarks:
                        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, color, -1)

            # Flip visually to mirror view for natural user alignment
            mirrored_frame = cv2.flip(frame, 1)

            # Calculate presence on the touch pads in mirrored coordinates
            is_left_present = False
            if left_centroid:
                mirrored_cx = 1 - left_centroid[0]
                is_left_present = (pad_config["left_x_min"] <= mirrored_cx <= pad_config["left_x_max"]) and \
                                  (pad_config["y_min"] <= left_centroid[1] <= pad_config["y_max"])

            is_right_present = False
            if right_centroid:
                mirrored_cx = 1 - right_centroid[0]
                is_right_present = (pad_config["right_x_min"] <= mirrored_cx <= pad_config["right_x_max"]) and \
                                   (pad_config["y_min"] <= right_centroid[1] <= pad_config["y_max"])

            # Render Mat overlays directly on display stream
            def draw_pad_overlay(img, x_min, x_max, is_touch, label):
                if state in ["READY_COUNTDOWN", "STOP_COUNTDOWN"]:
                    color = (0, 165, 255)  # Yellow/Orange BGR
                elif state == "READY":
                    color = (0, 255, 0)    # Green
                elif state == "SOLVING":
                    color = (160, 160, 160) # Gray
                else: # WAITING
                    color = (0, 255, 0) if is_touch else (0, 0, 255) # Touch=Green, Open=Red

                # Make translucent overlay box
                overlay = img.copy()
                px1, px2 = int(x_min * w), int(x_max * w)
                py1, py2 = int(pad_config["y_min"] * h), int(pad_config["y_max"] * h)
                cv2.rectangle(overlay, (px1, py1), (px2, py2), color, -1)
                
                # Blend overlay
                cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
                cv2.rectangle(img, (px1, py1), (px2, py2), color, 2)
                cv2.putText(img, label, (px1 + 10, py1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            draw_pad_overlay(mirrored_frame, pad_config["left_x_min"], pad_config["left_x_max"], is_left_present, "LEFT PAD")
            draw_pad_overlay(mirrored_frame, pad_config["right_x_min"], pad_config["right_x_max"], is_right_present, "RIGHT PAD")

            # State transitions logic
            now_ms = int(time.perf_counter() * 1000)

            if state == "WAITING":
                if is_left_present and is_right_present:
                    state = "READY_COUNTDOWN"
                    countdown_start = now_ms
                solve_time = 0.0
            elif state == "READY_COUNTDOWN":
                if is_left_present and is_right_present:
                    if now_ms - countdown_start >= 1000:
                        state = "READY"
                else:
                    state = "WAITING"
                solve_time = 0.0
            elif state == "READY":
                if not is_left_present or not is_right_present:
                    state = "SOLVING"
                    start_time = time.time()
                    start_perf = time.perf_counter()
                    solve_frames = []
                solve_time = 0.0
            elif state == "SOLVING":
                solve_time = time.perf_counter() - start_perf
                
                # Record coordinates frame-by-frame
                solve_frames.append({
                    "frame_id": len(solve_frames) + 1,
                    "timestamp": int(time.time() * 1000),
                    "left_hand": [[lm.x, lm.y, lm.z] for lm in left_hand_found] if left_hand_found else None,
                    "right_hand": [[lm.x, lm.y, lm.z] for lm in right_hand_found] if right_hand_found else None
                })

                if is_left_present and is_right_present:
                    state = "STOP_COUNTDOWN"
                    stop_countdown_start = now_ms
            elif state == "STOP_COUNTDOWN":
                solve_time = time.perf_counter() - start_perf
                
                solve_frames.append({
                    "frame_id": len(solve_frames) + 1,
                    "timestamp": int(time.time() * 1000),
                    "left_hand": [[lm.x, lm.y, lm.z] for lm in left_hand_found] if left_hand_found else None,
                    "right_hand": [[lm.x, lm.y, lm.z] for lm in right_hand_found] if right_hand_found else None
                })

                if is_left_present and is_right_present:
                    if now_ms - stop_countdown_start >= 1000:
                        state = "STOPPED"
                        # Process and compile final analytics results
                        threading.Thread(target=process_and_save_solve, args=(solve_frames, start_time, time.time())).start()
                else:
                    state = "SOLVING"
            elif state == "STOPPED":
                if not is_left_present and not is_right_present:
                    state = "WAITING"

            current_state = state

            # Draw visual HUD HUD directly on display feed
            timer_text = f"TIME: {solve_time:.2f}s"
            state_text = f"STATE: {state.replace('_', ' ')}"
            cv2.putText(mirrored_frame, state_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(mirrored_frame, timer_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255) if state == "SOLVING" else (255, 255, 255), 2, cv2.LINE_AA)

            # Compress as JPG bytes
            ret_jpg, jpg_buffer = cv2.imencode('.jpg', mirrored_frame)
            if ret_jpg:
                with frame_lock:
                    latest_jpg = jpg_buffer.tobytes()

            time.sleep(0.01)

    finally:
        cap.release()
        detector.close()

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/video_feed':
            # Serve BGR/JPEG camera live-stream overlay (Task 1)
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    with frame_lock:
                        if latest_jpg is None:
                            time.sleep(0.01)
                            continue
                        jpg_bytes = latest_jpg
                    
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                    self.wfile.write(jpg_bytes)
                    self.wfile.write(b'\r\n')
                    time.sleep(0.033)  # ~30 FPS limit
            except Exception:
                pass
        elif self.path == '/api/state':
            # Return live dashboard state telemetry info
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            state_json = {
                "state": current_state,
                "timer": solve_time,
                "model_loaded": model_loaded
            }
            self.wfile.write(json.dumps(state_json).encode('utf-8'))
        else:
            # Fallback to serving static HTML file
            super().do_GET()

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Start the OpenCV camera thread
    threading.Thread(target=camera_thread_func, daemon=True).start()
    
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    
    try:
        with socketserver.ThreadingTCPServer(("", PORT), CustomHandler) as httpd:
            print(f"CubeMind AI: Dashboard Server running at http://localhost:{PORT}")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer shutting down.")
    except Exception as e:
        print(f"Error starting server: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
