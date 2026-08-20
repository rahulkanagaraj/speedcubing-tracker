import cv2
import numpy as np
import json
import time
import sys
import os

# State constants
STATE_WAITING = "WAITING"
STATE_READY_COUNTDOWN = "READY_COUNTDOWN"
STATE_READY = "READY"
STATE_SOLVING = "SOLVING"
STATE_STOP_COUNTDOWN = "STOP_COUNTDOWN"
STATE_STOPPED = "STOPPED"

def main():
    # Initialize webcam capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam video feed.", file=sys.stderr)
        return
        
    # State machine variables
    current_state = STATE_WAITING
    start_timestamp = 0
    stop_timestamp = 0
    
    solve_start_perf = 0.0
    solve_time_elapsed = 0.0
    
    countdown_start_time = 0.0
    stop_countdown_start_time = 0.0
    
    # Motion detection history
    prev_left_gray = None
    prev_right_gray = None
    
    # Skin color HSV bounds (standard range under reasonable lighting)
    lower_skin = np.array([0, 15, 60], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    print("Session Controller: Initialized.")
    print("Press 'q' in the window to safely quit.")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to grab frame from webcam.", file=sys.stderr)
                break
                
            # Flip horizontally for a natural mirror view
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # --- Define Pad Regions of Interest (ROI) ---
            # Positioned in the bottom center of the frame
            pad_w, pad_h = 130, 90
            y1 = h - pad_h - 50
            y2 = h - 50
            
            left_x1 = w // 2 - pad_w - 40
            left_x2 = w // 2 - 40
            
            right_x1 = w // 2 + 40
            right_x2 = w // 2 + pad_w + 40
            
            # --- Hand Presence Detection (Skin Color Segmentation) ---
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Analyze Left Pad
            left_hsv_roi = hsv[y1:y2, left_x1:left_x2]
            left_skin_mask = cv2.inRange(left_hsv_roi, lower_skin, upper_skin)
            left_skin_ratio = np.sum(left_skin_mask == 255) / (pad_w * pad_h)
            left_present = left_skin_ratio > 0.15
            
            # Analyze Right Pad
            right_hsv_roi = hsv[y1:y2, right_x1:right_x2]
            right_skin_mask = cv2.inRange(right_hsv_roi, lower_skin, upper_skin)
            right_skin_ratio = np.sum(right_skin_mask == 255) / (pad_w * pad_h)
            right_present = right_skin_ratio > 0.15
            
            # --- Motion Detection (Frame Differencing) ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            left_gray_roi = gray[y1:y2, left_x1:left_x2]
            right_gray_roi = gray[y1:y2, right_x1:right_x2]
            
            if prev_left_gray is None:
                prev_left_gray = left_gray_roi.copy()
                prev_right_gray = right_gray_roi.copy()
                
            # Left pad motion
            left_diff = cv2.absdiff(prev_left_gray, left_gray_roi)
            _, left_thresh = cv2.threshold(left_diff, 15, 255, cv2.THRESH_BINARY)
            left_motion_ratio = np.sum(left_thresh == 255) / (pad_w * pad_h)
            left_steady = left_motion_ratio < 0.02
            
            # Right pad motion
            right_diff = cv2.absdiff(prev_right_gray, right_gray_roi)
            _, right_thresh = cv2.threshold(right_diff, 15, 255, cv2.THRESH_BINARY)
            right_motion_ratio = np.sum(right_thresh == 255) / (pad_w * pad_h)
            right_steady = right_motion_ratio < 0.02
            
            # Update history
            prev_left_gray = left_gray_roi.copy()
            prev_right_gray = right_gray_roi.copy()
            
            # --- State Machine Processing ---
            now = time.time()
            
            if current_state == STATE_WAITING:
                # Pads turn Red/Orange indicating waiting
                pad_color = (0, 0, 255)
                hud_text = "PLACE BOTH HANDS ON PADS"
                
                # Transition: Both hands placed on pads
                if left_present and right_present:
                    current_state = STATE_READY_COUNTDOWN
                    countdown_start_time = now
                    
            elif current_state == STATE_READY_COUNTDOWN:
                # Pads turn Yellow indicating countdown active
                pad_color = (0, 255, 255)
                hud_text = "KEEP STEADY..."
                
                # Check presence
                if left_present and right_present:
                    # Check steadiness
                    if left_steady and right_steady:
                        if now - countdown_start_time >= 1.0:
                            current_state = STATE_READY
                    else:
                        # Reset timer if there is too much movement
                        countdown_start_time = now
                else:
                    # Hands removed, go back to waiting
                    current_state = STATE_WAITING
                    
            elif current_state == STATE_READY:
                # Pads turn Green indicating ready to lift
                pad_color = (0, 255, 0)
                hud_text = "READY! LIFT HANDS TO START"
                
                # Transition: Lift hands off pads (solve starts)
                if not left_present or not right_present:
                    current_state = STATE_SOLVING
                    start_timestamp = int(time.time() * 1000)
                    solve_start_perf = time.perf_counter()
                    solve_time_elapsed = 0.0
                    
            elif current_state == STATE_SOLVING:
                # Pads are drawn Grey during solve
                pad_color = (150, 150, 150)
                solve_time_elapsed = time.perf_counter() - solve_start_perf
                hud_text = f"SOLVING..."
                
                # Transition: Return hands to pads
                if left_present and right_present:
                    current_state = STATE_STOP_COUNTDOWN
                    stop_countdown_start_time = now
                    
            elif current_state == STATE_STOP_COUNTDOWN:
                # Pads turn Yellow indicating stopping in progress
                pad_color = (0, 255, 255)
                solve_time_elapsed = time.perf_counter() - solve_start_perf
                hud_text = "STOPPING..."
                
                if left_present and right_present:
                    if left_steady and right_steady:
                        if now - stop_countdown_start_time >= 1.0:
                            current_state = STATE_STOPPED
                            stop_timestamp = int(time.time() * 1000)
                            
                            # Export timestamps to session_state.json
                            session_state = {
                                "start_timestamp": start_timestamp,
                                "stop_timestamp": stop_timestamp,
                                "solve_time_seconds": round(solve_time_elapsed - 1.0, 3) # Subtract the 1.0s stop delay
                            }
                            try:
                                with open("session_state.json", "w") as sf:
                                    json.dump(session_state, sf, indent=2)
                                print("Session Controller: Solve complete. Session state exported.")
                            except IOError as e:
                                print(f"Error saving session state: {e}", file=sys.stderr)
                    else:
                        # Reset stop countdown if hands are moving
                        stop_countdown_start_time = now
                else:
                    # Hands lifted off again, resume solving
                    current_state = STATE_SOLVING
                    
            elif current_state == STATE_STOPPED:
                # Pads turn Green to show final solved state
                pad_color = (0, 255, 0)
                hud_text = f"STOPPED - SOLVE COMPLETE"
                
                # Transition: Lift hands off pads to reset back to waiting
                if not left_present and not right_present:
                    current_state = STATE_WAITING
            
            # --- Visual HUD Overlay ---
            # Draw Pads
            cv2.rectangle(frame, (left_x1, y1), (left_x2, y2), pad_color, 2)
            cv2.rectangle(frame, (right_x1, y1), (right_x2, y2), pad_color, 2)
            
            # Fill pad backgrounds slightly for UI touch feel
            overlay = frame.copy()
            cv2.rectangle(overlay, (left_x1, y1), (left_x2, y2), pad_color, -1)
            cv2.rectangle(overlay, (right_x1, y1), (right_x2, y2), pad_color, -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            
            # Overlay hand presence indicators (dots)
            if left_present:
                cv2.circle(frame, ((left_x1 + left_x2) // 2, (y1 + y2) // 2), 10, (0, 255, 0), -1)
            if right_present:
                cv2.circle(frame, ((right_x1 + right_x2) // 2, (y1 + y2) // 2), 10, (0, 255, 0), -1)
                
            # Draw State Text Header
            cv2.putText(frame, hud_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Draw Live Solve Timer
            if current_state in [STATE_SOLVING, STATE_STOP_COUNTDOWN]:
                timer_str = f"TIME: {solve_time_elapsed:.2f}s"
                cv2.putText(frame, timer_str, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
            elif current_state == STATE_STOPPED:
                timer_str = f"FINAL TIME: {session_state['solve_time_seconds']:.2f}s"
                cv2.putText(frame, timer_str, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
                
            # Display frame
            cv2.imshow("CubeMind AI - Session Controller", frame)
            
            # Exit check ('q')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Session Controller: Exit signal received.")
                break
                
    except KeyboardInterrupt:
        print("\nSession Controller: Interrupted by user.", file=sys.stderr)
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("Session Controller: Shutdown complete.")

if __name__ == '__main__':
    main()
