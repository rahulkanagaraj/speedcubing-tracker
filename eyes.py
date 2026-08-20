import cv2
# pyrefly: ignore [missing-import]
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json
import time
import sys

# Define standard hand joint connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (9, 10), (10, 11), (11, 12),           # Middle
    (13, 14), (14, 15), (15, 16),          # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17)              # Palm/Knuckles
]

def draw_landmarks(image, landmarks, label):
    """Draw hand landmarks and connections on the image frame using OpenCV."""
    h, w, _ = image.shape
    
    # Draw skeletal connection lines
    for start_idx, end_idx in HAND_CONNECTIONS:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start_lm = landmarks[start_idx]
            end_lm = landmarks[end_idx]
            start_pt = (int(start_lm.x * w), int(start_lm.y * h))
            end_pt = (int(end_lm.x * w), int(end_lm.y * h))
            cv2.line(image, start_pt, end_pt, (230, 230, 230), 2)
            
    # Draw joint points (colored based on Handedness: Left=Green, Right=Yellow)
    color = (0, 255, 0) if label == 'Left' else (0, 255, 255)
    for lm in landmarks:
        pt = (int(lm.x * w), int(lm.y * h))
        cv2.circle(image, pt, 5, color, -1)
        
    # Overlay label near the wrist (landmark 0)
    if landmarks:
        wrist_pt = (int(landmarks[0].x * w), int(landmarks[0].y * h) - 10)
        cv2.putText(image, label, wrist_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def main():
    # Initialize the Hand Landmarker Tasks API
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
    except Exception as e:
        print(f"Error initializing MediaPipe HandLandmarker: {e}", file=sys.stderr)
        return
        
    # Initialize webcam capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam capture.", file=sys.stderr)
        detector.close()
        return
        
    json_path = "hand_events.json"
    
    # Open the JSON stream file
    try:
        f = open(json_path, 'w')
        f.write('[\n')
        f.flush()
    except IOError as e:
        print(f"Error: Could not open {json_path} for writing: {e}", file=sys.stderr)
        cap.release()
        detector.close()
        return

    first_entry = True
    frame_id = 0
    start_time = time.perf_counter()
    
    print("EYES: Hand tracking module started.")
    print("Press 'q' in the camera window to exit.")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to grab frame from camera.", file=sys.stderr)
                break
                
            frame_id += 1
            current_time_ms = int((time.perf_counter() - start_time) * 1000)
            
            # Convert BGR OpenCV image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert NumPy array to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Process hand detection (running synchronously for the frame timestamp)
            result = detector.detect_for_video(mp_image, current_time_ms)
            
            left_hand_data = None
            right_hand_data = None
            left_hand_score = -1.0
            right_hand_score = -1.0
            
            if result.hand_landmarks and result.handedness:
                for hand_landmarks, handedness_info in zip(result.hand_landmarks, result.handedness):
                    if not handedness_info:
                        continue
                    
                    category = handedness_info[0]
                    label = category.category_name  # 'Left' or 'Right'
                    score = category.score
                    
                    # Form coordinate array for the 21 points
                    landmarks_list = [[lm.x, lm.y, lm.z] for lm in hand_landmarks]
                    
                    # Distinguish hands using confidence scores
                    if label == 'Left':
                        if score > left_hand_score:
                            left_hand_data = landmarks_list
                            left_hand_score = score
                    elif label == 'Right':
                        if score > right_hand_score:
                            right_hand_data = landmarks_list
                            right_hand_score = score
                            
                    # Draw landmarks on the unflipped frame
                    draw_landmarks(frame, hand_landmarks, label)
            
            # Prepare JSON entry
            entry = {
                "frame_id": frame_id,
                "timestamp": current_time_ms,
                "left_hand": left_hand_data,
                "right_hand": right_hand_data
            }
            
            # Format and stream the entry to file (memory-efficient)
            entry_str = json.dumps(entry, indent=2)
            indented_entry = "\n".join("  " + line for line in entry_str.splitlines())
            
            if not first_entry:
                f.write(",\n")
            else:
                first_entry = False
                
            f.write(indented_entry)
            f.flush()
            
            # Flip visually for mirror view feedback
            display_frame = cv2.flip(frame, 1)
            cv2.imshow("CubeMind AI - EYES Module", display_frame)
            
            # Exit check
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("EYES: Exit key pressed.")
                break
                
    except KeyboardInterrupt:
        print("\nEYES: Process interrupted by user.", file=sys.stderr)
    finally:
        print("EYES: Releasing resources...")
        try:
            f.write('\n]\n')
            f.close()
        except Exception as e:
            print(f"Error finalizing JSON file: {e}", file=sys.stderr)
            
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print("EYES: Shutdown complete.")

if __name__ == '__main__':
    main()
