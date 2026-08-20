import json
import numpy as np

def generate_hand_landmarks(base_x, base_y, base_z, noise_scale=0.0):
    """Generate 21 hand landmarks with relative positions centered around a base coordinate."""
    landmarks = []
    # Base offsets for a generic hand shape (21 landmarks)
    offsets = [
        [0.0, 0.0, 0.0],  # 0: Wrist
        [-0.05, 0.02, -0.01], [ -0.08, 0.05, -0.02], [-0.10, 0.07, -0.03], [-0.12, 0.09, -0.04],  # 1-4: Thumb
        [-0.03, 0.10, -0.01], [ -0.04, 0.15, -0.02], [-0.05, 0.18, -0.03], [-0.06, 0.20, -0.04],  # 5-8: Index
        [0.0, 0.11, -0.01],   [ 0.0, 0.17, -0.02],   [0.0, 0.21, -0.03],   [0.0, 0.24, -0.04],    # 9-12: Middle
        [0.03, 0.10, -0.01],  [ 0.04, 0.15, -0.02],  [0.05, 0.18, -0.03],  [0.06, 0.20, -0.04],   # 13-16: Ring
        [0.06, 0.08, -0.01],  [ 0.08, 0.12, -0.02],  [0.10, 0.15, -0.03],  [0.11, 0.17, -0.04]    # 17-20: Pinky
    ]
    
    for ox, oy, oz in offsets:
        x = base_x + ox + np.random.normal(0, noise_scale)
        y = base_y + oy + np.random.normal(0, noise_scale)
        z = base_z + oz + np.random.normal(0, noise_scale)
        # Keep inside bounds [0, 1]
        x = max(0.0, min(1.0, float(x)))
        y = max(0.0, min(1.0, float(y)))
        z = float(z)
        landmarks.append([x, y, z])
    return landmarks

def main():
    np.random.seed(42)
    frames = 300
    fps = 30
    dt_ms = int(1000 / fps)
    
    events = []
    timestamp = 0
    
    # Bounding box of the cube: [0.35, 0.65] for both x and y.
    
    for frame_id in range(1, frames + 1):
        timestamp += dt_ms + np.random.randint(-2, 3) # Add slight jitter to timestamp
        
        # --- LEFT HAND SIMULATION ---
        # Starts outside (0.2), moves inside (0.5), goes outside (0.2) for regrip, returns inside (0.5), exits.
        if frame_id <= 10:
            left_x = 0.2  # Outside
            left_active = True
        elif 10 < frame_id <= 100:
            # Active and inside
            # Generate sinusoidal movement for turning
            left_x = 0.45 + 0.05 * np.sin(frame_id * 0.1)
            left_active = True
            
            # Simulate a pause for both hands between frames 50 and 80
            if 50 <= frame_id <= 80:
                left_x = 0.47  # Constant (paused)
        elif 100 < frame_id <= 120:
            left_x = 0.2  # Outside the cube box (Regrip)
            left_active = True
        elif 120 < frame_id <= 250:
            # Back inside
            left_x = 0.45 + 0.05 * np.sin(frame_id * 0.1)
            left_active = True
        else:
            left_x = 0.1  # Exited completely at the end
            left_active = False
            
        # --- RIGHT HAND SIMULATION ---
        # Starts outside (0.8), moves inside (0.55), goes outside (0.8) for regrip, returns inside (0.55), exits.
        if frame_id <= 15:
            right_x = 0.8  # Outside
            right_active = True
        elif 15 < frame_id <= 150:
            # Active and inside
            right_x = 0.55 + 0.03 * np.cos(frame_id * 0.15)
            right_active = True
            
            # Simulate a pause for both hands between frames 50 and 80
            if 50 <= frame_id <= 80:
                right_x = 0.53  # Constant (paused)
        elif 150 < frame_id <= 170:
            right_x = 0.8  # Outside the cube box (Regrip)
            right_active = True
        elif 170 < frame_id <= 260:
            # Back inside
            right_x = 0.55 + 0.03 * np.cos(frame_id * 0.15)
            right_active = True
        else:
            right_x = 0.9  # Exited completely at the end
            right_active = False
            
        # Add noise to simulate hand trembling (lower noise during pause)
        left_noise = 0.0001 if (50 <= frame_id <= 80) else 0.005
        right_noise = 0.0001 if (50 <= frame_id <= 80) else 0.005
        
        left_hand = generate_hand_landmarks(left_x, 0.5, 0.0, left_noise) if left_active else None
        right_hand = generate_hand_landmarks(right_x, 0.5, 0.0, right_noise) if right_active else None
        
        events.append({
            "frame_id": frame_id,
            "timestamp": timestamp,
            "left_hand": left_hand,
            "right_hand": right_hand
        })
        
    # Write to hand_events.json
    with open("hand_events.json", "w") as f:
        json.dump(events, f, indent=2)
        
    print(f"Generated dummy hand_events.json with {frames} frames.")

if __name__ == "__main__":
    main()
