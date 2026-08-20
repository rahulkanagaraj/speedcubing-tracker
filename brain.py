import json
import numpy as np
import pandas as pd
import sys
import os

# Configuration parameters
VELOCITY_THRESHOLD = 0.1         # Coordinate units per second
MIN_PAUSE_DURATION_SECONDS = 0.25  # Filter out pauses shorter than 250 ms
VELOCITY_SMOOTHING_WINDOW = 5      # Rolling window size to smooth frame-to-frame noise
CUBE_BBOX = {
    'x_min': 0.35,
    'x_max': 0.65,
    'y_min': 0.35,
    'y_max': 0.65
}

def get_hand_centroid(hand_landmarks):
    """Calculate the 3D centroid (mean x, y, z) of the 21 hand landmarks."""
    if hand_landmarks is None or not isinstance(hand_landmarks, list) or len(hand_landmarks) == 0:
        return None
    arr = np.array(hand_landmarks)
    return np.mean(arr, axis=0)  # Returns shape (3,) representing [x, y, z]

def count_regrips(hand_inside_cube_series):
    """
    Count regrips for a hand.
    A regrip is defined when a hand's status transitions from 'inside' to 'outside' and then back to 'inside'.
    """
    regrips = 0
    has_been_inside = False
    current_state = 'outside'
    
    for inside in hand_inside_cube_series:
        state = 'inside' if inside else 'outside'
        if state == 'inside':
            if current_state == 'outside' and has_been_inside:
                regrips += 1
            has_been_inside = True
            current_state = 'inside'
        elif state == 'outside':
            current_state = 'outside'
            
    return regrips

def main():
    json_input_path = "hand_events.json"
    json_output_path = "results.json"
    
    if not os.path.exists(json_input_path):
        print(f"Error: {json_input_path} not found. Please run dummy generation first.", file=sys.stderr)
        sys.exit(1)
        
    try:
        with open(json_input_path, 'r') as f:
            content = f.read()
    except IOError as e:
        print(f"Error reading {json_input_path}: {e}", file=sys.stderr)
        sys.exit(1)
        
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        last_brace = content.rfind('}')
        if last_brace != -1:
            repaired_content = content[:last_brace+1] + '\n]'
            try:
                data = json.loads(repaired_content)
                print(f"Warning: {json_input_path} was not closed cleanly. Auto-repaired and loaded {len(data)} frames.", file=sys.stderr)
            except json.JSONDecodeError as e:
                print(f"Error parsing repaired JSON: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"Error: {json_input_path} contains no complete frames.", file=sys.stderr)
            sys.exit(1)
        
    # Convert list of dicts to Pandas DataFrame
    df = pd.DataFrame(data)
    
    # Ensure correct types
    df['frame_id'] = df['frame_id'].astype(int)
    df['timestamp'] = df['timestamp'].astype(float)
    
    # Calculate dt (time difference between consecutive frames in seconds)
    df['dt'] = df['timestamp'].diff() / 1000.0
    df.loc[0, 'dt'] = 0.0  # First frame has no previous frame
    
    # Extract centroids using NumPy
    left_centroids = [get_hand_centroid(h) for h in df['left_hand']]
    right_centroids = [get_hand_centroid(h) for h in df['right_hand']]
    
    # Calculate frame-to-frame movement distances using NumPy
    left_distances = np.zeros(len(df))
    right_distances = np.zeros(len(df))
    
    for i in range(1, len(df)):
        # Left hand distance
        c_l1 = left_centroids[i-1]
        c_l2 = left_centroids[i]
        if c_l1 is not None and c_l2 is not None:
            left_distances[i] = np.linalg.norm(c_l2 - c_l1)
            
        # Right hand distance
        c_r1 = right_centroids[i-1]
        c_r2 = right_centroids[i]
        if c_r1 is not None and c_r2 is not None:
            right_distances[i] = np.linalg.norm(c_r2 - c_r1)
            
    df['left_distance'] = left_distances
    df['right_distance'] = right_distances
    
    # Calculate velocities (distance / dt)
    df['left_velocity'] = np.where(df['dt'] > 0, df['left_distance'] / df['dt'], 0.0)
    df['right_velocity'] = np.where(df['dt'] > 0, df['right_distance'] / df['dt'], 0.0)
    
    # Smooth velocities using rolling average to filter out high-frequency noise
    df['left_velocity_smooth'] = df['left_velocity'].rolling(VELOCITY_SMOOTHING_WINDOW, min_periods=1).mean()
    df['right_velocity_smooth'] = df['right_velocity'].rolling(VELOCITY_SMOOTHING_WINDOW, min_periods=1).mean()
    
    # Combined velocity represents the maximum active speed of either hand in the frame
    df['combined_velocity'] = np.maximum(df['left_velocity_smooth'], df['right_velocity_smooth'])
    
    # Initial classification: 'paused' or 'turning'
    df['state'] = np.where(df['combined_velocity'] < VELOCITY_THRESHOLD, 'paused', 'turning')
    
    # Duration-based filtering: Group contiguous identical states to filter out short momentary pauses
    df['block_id'] = (df['state'] != df['state'].shift()).cumsum()
    block_durations = df.groupby('block_id')['dt'].transform('sum')
    
    # Re-classify short pauses as 'turning'
    df.loc[(df['state'] == 'paused') & (block_durations < MIN_PAUSE_DURATION_SECONDS), 'state'] = 'turning'
    
    # --- METRICS CALCULATIONS ---
    
    # 1. Total solve time (based on timestamps in seconds)
    total_solve_time = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]) / 1000.0
    
    # 2. Pause count and total pause duration
    # Sum of dt for all paused frames gives total duration
    total_pause_duration = df.loc[df['state'] == 'paused', 'dt'].sum()
    # A pause event starts when state shifts from turning (or start of solve) to paused
    df['pause_start'] = (df['state'] == 'paused') & (df['state'].shift(1) != 'paused')
    pause_count = int(df['pause_start'].sum())
    
    # 3. Regrip count (hands leaving the bounding box of the cube and re-entering)
    left_inside = []
    right_inside = []
    
    for c in left_centroids:
        if c is not None:
            inside = (CUBE_BBOX['x_min'] <= c[0] <= CUBE_BBOX['x_max']) and \
                     (CUBE_BBOX['y_min'] <= c[1] <= CUBE_BBOX['y_max'])
            left_inside.append(inside)
        else:
            left_inside.append(False)
            
    for c in right_centroids:
        if c is not None:
            inside = (CUBE_BBOX['x_min'] <= c[0] <= CUBE_BBOX['x_max']) and \
                     (CUBE_BBOX['y_min'] <= c[1] <= CUBE_BBOX['y_max'])
            right_inside.append(inside)
        else:
            right_inside.append(False)
            
    left_regrips = count_regrips(left_inside)
    right_regrips = count_regrips(right_inside)
    total_regrips = left_regrips + right_regrips
    
    # 4. Left vs. Right hand usage percentage
    left_cum_dist = df['left_distance'].sum()
    right_cum_dist = df['right_distance'].sum()
    total_cum_dist = left_cum_dist + right_cum_dist
    
    if total_cum_dist > 0:
        left_usage_pct = (left_cum_dist / total_cum_dist) * 100
        right_usage_pct = (right_cum_dist / total_cum_dist) * 100
    else:
        left_usage_pct = 0.0
        right_usage_pct = 0.0
        
    # Group contiguous identical states to create the rhythm timeline
    timeline = []
    for name, group in df.groupby('block_id'):
        state = group['state'].iloc[0]
        duration = group['dt'].sum()
        timeline.append({
            "state": state,
            "duration_seconds": float(round(duration, 3))
        })

    # Compile results
    results = {
        "total_solve_time_seconds": float(round(total_solve_time, 3)),
        "pause_count": pause_count,
        "total_pause_duration_seconds": float(round(total_pause_duration, 3)),
        "regrip_count": {
            "left_hand": left_regrips,
            "right_hand": right_regrips,
            "total": total_regrips
        },
        "hand_usage_percentage": {
            "left_hand": float(round(left_usage_pct, 2)),
            "right_hand": float(round(right_usage_pct, 2))
        },
        "rhythm_timeline": timeline,
        "settings_used": {
            "velocity_threshold": VELOCITY_THRESHOLD,
            "minimum_pause_duration_seconds": MIN_PAUSE_DURATION_SECONDS,
            "velocity_smoothing_window": VELOCITY_SMOOTHING_WINDOW,
            "cube_bbox": CUBE_BBOX
        }
    }
    
    # Export to results.json and public/assets/results.json
    paths_to_write = [json_output_path, os.path.join("public", "assets", "results.json")]
    
    for path in paths_to_write:
        try:
            # Ensure directories exist
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Analysis complete. Metrics exported to {path}")
        except IOError as e:
            print(f"Error writing to {path}: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == '__main__':
    main()
