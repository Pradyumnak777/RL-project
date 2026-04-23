import torch
import cv2
import numpy as np
from collections import deque
from data_utils import pointStateProducer
from model import DQN
import os
from func_utils import get_optical_flow, get_dino_sscores, precompute_fb_errors, precompute_neighborhood_errors

# --- SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_vid_path = "UCF_Rep/val/v_JumpingJack_g24_c03.mp4" # Put your test video here
output_vid_path = "tracking_result.mp4"

# 1. Load the Trained Brain
print("Loading trained model...")
policy_net = DQN(state_dim=3, action_dim=3).to(device)
policy_net.load_state_dict(torch.load("point_tracker_dqn.pth"))
policy_net.eval()

# 2. Initialize the Environment
print(f"Processing Test Video: {test_vid_path}")

# --- AUTO-PRECOMPUTE BLOCK ---
vid_name = os.path.basename(test_vid_path) # e.g., "v_BenchPress_g21_c01.mp4"
npz_name = vid_name.replace('.mp4', '.npz')
save_dir = "precomputed_data"
save_path = os.path.join(save_dir, npz_name)

# Ensure the directory exists
os.makedirs(save_dir, exist_ok=True)

# If we haven't processed this test video yet, do it now on the fly
if not os.path.exists(save_path):
    print(f"Precomputed data not found for {vid_name}. Running DINO & Flow now (this will take a minute)...")
    
    # 1. Optical Flow
    all_flows, start_points = get_optical_flow(test_vid_path)
    # 2. DINO Features
    all_descriptors = get_dino_sscores(test_vid_path, all_flows, start_points)
    # 3. Errors
    fb_errors = precompute_fb_errors(test_vid_path, all_flows, start_points)
    nb_errors = precompute_neighborhood_errors(all_flows, start_points, radius=50)
    
    # Save it so we never have to compute it again for this video
    np.savez_compressed(
        save_path,
        flows=all_flows,
        start_points=start_points, # ADD THIS LINE
        descriptors=all_descriptors,
        fb_errors=fb_errors,
        nb_errors=nb_errors
    )
    print("Precomputation complete!")

# Initialize the producer using just the filename, not the full path
producer = pointStateProducer(vid_name) 
active_points = np.ones(producer.num_points, dtype=bool)
# --- NEW: Animation Timers ---
fade_frames = 5
flash_frames = 3
death_timers = np.zeros(producer.num_points, dtype=int)
flash_timers = np.zeros(producer.num_points, dtype=int)

# 3. Setup Video Writer for Visualization
cap = cv2.VideoCapture(test_vid_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_vid_path, fourcc, fps, (width, height))

curr_positions = producer.start_points.reshape(-1, 2).copy()
trail_len = 6
point_trails = [deque(maxlen=trail_len) for _ in range(producer.num_points)]

for p_idx in range(producer.num_points):
    x, y = curr_positions[p_idx]
    if not np.isnan(x) and not np.isnan(y):
        point_trails[p_idx].append((float(x), float(y)))

ret, frame = cap.read()
if ret:
    out.write(frame)

print("Running Inference...")
for t in range(producer.num_frames - 1):
    ret, frame = cap.read()
    if not ret:
        break
        
    curr_positions = curr_positions + np.nan_to_num(producer.all_flows[t])

    # Let the Agent make decisions for every point
    for p_idx in range(producer.num_points):
        if not active_points[p_idx]:
            continue
            
        state = producer.get_state(p_idx, t)
        
        # If OpenCV lost it, it dies (Treat it like a kill)
        if state is None:
            active_points[p_idx] = False
            death_timers[p_idx] = fade_frames
            continue
            
        # THE AGENT ACTS
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy_net(state_t)
            action = torch.argmax(q_values).item()
            
        # Apply the action
        if action == 0: # KILL
            active_points[p_idx] = False
            death_timers[p_idx] = fade_frames # Trigger the fade animation
        elif action == 2: # RE-ANCHOR
            producer.update_anchor(p_idx, t)
            flash_timers[p_idx] = flash_frames # Trigger the pop/flash animation

    # --- VISUALIZATION ---
    old_color = np.array([30, 60, 20], dtype=np.float32)
    new_color = np.array([90, 255, 50], dtype=np.float32)

    for p_idx in range(producer.num_points):
        x, y = curr_positions[p_idx]
        if np.isnan(x) or np.isnan(y):
            continue

        # 1. DYING POINTS (Kill Action)
        if not active_points[p_idx] and death_timers[p_idx] > 0:
            death_timers[p_idx] -= 1
            
            # Shrink the point and turn it Red (BGR: 0, 0, intensity)
            radius = max(1, death_timers[p_idx])
            intensity = int(255 * (death_timers[p_idx] / fade_frames))
            cv2.circle(frame, (int(x), int(y)), radius, (0, 0, intensity), -1, lineType=cv2.LINE_AA)
            
            if death_timers[p_idx] == 0:
                point_trails[p_idx].clear() # Erase from memory once fully faded
            continue

        # 2. ACTIVE POINTS (Keep Action)
        if active_points[p_idx]:
            point_trails[p_idx].append((float(x), float(y)))
            trail = point_trails[p_idx]

            # Draw standard trail
            for age_idx, (tx, ty) in enumerate(trail):
                if np.isnan(tx) or np.isnan(ty):
                    continue

                ratio = (age_idx + 1) / len(trail)
                color = old_color + ratio * (new_color - old_color)
                bgr = (int(color[0]), int(color[1]), int(color[2]))
                cv2.circle(frame, (int(tx), int(ty)), 1, bgr, -1, lineType=cv2.LINE_AA)
            
            # 3. RE-ANCHORING FLASH (Action 2)
            if flash_timers[p_idx] > 0:
                flash_timers[p_idx] -= 1
                # Draw a bright Cyan expanding ring and white core to "pop" it back in
                radius = 5 - flash_timers[p_idx] # Expands from 2 to 4
                cv2.circle(frame, (int(x), int(y)), radius, (255, 255, 0), 1, lineType=cv2.LINE_AA) # Cyan Halo
                cv2.circle(frame, (int(x), int(y)), 2, (255, 255, 255), -1, lineType=cv2.LINE_AA) # White Core
                
    out.write(frame)

cap.release()
out.release()
print(f"Testing complete! Watch the results in: {output_vid_path}")