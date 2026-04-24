import torch
import cv2
import numpy as np
import os
from data_utils import pointStateProducer
from model import DQN
from func_utils import get_optical_flow, get_dino_sscores, precompute_fb_errors, precompute_neighborhood_errors

# --- SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_vid_path = "UCF_Rep/val/v_BodyWeightSquats_g24_c01.mp4" # Put your test video here
output_vid_path = "tracking_result.mp4"

# 1. Load the Trained Brain
print("Loading trained model...")
# UPDATE 1: state_dim is now 4 to match the new budget state feature
policy_net = DQN(state_dim=4, action_dim=3).to(device)
policy_net.load_state_dict(torch.load("point_tracker_dqn.pth", map_location=device))
policy_net.eval()

# 2. Initialize the Environment
print(f"Processing Test Video: {test_vid_path}")

# --- AUTO-PRECOMPUTE BLOCK ---
vid_name = os.path.basename(test_vid_path)
npz_name = vid_name.replace('.mp4', '.npz')
save_dir = "precomputed_data"
save_path = os.path.join(save_dir, npz_name)

os.makedirs(save_dir, exist_ok=True)

if not os.path.exists(save_path):
    print(f"Precomputed data not found for {vid_name}. Running DINO & Flow now (this will take a minute)...")
    all_flows, start_points = get_optical_flow(test_vid_path)
    all_descriptors = get_dino_sscores(test_vid_path, all_flows, start_points)
    fb_errors = precompute_fb_errors(test_vid_path, all_flows, start_points)
    nb_errors = precompute_neighborhood_errors(all_flows, start_points, radius=50)
    
    np.savez_compressed(
        save_path,
        flows=all_flows,
        start_points=start_points,
        descriptors=all_descriptors,
        fb_errors=fb_errors,
        nb_errors=nb_errors
    )
    print("Precomputation complete!")

producer = pointStateProducer(vid_name) 
active_points = np.ones(producer.num_points, dtype=bool)

# 3. Setup Video Writer for Visualization
cap = cv2.VideoCapture(test_vid_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_vid_path, fourcc, fps, (width, height))

curr_positions = producer.start_points.reshape(-1, 2).copy()

ret, frame = cap.read()
if ret:
    out.write(frame)

print("Running Inference...")
for t in range(producer.num_frames - 1):
    ret, frame = cap.read()
    if not ret:
        break
        
    curr_positions = curr_positions + np.nan_to_num(producer.all_flows[t])

    # NEW: Array to hold the EXACT action taken by each point on THIS specific frame
    frame_actions = np.full(producer.num_points, -1, dtype=int)

    # Let the Agent make decisions for every point
    for p_idx in range(producer.num_points):
        if not active_points[p_idx]:
            continue
            
        state = producer.get_state(p_idx, t)
        
        # If OpenCV lost it, it dies
        if state is None:
            active_points[p_idx] = False
            frame_actions[p_idx] = 0 # Visualize as a Kill
            continue
            
        # UPDATE 2: CHECK BUDGET
        can_reanchor = producer.reanchor_counts[p_idx] < producer.max_reanchors
            
        # THE AGENT ACTS
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy_net(state_t)
            
            # UPDATE 3: ACTION MASKING AT TEST TIME
            if not can_reanchor:
                # Force the model to choose between Keep (1) or Kill (0)
                q_values[0, 2] = -float('inf') 
                
            action = torch.argmax(q_values).item()
            
        # Log the action for visualization BEFORE applying consequences
        frame_actions[p_idx] = action
            
        # Apply the action
        if action == 0: # KILL
            active_points[p_idx] = False
        elif action == 2: # RE-ANCHOR
            producer.update_anchor(p_idx, t)

    # --- STRICT FRAME-BY-FRAME VISUALIZATION ---
    for p_idx in range(producer.num_points):
        x, y = curr_positions[p_idx]
        if np.isnan(x) or np.isnan(y):
            continue

        act = frame_actions[p_idx]

        # Action 1: KEEP (Green Circle)
        if act == 1: 
            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1, lineType=cv2.LINE_AA)

        # Action 2: RE-ANCHOR (Cyan Square)
        elif act == 2: 
            cv2.rectangle(frame, (int(x)-3, int(y)-3), (int(x)+3, int(y)+3), (255, 255, 0), -1)

        # Action 0: KILL (Red 'X')
        elif act == 0: 
            cv2.line(frame, (int(x)-4, int(y)-4), (int(x)+4, int(y)+4), (0, 0, 255), 2, lineType=cv2.LINE_AA)
            cv2.line(frame, (int(x)-4, int(y)+4), (int(x)+4, int(y)-4), (0, 0, 255), 2, lineType=cv2.LINE_AA)

    out.write(frame)

cap.release()
out.release()
print(f"Testing complete! Watch the results in: {output_vid_path}")