import torch
import cv2
import numpy as np
from data_utils import pointStateProducer
from model import DQN

# --- SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_vid_path = "UCF_Rep/val/v_HammerThrow_g01_c01.mp4" # Put your test video here
output_vid_path = "tracking_result.mp4"

# 1. Load the Trained Brain
print("Loading trained model...")
policy_net = DQN(state_dim=3, action_dim=3).to(device)
policy_net.load_state_dict(torch.load("point_tracker_dqn.pth"))
policy_net.eval() # CRITICAL: Sets the network to evaluation mode (disables dropout, etc.)

# 2. Initialize the Environment
print(f"Processing Test Video: {test_vid_path}")
producer = pointStateProducer(test_vid_path)
active_points = np.ones(producer.num_points, dtype=bool)

# 3. Setup Video Writer for Visualization
cap = cv2.VideoCapture(test_vid_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_vid_path, fourcc, fps, (width, height))

# We need to track physical coordinates to draw them
# curr_positions starts with the initial goodFeaturesToTrack
curr_positions = producer.start_points.reshape(-1, 2).copy()

# Read the first frame
ret, frame = cap.read()
if ret:
    out.write(frame)

print("Running Inference...")
for t in range(producer.num_frames - 1):
    ret, frame = cap.read()
    if not ret:
        break
        
    # Update physical positions using the precomputed optical flow
    curr_positions = curr_positions + np.nan_to_num(producer.all_flows[t])

    # Let the Agent make decisions for every point
    for p_idx in range(producer.num_points):
        if not active_points[p_idx]:
            continue
            
        state = producer.get_state(p_idx, t)
        
        # If OpenCV lost it, it dies
        if state is None:
            active_points[p_idx] = False
            continue
            
        # THE AGENT ACTS (Pure Exploitation, no Epsilon)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy_net(state_t)
            action = torch.argmax(q_values).item()
            
        # Apply the action
        if action == 0: # KILL
            active_points[p_idx] = False
        elif action == 2: # RE-ANCHOR
            producer.update_anchor(p_idx, t)

    # --- VISUALIZATION ---
    # Draw a green circle for every point that the Agent has kept alive
    for p_idx in range(producer.num_points):
        if active_points[p_idx]:
            x, y = curr_positions[p_idx]
            # Ensure coordinates are valid before drawing
            if not np.isnan(x) and not np.isnan(y):
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                
    out.write(frame)

cap.release()
out.release()
print(f"Testing complete! Watch the results in: {output_vid_path}")