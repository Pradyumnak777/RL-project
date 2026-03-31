import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from rl_stuff import DQNAgent # importing our new rl brain

video_path = 'UCF_Rep/train/v_Biking_g01_c01.mp4'
video_path_test = 'UCF_Rep/train/v_Biking_g01_c01.mp4'
video_name = os.path.splitext(os.path.basename(video_path))[0] # get name without .mp4

video_name_test = os.path.splitext(os.path.basename(video_path_test))[0]
# create separate directories for this specific video
train_output_dir = os.path.join(video_name, 'training_frames')
test_output_dir = os.path.join(video_name_test, 'testing_frames')

for d in [train_output_dir, test_output_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

vid = cv2.VideoCapture(video_path)

# initialize rl agent
agent = DQNAgent(state_size=3, action_size=2)

# checkpoint config
checkpoint_dir = '/scratch/pbk5339/rl_project/checkpoints'
checkpoint_path = os.path.join(checkpoint_dir, f'{video_name}_dqn.pt')

# run_training=False => testing uses checkpoint only
run_training = True

# params for sampling and flow
feats = dict(maxCorners=100, qualityLevel=0.35, minDistance=7, blockSize=7) 
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# helper for similarity (using simple ncc for demo instead of dino)
def get_patch_similarity(img1, img2, p1, p2, size=9):
    h, w = img1.shape
    half = size // 2
    y1, x1 = int(p1[1]), int(p1[0])
    y2, x2 = int(p2[1]), int(p2[0])
    
    if y1-half < 0 or y1+half >= h or x1-half < 0 or x1+half >= w or \
       y2-half < 0 or y2+half >= h or x2-half < 0 or x2+half >= w:
        return 0.0
        
    patch1 = img1[y1-half:y1+half+1, x1-half:x1+half+1].astype(np.float32)
    patch2 = img2[y2-half:y2+half+1, x2-half:x2+half+1].astype(np.float32)
    
    num = np.sum((patch1 - np.mean(patch1)) * (patch2 - np.mean(patch2)))
    den = np.sqrt(np.sum((patch1 - np.mean(patch1))**2) * np.sum((patch2 - np.mean(patch2))**2))
    return num / (den + 1e-5)

# --- PHASE 1: TRAINING PASS ---
if run_training:
    print(f"Starting Training Pass for {video_name}...")
    ret, old_frame = vid.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feats)
    anchors = [(old_gray.copy(), p[0]) for p in p0]
    active_colors = [np.random.randint(0, 255, (3,)).tolist() for _ in range(len(p0))]
    mask = np.zeros_like(old_frame)

    frame_count = 0
    while True:
        ret, frame = vid.read()
        if not ret: break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        
        if p1 is not None:
            for i in range(len(p1)):
                if st[i] == 1:
                    sim = get_patch_similarity(anchors[i][0], frame_gray, anchors[i][1], p1[i][0])
                    state = [p1[i][0][0] - p0[i][0][0], p1[i][0][1] - p0[i][0][1], sim]
                    action = agent.act(state, train=True)
                    
                    # unsupervised reward signal based on similarity
                    reward = 1 if (action == 1 and sim > 0.7) else -1
                    agent.store_experience(state, action, reward, state, False)
                    agent.learn()
            
            display_frame = frame.copy()
            mask = cv2.addWeighted(mask, 0.95, np.zeros_like(mask), 0.05, 0)
            for i, p in enumerate(p1):
                if st[i] == 1:
                    a, b = p.ravel().astype(int)
                    c, d = p0[i].ravel().astype(int)
                    mask = cv2.line(mask, (a, b), (c, d), active_colors[i], 2)
                    display_frame = cv2.circle(display_frame, (a, b), 5, active_colors[i], -1)
            
            output_img = cv2.add(display_frame, mask)
            # cv2.putText(output_img, "train", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            plt.figure(figsize=(10, 6))
            plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.savefig(os.path.join(train_output_dir, f"train_{frame_count:04d}.png"), bbox_inches='tight')
            plt.close()

            old_gray = frame_gray.copy()
            p0 = p1
            frame_count += 1
            if frame_count % 10 == 0: agent.update_target_network()

    # save trained agent after training pass
    agent.save(checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
else:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    agent.load(checkpoint_path, load_optimizer=False)
    print(f"Loaded checkpoint for testing: {checkpoint_path}")

# --- PHASE 2: TESTING PASS ---
print(f"Training Complete. Starting Testing Pass for {video_name}...")
vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
agent.epsilon = 0 # stop exploring

ret, old_frame = vid.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feats)
anchors = [(old_gray.copy(), p[0]) for p in p0]
active_colors = [np.random.randint(0, 255, (3,)).tolist() for _ in range(len(p0))]
mask = np.zeros_like(old_frame)
frame_count = 0

while True:
    ret, frame = vid.read()
    if not ret: break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    if p1 is not None:
        rl_decisions = []
        for i in range(len(p1)):
            if st[i] == 1:
                sim = get_patch_similarity(anchors[i][0], frame_gray, anchors[i][1], p1[i][0])
                state = [p1[i][0][0] - p0[i][0][0], p1[i][0][1] - p0[i][0][1], sim]
                rl_decisions.append(agent.act(state, train=False)) # choose best action [cite: 136]
            else:
                rl_decisions.append(0)

        rl_decisions = np.array(rl_decisions)
        valid_indices = (st.flatten() == 1) & (rl_decisions == 1)
        
        good_new = p1[valid_indices]
        good_old = p0[valid_indices]
        anchors = [anchors[j] for j, v in enumerate(valid_indices) if v]
        active_colors = [active_colors[j] for j, v in enumerate(valid_indices) if v]
        
        mask = cv2.addWeighted(mask, 0.95, np.zeros_like(mask), 0.05, 0)
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            mask = cv2.line(mask, (a, b), (c, d), active_colors[i], 2)
            frame = cv2.circle(frame, (a, b), 5, active_colors[i], -1)
                
        output_img = cv2.add(frame, mask)
        # cv2.putText(output_img, "eval", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.savefig(os.path.join(test_output_dir, f"test_{frame_count:04d}.png"), bbox_inches='tight')
        plt.close()
        
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
        frame_count += 1
        
vid.release()
print(f"Finished! Results saved in directory: {video_name}")