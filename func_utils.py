import cv2
import numpy as np
import os
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

'''
given a video, generate its optical flow vectors and return numpy flow array for
specific points
'''
def get_optical_flow(vid_path, point_sampling_algo = None): 
    vid_1 = cv2.VideoCapture(vid_path)

    if not vid_1.isOpened():
        vid_1.release()
        raise ValueError("Could not open video path.")

    flow_vectors = []

    def sample_points(gray_frame):
        if point_sampling_algo is None:
            '''
            NOTE: this is "loose", to create more points..
            '''
            return cv2.goodFeaturesToTrack(
                gray_frame,
                maxCorners=500,
                qualityLevel=0.005,
                minDistance=4,
                blockSize=3
            )

        if callable(point_sampling_algo):  #pass a point_sampling function here...(#TODO)
            return point_sampling_algo(gray_frame)

        raise ValueError("point_sampling_algo must be None or a callable.")

    try:
        ret1, frame1 = vid_1.read()
        if not ret1:
            return np.empty((0, 0, 2), dtype=np.float32)

        prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        points = sample_points(prev_gray)
        start_points = points

        while True:
            ret2, frame2 = vid_1.read()

            if not ret2:
                break

            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            if points is None or len(points) == 0:
                points = sample_points(prev_gray)

            if points is None or len(points) == 0:
                flow_vectors.append(np.empty((0, 2), dtype=np.float32))
                prev_gray = gray2
                continue

            next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray2, points, None)

            frame_flow = np.full((points.shape[0], 2), np.nan, dtype=np.float32)
            if next_points is not None and status is not None:
                p0 = points.reshape(-1, 2)
                p1 = next_points.reshape(-1, 2)
                valid = status.reshape(-1) == 1
                frame_flow[valid] = p1[valid] - p0[valid]
                points = next_points[valid].reshape(-1, 1, 2)
            else:
                points = None

            flow_vectors.append(frame_flow)
            prev_gray = gray2
    finally:
        vid_1.release()

    if len(flow_vectors) == 0:
        return np.empty((0, 0, 2), dtype=np.float32)

    max_points = max(f.shape[0] for f in flow_vectors)
    flow_array = np.full((len(flow_vectors), max_points, 2), np.nan, dtype=np.float32)

    for i, frame_flow in enumerate(flow_vectors):
        flow_array[i, :frame_flow.shape[0], :] = frame_flow

    return flow_array, start_points #for flow array: (num_steps, max_points in full video, 2)

    '''
    1. num_steps is just frames-1, as optical flow cant be done on first/last frame
    2. max_points is going to be the same throughout, as the number of points
    in every frame is going to be the same.
    '''
    
'''
getting dino scores for a specific point's trajectory, throughout the video
'''

def get_dino_sscores(vid_path, flow_array, start_points):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
    dinov2.eval()
    
    # current_pos tracks where every point is at frame t
    # shape: (num_frames, num_points, 2)
    current_pos = np.zeros((flow_array.shape[0] + 1, flow_array.shape[1], 2))
    curr = start_points.reshape(-1, 2).copy()
    current_pos[0] = curr
    
    # Accumulate flow to get absolute coordinates for every frame
    for t in range(flow_array.shape[0]):
        curr = curr + np.nan_to_num(flow_array[t])
        current_pos[t+1] = curr
        
    cap = cv2.VideoCapture(vid_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    all_descriptors = []
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame_idx >= current_pos.shape[0]:
                break
            
            # 1. Pre-process frame
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            transform = transforms.Compose([
                transforms.Resize((448, 448)), 
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_tensor = transform(img_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                # Features shape: [1, 1 + num_patches, 384] (1 is the CLS token)
                features = dinov2.get_intermediate_layers(input_tensor, n=1)[0]
                
                # 2. Reshape into a 2D grid [Batch, Channels, H_grid, W_grid]
                # Remove CLS token and reshape 1024 patches to 32x32
                grid_features = features[:, 1:, :].reshape(1, 32, 32, 384).permute(0, 3, 1, 2)

            # 3. Normalize coordinates to [-1, 1] for grid_sample
            # current_pos[frame_idx] has shape (num_points, 2)
            points_at_t = current_pos[frame_idx].copy()
            
            # Normalized coordinates formula:
            # x_norm = (x / width) * 2 - 1
            # y_norm = (y / height) * 2 - 1
            norm_pts = points_at_t.reshape(1, 1, -1, 2) # [Batch, H, W, 2]
            norm_pts[..., 0] = (norm_pts[..., 0] / w) * 2 - 1
            norm_pts[..., 1] = (norm_pts[..., 1] / h) * 2 - 1
            
            # 4. Bilinear Sampling
            # This extracts the 384-dim vector for every point
            sampled_feat = F.grid_sample(grid_features, torch.from_numpy(norm_pts).to(device).float(), 
                                         mode='bilinear', align_corners=False)
            
            # Reshape to (num_points, 384) and move to CPU
            all_descriptors.append(sampled_feat.reshape(384, -1).t().cpu().numpy())
            
            frame_idx += 1
    finally:
        cap.release()

    return np.array(all_descriptors) # (num_frames, num_points, 384)    


'''
precomputing the cycle consistency error for a point's trajectory throughout a video..
'''
