import torch
from torch.utils.data import Dataset
import os
from func_utils import get_optical_flow, get_dino_sscores, precompute_fb_errors, precompute_neighborhood_errors
import torch.nn.functional as F
import numpy as np

'''
create a state producer:

1. takes in video_path -> retrieves the point trajectories, and the dino features for every point, at every time
2. forms a beginning state
3. explores => (s, a, r, s')
5. This is then fed to the DQN
'''

class pointStateProducer:
    def __init__(self, vid_path):
        self.all_flows, self.start_points = get_optical_flow(vid_path) #start_points are the intiial points for this video!!
        self.all_descriptors = get_dino_sscores(vid_path, self.all_flows, self.start_points)

        self.num_frames = self.all_descriptors.shape[0]
        self.num_points = self.all_descriptors.shape[1]
        
        self.anchor_indices = np.zeros(self.num_points, dtype=int) #anchor is start frame, so all 0
        
        #getting the precomputed errors-
        self.fb_errors = precompute_fb_errors(vid_path, self.all_flows, self.start_points)
        self.nb_errors = precompute_neighborhood_errors(self.all_flows, self.start_points, radius=50)
        
    def get_state(self, point_idx, frame_idx):
        '''
        this will return a state/feature vector, which is for now: [dx, dy, similarity]
        '''
        #getting dx, dy stuff
        
        if frame_idx < self.all_flows.shape[0]:
            dx, dy = self.all_flows[frame_idx, point_idx]
        else:
            dx, dy = 0.0, 0.0 #if its the last frame
        
        #getting descriptors..
        current_desc = torch.from_numpy(self.all_descriptors[frame_idx, point_idx])
        anchor_idx = self.anchor_indices[point_idx] #should be 0, typically (so 0th frame/initial frame)
        anchor_desc = torch.from_numpy(self.all_descriptors[anchor_idx, point_idx])
        
        sim = F.cosine_similarity(current_desc.unsqueeze(0), anchor_desc.unsqueeze(0)).item()
        if np.isnan(dx):
            return None
        
        return np.array([dx, dy, sim], dtype=np.float32)

    def update_anchor(self, point_idx, new_frame_idx):
        self.anchor_indices[point_idx] = new_frame_idx #change of anchor from initial frame to something else..

    def get_reward(self, point_idx, frame_idx, action):
        '''
        reward is a combination of cycle consistency and neighborhood consensus.
        '''
        # 1. Physical Failure (Point lost by OpenCV)
        if np.isnan(self.all_flows[frame_idx, point_idx, 0]):
            return -5.0 if action != 0 else 0.0 # Heavy penalty for failing to predict the crash

        # 2. Get Errors
        fb_err = self.fb_errors[frame_idx, point_idx]
        nb_err = self.nb_errors[frame_idx, point_idx]
        total_error = np.clip(fb_err, 0, 5) + np.clip(nb_err, 0, 5)

        # 3. Action Logic
        if action == 1 or action == 2: # KEEP or RE-ANCHOR
            # SURVIVAL BONUS (e.g., 1.5) minus the penalty of being imprecise
            survival_bonus = 1.5 
            reward = survival_bonus - total_error
            
            if action == 2: # Small cost to re-anchor
                reward -= 0.2
            return reward

        elif action == 0: # KILL
            # If the point was healthy (low error), killing it is a HUGE mistake.
            # This 'Opportunity Cost' forces the agent to keep points.
            if total_error < 1.0: 
                return -5.0 
            
            # If the point was actually bad, killing it is a 'Neutral Exit'
            return 0.0