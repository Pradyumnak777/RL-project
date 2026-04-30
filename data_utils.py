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
    def __init__(self, vid_name, max_reanchors=2, data_dir="precomputed_data_old"):
        data_path = os.path.join(data_dir, vid_name.replace('.mp4', '.npz'))
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Missing precomputed data for {vid_name}")

        data = np.load(data_path)
        
        self.all_flows = data['flows']
        self.all_descriptors = data['descriptors']
        self.fb_errors = data['fb_errors']
        self.nb_errors = data['nb_errors']
        try:
            self.start_points = data['start_points']
        except KeyError:
            # main.py doesn't use this, so we can just set it to None 
            # to avoid the crash.
            self.start_points = None
        
        self.num_frames = self.all_descriptors.shape[0]
        self.num_points = self.all_descriptors.shape[1]
        self.anchor_indices = np.zeros(self.num_points, dtype=int)
        
        self.max_reanchors = max_reanchors
        self.reanchor_counts = np.zeros(self.num_points, dtype=int)
        
    def get_state(self, point_idx, frame_idx):
        '''
        this will return a state/feature vector, which is for now: [speed, similarity, fb_error, nb_error, budget_used]
        '''
        #getting dx, dy stuff
        
        # 1. Get dx, dy and Errors (Handle the boundary)
        if frame_idx < self.all_flows.shape[0]: # Checks if we are within the N-1 limit
            dx, dy = self.all_flows[frame_idx, point_idx]
            current_fb_err = self.fb_errors[frame_idx, point_idx]
            current_nb_err = self.nb_errors[frame_idx, point_idx]
        else:
            # We are at the very last frame; no more flow exists
            dx, dy = 0.0, 0.0 
            current_fb_err = 0.0
            current_nb_err = 0.0
        
        #getting descriptors..
        current_desc = torch.from_numpy(self.all_descriptors[frame_idx, point_idx])
        anchor_idx = self.anchor_indices[point_idx] #should be 0, typically (so 0th frame/initial frame)
        anchor_desc = torch.from_numpy(self.all_descriptors[anchor_idx, point_idx])
        
        sim = F.cosine_similarity(current_desc.unsqueeze(0), anchor_desc.unsqueeze(0)).item()
        if np.isnan(dx):
            return None

        speed = np.sqrt(dx**2 + dy**2)
        
        # Normalize the count between 0.0 and 1.0 to keep neural net inputs stable
        budget_used = self.reanchor_counts[point_idx] / self.max_reanchors
                
        # State is now size 4: [dx, dy, sim, fb_error, nb_error, budget_used]
        return np.array([
            speed / 10.0, 
            sim, 
            np.clip(current_fb_err, 0, 5) / 5.0, 
            # np.clip(current_nb_err, 0, 5) / 5.0, #NOTE: removing this..
            budget_used
        ], dtype=np.float32)

    def update_anchor(self, point_idx, new_frame_idx):
        self.anchor_indices[point_idx] = new_frame_idx #change of anchor from initial frame to something else..
        self.reanchor_counts[point_idx] += 1
        
    def get_reward(self, point_idx, frame_idx, action):
        '''
        reward is a combination of cycle consistency and neighborhood consensus.
        '''
        # 1. Physical Failure (Point lost by OpenCV)
        if np.isnan(self.all_flows[frame_idx, point_idx, 0]):
            return -5.0 if action != 0 else 0.0 # Heavy penalty for failing to predict the crash

        # 2. Get Errors
        fb_err = self.fb_errors[frame_idx, point_idx]
        
        is_physically_stable = fb_err < 2.0
        
        state = self.get_state(point_idx, frame_idx)
        similarity = state[1] 
        looks_like_anchor = similarity > 0.92

        if action == 1: # KEEP
            # Best case: Stable and looks correct
            if is_physically_stable and looks_like_anchor:
                return 2.0
            # Danger case: Stable physics but appearance is drifting
            elif is_physically_stable and not looks_like_anchor:
                return -0.5 # but tells the agent "something is wrong"
            else:
                return -2.0 # Physics failed, keeping is a mistake

        elif action == 2: # RE-ANCHOR (The "Rescue" Decision)
            # The Sweet Spot: Physics are good, but appearance is drifting.
            # This is the "Alternative to Death" you were thinking of!
            if is_physically_stable and not looks_like_anchor:
                return 1.2 # High reward for a well-timed rescue!
            elif is_physically_stable and looks_like_anchor:
                return -0.2 # Allowed, but wasteful to use a budget if sim is already high
            else:
                return -5.0 # Cannot re-anchor a point with broken physics

        elif action == 0: # KILL
            # If it's physically broken OR semantically drifting, allow the 0.1 exit.
            if not is_physically_stable or not looks_like_anchor:
                return 0.05 
            else:
                return -5.0 # ONLY punish if the point is physically AND visually perfect

    # def get_reward(self, point_idx, frame_idx, action):
    #     '''
    #     reward is a combination of cycle consistency and neighborhood consensus.
    #     '''
    #     # 1. Physical Failure (Point lost by OpenCV)
    #     if np.isnan(self.all_flows[frame_idx, point_idx, 0]):
    #         return -5.0 if action != 0 else 0.0 # Heavy penalty for failing to predict the crash

    #     # 2. Get Errors
    #     fb_err = self.fb_errors[frame_idx, point_idx]
    #     nb_err = self.nb_errors[frame_idx, point_idx]
    #     w_fb = 0.8  # Primary focus: Cycle Consistency
    #     w_nb = 0.2  # Secondary focus: Neighborhood Consensus

    #     # Calculate weighted error
    #     total_error = (w_fb * np.clip(fb_err, 0, 5)) + (w_nb * np.clip(nb_err, 0, 5))
    #     # total_error = np.clip(fb_err, 0, 5) + np.clip(nb_err, 0, 5)

    #     # 3. Action Logic
    #     survival_bonus = 1.8
    #     if action == 1: # KEEP
    #         return survival_bonus - total_error

    #     elif action == 2: # RE-ANCHOR
    #         # survival_bonus = 2
    #         # reanchor_cost = 1
    #         return (survival_bonus - 0.2) - total_error

    #     elif action == 0: # KILL
    #         # If the point was healthy (low error), killing it is a HUGE mistake.
    #         # This 'Opportunity Cost' forces the agent to keep points.
    #         if total_error < 3.0: 
    #             return -5.0 
            
    #         # If the point was actually bad, killing it is a 'Neutral Exit'
    #         return 0.0

    #     raise ValueError(f"Invalid action {action}. Expected one of [0, 1, 2].")