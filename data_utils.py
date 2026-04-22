import torch
from torch.utils.data import Dataset
import os
from func_utils import get_optical_flow

'''
create a state producer:

1. takes in video_path -> retrieves the point trajectories, and the dino features for every point, at every time
2. forms a beginning state
3. explores => (s, a, r, s')
5. This is then fed to the DQN
'''

class pointStateProducer:
    def __init__(self, vid_path):
        self.all_flows = get_optical_flow(vid_path)

        
