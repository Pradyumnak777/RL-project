import cv2
import numpy as np
import os

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
            return cv2.goodFeaturesToTrack(
                gray_frame,
                maxCorners=200,
                qualityLevel=0.01,
                minDistance=7,
                blockSize=7
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

    return flow_array #(num_steps, max_points in full video, 2)

    '''
    1. num_steps is just frames-1, as optical flow cant be done on first/last frame
    2. max_points is going to be the same throughout, as the number of points
    in every frame is going to be the same.
    '''