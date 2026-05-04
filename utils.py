import os
import cv2

def create_video(image_folder, video_name, fps=30):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort() # ensures frames are in order

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    video.release()
    print(f"Video saved as {video_name}")
    
def video_to_frames(video_path, output_folder, prefix="frame"):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_folder, f"{prefix}_{frame_idx:06d}.png")
        cv2.imwrite(frame_path, frame)
        frame_idx += 1

    cap.release()
    print(f"Saved {frame_idx} frames to {output_folder}")


# Extract frames and store in new_vids/{vid_name}/
# vid_name = 'v_Biking_g01_c01'
# video_to_frames(f'UCF_Rep/val/{vid_name}.mp4', f'new_vids/{vid_name}/')
# vid_name = 'validation_renders/v_HammerThrow_g21_c02'
vid_name = 'optical_flow_tracking_overlay_2'
os.makedirs(f'validation_renders/{vid_name}/', exist_ok=True)
video_to_frames(f'{vid_name}.mp4', f'new_vids/{vid_name}/')

# create_video(f'UCF_Rep/val/{vid_name}', f'/new_vids/{vid_name}_train.mp4')
    
# create_video('v_Biking_g01_c01/training_frames', 'v_Biking_g01_c01_train.mp4')