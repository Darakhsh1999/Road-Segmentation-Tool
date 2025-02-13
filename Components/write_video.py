import cv2
import numpy as np
import matplotlib.pyplot as plt


frame_shape = (800,500) # (W,H,3)
FPS = 30
#video_source = cv2.VideoWriter(r"Videos\test_video.avi", cv2.VideoWriter_fourcc("M","J","P","G"), FPS, frame_shape)
video_source = cv2.VideoWriter(r".\test_video.avi", cv2.VideoWriter_fourcc("M","J","P","G"), FPS, frame_shape)


for i in range(300): # frames

    if i % FPS == 0:
        array = np.random.choice([0,255], size=frame_shape, replace=True, p=[0.9,0.1]).astype(np.uint8).T
        video_frame = np.stack((array,array,array), axis=-1) 
    video_source.write(video_frame)

print(video_frame.shape)


video_source.release()
cv2.destroyAllWindows()