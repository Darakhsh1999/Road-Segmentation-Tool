import cv2
import numpy as np
import matplotlib.pyplot as plt

screen_res = (720,1280) # (H,W)
flipped_shape = (1280,720) # (W,H)
FPS = 30
video_source = cv2.VideoWriter(r".\test_video.avi", cv2.VideoWriter_fourcc("M","J","P","G"), FPS, flipped_shape) # (W,H)


for i in range(300): # frames

    if i % FPS == 0:
        array = np.random.choice([0,255], size=screen_res, replace=True, p=[0.9,0.1]).astype(np.uint8) # (H,W)
        video_frame = np.stack((array,array,array), axis=-1) # (H,W,3) 
    video_source.write(video_frame)

print(video_frame.shape)


video_source.release()
cv2.destroyAllWindows()