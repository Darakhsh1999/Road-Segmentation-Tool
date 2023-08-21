import cv2
import numpy as np
import matplotlib.pyplot as plt


frame_shape = (500,500)
FPS = 30
video_source = cv2.VideoWriter(r"Videos\test_video.avi", cv2.VideoWriter_fourcc("M","J","P","G"), FPS, frame_shape)


for i in range(300): # frames

    if i % FPS == 0:
        array = np.random.randint(0, 255, size=frame_shape, dtype=np.uint8)
        video_frame = np.stack((array,array,array), axis=-1)
    video_source.write(video_frame)


video_source.release()
cv2.destroyAllWindows()