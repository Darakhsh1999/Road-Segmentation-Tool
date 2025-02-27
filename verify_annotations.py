import cv2

video_number = 7465
source_path = r"C:\Users\arash\Documents\python projects\RST\Videos\ScenicDrive_short.mp4"
binary_source_path = r"C:\Users\arash\Documents\python projects\RST\Videos\labeled\labeled_video1578.avi"
source = cv2.VideoCapture(source_path)
binary_source = cv2.VideoCapture(binary_source_path)
cv2.namedWindow("root", cv2.WINDOW_NORMAL)
FPS = 30
deley = 1000//FPS
counter = 1
alpha = 0.5

# Play video where we combine frames
while cv2.waitKey(deley) != ord("q"):

    success, frame = source.read()
    success2, binary_frame = binary_source.read()

    if (not success) or (not success2): break

    binary_frame[:,:,1:3] = 0
    combined_frame = cv2.addWeighted(frame,1.0,binary_frame,alpha,0)
    cv2.imshow("root", combined_frame)
    counter += 1
    

source.release()
binary_source.release()
cv2.destroyAllWindows()