""" Simple video player """
import cv2

video_number = 7465
binary_source_path = r"C:\Users\arash\Documents\python projects\RST\Videos\labeled\labeled_video7465.avi"
source = cv2.VideoCapture(binary_source_path)
cv2.namedWindow("root", cv2.WINDOW_NORMAL)
FPS = 30
deley = 1000//FPS
counter = 1

while cv2.waitKey(deley) != ord("q"):

    success, frame = source.read()

    if not success:
        print("failed frame")
        break
    
    cv2.imshow("root", frame)
    counter += 1
    

source.release()
cv2.destroyAllWindows()