""" Simple video player """
import cv2

source_path = r".\Videos\ScenicDrive.mp4"
source = cv2.VideoCapture(source_path)
cv2.namedWindow("root", cv2.WINDOW_NORMAL)
FPS = 30
deley = 1000//FPS
counter = 1

while cv2.waitKey(deley) != ord("q"):

    success, frame = source.read()

    cv2.putText(
        frame,
        f"{counter}",
        (30,30),
        fontFace= cv2.FONT_HERSHEY_PLAIN,
        fontScale=2,
        color=(255,255,255),
        thickness=2
        )

    if not success: break
    
    cv2.imshow("root", frame)
    counter += 1
    

source.release()
cv2.destroyAllWindows()