""" Skips through frames with key """
import cv2

source_path = r".\Videos\ScenicDrive.mp4"
source = cv2.VideoCapture(source_path)
cv2.namedWindow("root", cv2.WINDOW_NORMAL)
counter = 1
success, frame = source.read()
res = None

while True:
    
    cv2.imshow("root", frame)
    
    res = cv2.waitKey(0)
    if res == ord("f"):
        counter += 10
        for i in range(10):
            success, frame = source.read()
        cv2.putText(
            frame,
            f"{counter}",
            (30,30),
            fontFace= cv2.FONT_HERSHEY_PLAIN,
            fontScale= 2,
            color= (255,255,255),
            thickness= 2)
    elif res == ord("q"):
        break
    else:
        continue

source.release()
cv2.destroyAllWindows()