""" Changes modes """
import cv2

def update_mode(new_mode):

    cv2.putText(
        frame,
        f"Mode: {new_mode}",
        (30,30),
        fontFace= cv2.FONT_HERSHEY_PLAIN,
        fontScale= 2,
        color= (255,255,255),
        thickness= 2)

source_path = r".\Videos\ScenicDrive.mp4"
source = cv2.VideoCapture(source_path)
mode = "Visual"
success, frame = source.read()
frame0 = frame.copy()
update_mode(mode)
cv2.namedWindow("root", cv2.WINDOW_NORMAL)


while True:
    
    cv2.imshow("root", frame)

    res = cv2.waitKey(1)
    if res is None: continue

    if res == ord("f"): # skip forward
        for i in range(10):
            success, frame = source.read()
            frame0 = frame.copy() # save raw frame
            update_mode(mode)
            path = []
    elif res == ord("v"):
        frame = frame0.copy()
        mode = "Visual"
        update_mode(mode)
    elif res == ord("e"):
        frame = frame0.copy()
        mode = "Edit"
        update_mode(mode)
    elif res == ord("i"):
        frame = frame0.copy()
        mode = "Insert"
        update_mode(mode)
    elif res == ord("q"): # quit
        break
    else:
        continue

source.release()
cv2.destroyAllWindows()
