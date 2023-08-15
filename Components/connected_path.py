""" Connects a linearly piecewise path """
import cv2

def on_click(event, x, y, flags, params):
    global path, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(frame,(x,y),1,(255,0,0),-1)
        if path:
            cv2.line(frame, path[-1], (x,y), (255,0,0), 1)
        print(f"x: {x}, y: {y}")
        path.append((x,y))


source_path = r".\Videos\ScenicDrive.mp4"
source = cv2.VideoCapture(source_path)
counter = 1
success, frame = source.read()
res = None
path = []
cv2.namedWindow("root", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("root", on_click, [frame])

while True:
    
    cv2.imshow("root", frame)
    counter += 10
    
    res = cv2.waitKey(1)
    if res == ord("f"): # skip forward
        for i in range(10):
            success, frame = source.read()
            cv2.setMouseCallback("root", on_click)
            path = []
    elif res == ord("q"): # quit
        break
    else:
        continue

source.release()
cv2.destroyAllWindows()
