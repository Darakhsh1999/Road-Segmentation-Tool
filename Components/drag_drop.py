""" Implementing drag and drop functionality for points """
import cv2
import math

def mouse_callback(event, x, y, flags, params):
    global moving_circle
    global frame
    global point_xy

    if event == cv2.EVENT_LBUTTONDOWN: # MB1 down

        print(f"Clicked coordinate x: {x}, y: {y}")
        dist = math.sqrt((point_xy[0]-x)**2 + (point_xy[1]-y)**2)
        print(dist)
        if dist < click_lim: # Clicking on circle
            print("In circle distance")
            moving_circle = True
            return

    if moving_circle:
        frame = frame0.copy()
        point_xy = (x,y)
        cv2.circle(frame, point_xy, 15, (255,0,0), -1)

    if  event == 4:
        moving_circle = False




moving_circle = False
source_path = r".\Videos\maze.jpg"
frame0 = cv2.imread(source_path, -1)
click_lim = 10
im_shape = frame0.shape
maze_start = (555, 930)
point_xy = maze_start
cv2.namedWindow("root", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("root", mouse_callback)
frame = frame0.copy()
cv2.circle(frame, maze_start, 15, (255,0,0), -1)

while True:

    cv2.imshow("root", frame)
    res = cv2.waitKey(1)
    if res == ord("q"):
        break


cv2.destroyAllWindows()