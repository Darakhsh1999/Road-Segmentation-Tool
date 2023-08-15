""" Event listener of callback events for opencv window """
import cv2

def on_click(event, x, y, flags, params):
    if event != cv2.EVENT_MOUSEMOVE:
        command = event_dict.get(event, None)
        if command is not None:
            print(command)
        if event > 10:
            print(f"New event detected")

event_dict = {
    0: "mouse move", # EVENT_MOUSEMOVE
    1: "MB1 down", # EVENT_LBUTTONDOWN
    2: "MB2 down", # EVENT_RBUTTONDOWN
    3: "scroll down", # EVENT_MBUTTONDOWN
    4: "MB1 up", # EVENT_LBUTTONUP
    5: "MB2 up", # EVENT_RBUTTONUP
    6: "scroll up", # EVENT_MBUTTONUP
    7: "MB1 double click", # EVENT_LBUTTONDBLCLK
    8: "MB2 double click", # EVENT_RBUTTONDBLCLK
    9: "scroll double click", # EVENT_MBUTTONDBLCLK
    10: "Mouse wheel scroll" # EVENT_MOUSEWHEEL
}

cv2.namedWindow("root", cv2.WINDOW_NORMAL) # create window
cv2.setMouseCallback("root", on_click) # set callback

while cv2.waitKey(10) != ord("q"): # inf loop
    continue

cv2.destroyAllWindows() # exit window