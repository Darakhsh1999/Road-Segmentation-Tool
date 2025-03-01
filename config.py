""" Config class """
import os
import cv2 

class Config():

    # Debug level
    debug = 1 # 0 = None, 1 = prints

    video_name = "labeled_video"
    FPS = 30
    window_mode = cv2.WINDOW_NORMAL
    min_px_dist = 5 # Points in this range are connected
    frame_skips = 5 # number of frame skips between each "f" call
    color = (0,255,0) # color for points and fill, BGR format
    alpha = 0.4 
    t = 1 # thickness [px]
    r = 5 # radius [px]

    ### Kwargs
    text_kwargs = {
        "org": (30,30),
        "fontFace": cv2.FONT_HERSHEY_PLAIN,
        "fontScale": 2,
        "color": (255,255,255),
        "thickness": 2
    }
    circle_kwargs = {
        "radius": r,
        "color": color,
        "thickness": -1
    }
    highlight_kwargs = {
        "radius": r,
        "color": (0,0,0),
        "thickness": 2
    }
    line_kwargs = {
        "color": color,
        "thickness": 1
    }
    
    # Binary kwargs
    binary_circle_kwargs = {
        "radius": 1,
        "color": 255,
        "thickness": -1
    }
    binary_line_kwargs = {
        "color": 255,
        "thickness": 1
    }
