import numpy as np
import cv2
import pynput
# BLUE IMAGE
img = np.zeros( (600,600,3), dtype= np.uint8)
img[:,:,0] = 255 # BGR

contours = np.array([[[150,150],[450,150],[450,450],[150,450]]])
dummy = img.copy()
cv2.fillPoly(dummy, pts =[contours], color=(0,255,0))

alpha = 1.0 
new_img = cv2.addWeighted(img,alpha,dummy,1-alpha,0)
cv2.imshow("",new_img)
cv2.imwrite("transimg.png", new_img)
cv2.waitKey()