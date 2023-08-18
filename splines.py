import cv2
import utils
import numpy as np
import matplotlib.pyplot as plt
from point import Point

N = 600
canvas = np.zeros((N,N), dtype= np.uint8)

points = np.array([[150,300],[300,450],[450,300]]) # linear piecewise
cv2.polylines(canvas, [points], color=(255,255,255), isClosed= False)

# 2nd degree polynomial
p1 = Point(150,300)
p2 = Point(300,450)
p3 = Point(450,300)
a,b,c = utils.poly_coefficients(p1,p2,p3)
x,y =  utils.spline_mask(a,b,c,p1,p3)
canvas[y,x] = 255
cv2.imshow("",canvas)
cv2.waitKey(0)