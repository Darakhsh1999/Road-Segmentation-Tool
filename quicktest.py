import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline


### Combine images
## BLUE IMAGE
#img = np.zeros( (600,600,3), dtype= np.uint8)
#img[:,:,0] = 255 # BGR

#contours = np.array([[[150,150],[450,150],[450,450],[150,450]]])
#dummy = img.copy()
#cv2.fillPoly(dummy, pts =[contours], color=(0,255,0))

#alpha = 1.0 
#new_img = cv2.addWeighted(img,alpha,dummy,1-alpha,0)
#cv2.imshow("",new_img)
#cv2.imwrite("transimg.png", new_img)
#cv2.waitKey()


### Spline test

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


# Define some points:
points = np.array([[0, 1, 0.5],
                   [1, 1.5, -1]]).T  # a (nbre_points x nbre_dim) array

# Linear length along the line:
distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
distance = np.insert(distance, 0, 0)/distance[-1]

# Interpolation for different methods:
alpha = np.linspace(0, 1, 75)

interpolator =  interp1d(distance, points, kind="quadratic", axis=0)
interpolated_points = interpolator(alpha)

# Graph:
plt.figure(figsize=(7,7))
plt.plot(*interpolated_points.T, '-', label="cubic")

plt.plot(*points.T, 'ok', label='original points')
plt.axis('equal'); plt.legend(); plt.xlabel('x'); plt.ylabel('y')
plt.show()