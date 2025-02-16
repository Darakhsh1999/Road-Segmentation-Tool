import numpy as np
import matplotlib.pyplot as plt


p1 = np.array([1,1])
p2 = np.array([5,3])
p3 = np.array([0,8])
P = np.vstack((p1,p2,p3))
print(P.shape)

# draw line between P1 P2: d
d = (p2 - p1)

# calculate vector P1->P3
p3p1 = p3 - p1


# shortest distance vector is orthogonal projection of P1->P3 on line d
projection = (np.dot(p3p1,d) / (np.dot(d,d))) * d
orthogonal = p3p1 - projection
intersection = p3 - orthogonal



# DONT CHANGE
plt.scatter(P[:,0],P[:,1]) # all points
plt.plot([p1[0],p2[0]], [p1[1],p2[1]], "-") # lines between p1 p2

plt.plot([p3[0],intersection[0]], [p3[1],intersection[1]], "-") # lines between p3 intersection


plt.grid()
plt.xlim([0,10])
plt.ylim([0,10])
plt.xticks(np.arange(10))
plt.yticks(np.arange(10))
plt.show()