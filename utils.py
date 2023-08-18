import math
import numpy as np

def min_distance(path, x, y):
    """ Returns length and index to closest point """

    min_dist = math.inf

    if path:
        for idx, point in enumerate(path):
            px, py = point.pos
            dist = math.sqrt((x-px)**2 + (y-py)**2)
            if dist < min_dist:
                min_dist = dist
                min_idx = idx

        return min_dist, min_idx
    else:
        return min_dist, None

def distance(point_pos, x, y):
    """ Returns distance between point and click """
    return math.sqrt((x-point_pos[0])**2 + (y-point_pos[1])**2)


def poly_coefficients(p1, p2, p3):
    x1, y1 = p1.pos
    x2, y2 = p2.pos
    x3, y3 = p3.pos
    a = y1/(x1*x1 - x1*x2 - x1*x3 + x2*x3) + y2/(-x1*x2 + x1*x3 + x2*x2 - x2*x3) + y3/(x1*x2 - x1*x3 - x2*x3 + x3*x3)
    b = - (x2*y1)/(x1*x1 - x1*x2 - x1*x3 + x2*x3) - (x3*y1)/(x1*x1 - x1*x2 - x1*x3 + x2*x3) - (x1*y2)/(-x1*x2 + x1*x3 + x2*x2 - x2*x3) - (x3*y2)/(-x1*x2 + x1*x3 + x2*x2 - x2*x3) - (x1*y3)/(x1*x2 - x1*x3 - x2*x3 + x3*x3) - (x2*y3)/(x1*x2 - x1*x3 - x2*x3 + x3*x3)
    c = (x2*x3*y1)/(x1*x1 - x1*x2 - x1*x3 + x2*x3) + (x1*x3*y2)/(-x1*x2 + x1*x3 + x2*x2 - x2*x3) + (x1*x2*y3)/(x1*x2 - x1*x3 - x2*x3 + x3*x3)
    return (a,b,c)

def spline_mask(a,b,c,p1,p3):
    x1, y1 = p1.pos
    x3, y3 = p3.pos

    x = np.arange(x1,x3+1)
    y = np.round((a*x**2 + b*x + c)).astype(np.uint16)

    return x,y



if __name__ == "__main__":

    from point import Point

    p1 = Point(150,300)
    p2 = Point(300,450)
    p3 = Point(450,300)
    a,b,c = poly_coefficients(p1,p2,p3)
    print(a)
    print(b)
    print(c)
