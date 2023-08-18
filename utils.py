import math
import numpy as np
from point import Point
from scipy.interpolate import interp1d

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

def mid_point(point1: Point, point2: Point):
    """ Returns the mid point coordinate between two points """
    x1, y1 = point1.pos
    x2, y2 = point2.pos
    return ((x1+x2)//2, (y1+y2)//2)

def poly_coefficients(p1, p2, p3):
    """ Returns the 2nd degree polynomial coefficients (a,b,c) that pass through the points p1,p2,p3 """
    # Equation taken from https://math.stackexchange.com/a/3967249/1067530
    x1, y1 = p1.pos
    x2, y2 = p2.pos
    x3, y3 = p3.pos
    
    # Short hand notation
    x11, x22, x33 = x1**2, x2**2, x3**2
    x12, x13, x23 = x1*x2, x1*x3, x2*x3

    # Coefficients
    a = y1/(x11 - x12 - x13 + x23) + y2/(-x12 + x13 + x22 - x23) + y3/(x12 - x13 - x23 + x33)
    b = - 1.0/(x11 - x12 - x13 + x23)*(x2*y1 + x3*y1) - 1.0/(-x12 + x13 + x22 - x23)*(x1*y2 + x3*y2) - 1.0/(x12 - x13 - x23 + x33)*(x1*y3 + x2*y3) 
    c = (x23*y1)/(x11 - x12 - x13 + x23) + (x13*y2)/(-x12 + x13 + x22 - x23) + (x12*y3)/(x12 - x13 - x23 + x33)

    return (a,b,c)

def spline_curve(path_list: list[Point]):

    # get the relevant points
    points = np.array([x.pos for x in path_list]) # (3,2)

    # calculate distance
    distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
    n_pixel_samples = int(3*distance[-1])
    distance = np.insert(distance, 0, 0)/distance[-1]

    # interpolate
    alpha = np.linspace(0, 1, n_pixel_samples)

    interpolator =  interp1d(distance, points, kind=len(path_list)-1, axis=0)
    interpolated_points = interpolator(alpha)

    # convert to pixel values (rounding and remove duplicates)
    rounded_points = np.round(interpolated_points).astype(np.int64) # (N,2)
    rounded_points = np.unique(rounded_points, axis=0)

    return rounded_points


if __name__ == "__main__":

    from point import Point

    # 3x^2 + 2x - 7
    p1 = Point(0,3)
    p2 = Point(1,0)
    p3 = Point(2,-11)
    a,b,c = poly_coefficients(p1,p2,p3)
    print(a)
    print(b)
    print(c)
