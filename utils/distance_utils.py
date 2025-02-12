import math

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