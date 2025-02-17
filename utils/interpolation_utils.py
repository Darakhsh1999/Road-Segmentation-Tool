import numpy as np

def path_point_mapping(D: np.ndarray):

    M = D.shape[0]
    mapping_dict = {}

    for m in range(M):
        min_row_idx = D.min(axis=1).argmin()
        min_row = D[min_row_idx,:]
        min_col_idx = min_row.argmin()
        D[:,min_col_idx] = np.inf
        mapping_dict[int(min_row_idx)] = int(min_col_idx)

    return mapping_dict

def are_deep_copies(last_path, path):
    if len(path) != len(last_path):
        return False
    for point1, point2 in zip(path, last_path):
        if (point1.pos != point2.pos) or (point1.type != point2.type):
            return False
    return True


def shortest_path_intersection(p1,p2,p3):
    """
    Find shortest path intersection from point p3 to line
    going through points p1 and p2
    """

    d = p2 - p1
    p3p1 = p3 - p1
    projection = (np.dot(p3p1,d) / np.dot(d,d)) * d
    orthogonal = p3p1 - projection
    p_intersection = p3 - orthogonal
    return p_intersection

