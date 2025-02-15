import sys
sys.path.append("../")
from point import Point
import numpy as np


def distance_matrix(last_path, path):
    M = len(last_path)
    N = len(path)
    p1_xy = np.array([list(x.pos) for x in last_path]) # (M,2)
    p2_xy = np.array([list(x.pos) for x in path]) # (N,2)

    D = np.zeros((M,N))
    for m, p1 in enumerate(p1_xy):
        for n, p2 in enumerate(p2_xy):
            D[m,n] = np.linalg.norm(p1-p2)

    return D


def find_mapping(D: np.ndarray):

    M = D.shape[0]
    mapping_dict = {}

    for m in range(M):
        min_row_idx = D.min(axis=1).argmin()
        min_row = D[min_row_idx,:]
        min_col_idx = min_row.argmin()
        D[:,min_col_idx] = np.inf
        mapping_dict[int(min_row_idx)] = int(min_col_idx)


    return mapping_dict

def calculate_delta(last_path, path, mapping_dict:dict):

    delta = np.zeros((len(last_path),2)) # (M,2)
    
    for k,v in mapping_dict.items():

        p_start = last_path[k].pos
        p_end = path[v].pos
        delta[k,:] = [p_end[0]-p_start[0],p_end[1]-p_start[1]] # [dx,dy]
    
    delta /= frame_skips

    return delta


def increment_path(path, delta, N):

    path_list = []
    for n in range(N-1):
        _path = []
        for idx, point in enumerate(path):
            xy = point.pos
            new_point = Point(int(xy[0]+(n+1)*delta[idx,0]), int(xy[1]+(n+1)*delta[idx,1]), point.type)
            _path.append(new_point)
        path_list.append(_path)

    return path_list


def print_path(path):
    for point_idx, point in enumerate(path):
        print(f"Point {point_idx}, (x,y)=({point.pos}), type= {point.type}")


# Linear M = N
frame_skips = 10
last_path = [Point(324,633),Point(609,472),Point(707,475),Point(948,668,type="end")]
path = [Point(324,633),Point(558,400),Point(707,394),Point(948,668,type="end")]
p1_xy = np.array([list(x.pos) for x in last_path]) # (M,2)
p2_xy = np.array([list(x.pos) for x in path]) # (N,2)


D = distance_matrix(last_path, path)
print(D)
map_dict = find_mapping(D)
delta = calculate_delta(last_path, path, map_dict)
print(delta)
path_list = increment_path(last_path,delta,N=frame_skips)


for paath in path_list:
    print_path(paath)
print("-----")
print_path(path)
