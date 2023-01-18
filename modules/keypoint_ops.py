import cv2
from math import sqrt
import numpy as np
from numba import jit
from numba.core import types
from numba.typed import Dict

def init(find_sparse_keypoints, n_keypoints):
    global FIND_SPARSE_KEYPOINTS, N_KEYPOINTS,detector
    FIND_SPARSE_KEYPOINTS = find_sparse_keypoints
    N_KEYPOINTS = n_keypoints
    if FIND_SPARSE_KEYPOINTS == True:
        detector = cv2.SIFT_create(contrastThreshold=-1)
    else:
        detector = cv2.SIFT_create(nfeatures=N_KEYPOINTS)

@jit(nopython=True, cache=True, fastmath=True)
def check_distance(target_keypoint_x, target_keypoint_y, keypoints, keypoints_neighbors):
    skip_flag = False
    for kpt in keypoints:
        if kpt[0] == 0 and kpt[1] == 0: #_keypoints is zeroed
            break
        x,y = kpt
        dist = sqrt((target_keypoint_y-y)**2 + (target_keypoint_x-x)**2)
        if dist < 50: #is_neighbor
            hash = ((x + y)*(x + y + 1)/2) + y # https://stackoverflow.com/a/682617
            if keypoints_neighbors[hash] >= 3:
                skip_flag = True
                break
    return skip_flag

@jit(nopython=True, cache=True, fastmath=True)
def update_neighbors(target_keypoint_x, target_keypoint_y, keypoints, keypoints_neighbors):
    new_kpt_hash = ((target_keypoint_x + target_keypoint_y)*(target_keypoint_x + target_keypoint_y + 1)/2) + target_keypoint_y
    keypoints_neighbors[new_kpt_hash]=0
    for kpt in keypoints:
        x,y = kpt
        if x == target_keypoint_x and y == target_keypoint_y:
            continue
        if x == 0 and y == 0: #_keypoints is zeroed
            break
        dist = sqrt((target_keypoint_y-y)**2 + (target_keypoint_x-x)**2)
        if dist < 50: #is_neighbor
            hash = ((x + y)*(x + y + 1)/2) + y # https://stackoverflow.com/a/682617
            keypoints_neighbors[hash]+=1

def get_keypoints_sparse(img,n_features=200):
    height= img.shape[0]
    width= img.shape[1]
    height_divided_by_2 = img.shape[0]//2
    width_divided_by_2 = img.shape[1]//2
    _keypoints = np.zeros((n_features, 2)) #_keypoints is only used to pass it to numba optimized functions, because they can't use keypoints(it's list of cv2.Keypoint)
    kps = detector.detect(img,None)
    kps = sorted(kps, key = lambda x:x.response,reverse=True)
    keypoints_count = [0,0,0,0]
    keypoints=[]
    N = n_features//4
    used_keypoints = 0
    keypoints_neighbors = Dict.empty(key_type=types.float64, value_type=types.int64)
    def add_kpt(kpt):
        nonlocal used_keypoints, _keypoints, keypoints
        x,y = kpt.pt
        _keypoints[used_keypoints][0] = x
        _keypoints[used_keypoints][1] = y
        keypoints.append(kpt)
        update_neighbors(x, y, _keypoints, keypoints_neighbors)
        used_keypoints+=1

    for keypoint in kps:
        keypoint_x,keypoint_y=keypoint.pt
        if len(keypoints) != 0:
            skip_keypoint = check_distance(keypoint_x, keypoint_y, _keypoints, keypoints_neighbors)
            if skip_keypoint:
                continue

        if used_keypoints == 200:
            break

        if keypoints_count[0]<N and 0<keypoint_y<height_divided_by_2 and 0<keypoint_x<width_divided_by_2:
            add_kpt(keypoint)
            keypoints_count[0]+=1
            continue

        if keypoints_count[1]<N and 0<keypoint_y<height_divided_by_2 and width_divided_by_2<keypoint_x<width:
            add_kpt(keypoint)
            keypoints_count[1]+=1
            continue

        if keypoints_count[2]<N and height_divided_by_2<keypoint_y<height and 0<keypoint_x<width_divided_by_2:
            add_kpt(keypoint)
            keypoints_count[2]+=1
            continue

        if keypoints_count[3]<N and height_divided_by_2<keypoint_y<height and 0<width_divided_by_2<keypoint_x<width:
            add_kpt(keypoint)
            keypoints_count[3]+=1
            continue
    return keypoints

def get_keypoints(img):
    if FIND_SPARSE_KEYPOINTS:
        return get_keypoints_sparse(img, N_KEYPOINTS)
    return detector.detect(img, None)