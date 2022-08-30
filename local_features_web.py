import uvicorn
if __name__ == '__main__':
    uvicorn.run('local_features_web:app', host='127.0.0.1', port=33333, log_level="info")
    exit()

from typing import Optional, Union
import cv2
import torch
import numpy as np
from math import sqrt
import kornia as K
import asyncio
import kornia.feature as KF
from os.path import exists
from fastapi import FastAPI, File, Form, HTTPException, Response, status
from pydantic import BaseModel
import faiss
import lmdb
import psycopg2
from kornia_moons import feature
from PIL import Image
import io 
from numba import jit
from numba.core import types
from numba.typed import Dict
from lru import LRU

dim = 128
index = None
DATA_CHANGED_SINCE_LAST_SAVE = False
laf_from_opencv_SIFT_kpts = feature.laf_from_opencv_SIFT_kpts
device = "cuda" if torch.cuda.is_available() else "cpu"
HardNet8 = KF.HardNet8(True).eval().to(device)
LAST_POINT_ID = 1
FIND_SPARSE_KEYPOINTS = True
N_KEYPOINTS = 200
LRU_CACHE = LRU(100)
#FIND_MIRRORED = True
app = FastAPI()
if FIND_SPARSE_KEYPOINTS == True:
    detector = cv2.SIFT_create(contrastThreshold=-1)
else:
    detector = cv2.SIFT_create(nfeatures=N_KEYPOINTS)

def main():
    global app, DB_img_points, DB_keypoints, DB_descriptors, LAST_POINT_ID
    DB_img_points = prepare_db()
    DB_keypoints = lmdb.open('./keypoints.lmdb',map_size=1 * 1000 * 1_000_000) #1gb
    DB_descriptors = lmdb.open('./descriptors.lmdb',map_size=5 * 1000 * 1_000_000) #5gb
    init_index()
    if DB_keypoints.stat()["entries"] != DB_descriptors.stat()["entries"]:
        print('DB_keypoints.stat()["entries"] != DB_descriptors.stat()["entries"]')
        exit()
    with DB_keypoints.begin(buffers=True) as txn:
        with txn.cursor() as curs:
            for key in curs.iternext(keys=True, values=False):
                key = int_from_bytes(key)
                LAST_POINT_ID = max(LAST_POINT_ID,key)
    if LAST_POINT_ID != 1:
        LAST_POINT_ID+=1
    loop = asyncio.get_event_loop()
    loop.call_later(10, periodically_save_index,loop)

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

def int_from_bytes(xbytes: bytes) -> int:
    return int.from_bytes(xbytes, 'big')

def create_table(conn):
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS img_points (
            image_id integer NOT NULL UNIQUE,
            point_id_range int4range NOT NULL)""")
    cur.execute('CREATE INDEX IF NOT EXISTS point_id_range_gist_index ON img_points USING GIST (point_id_range);')
    conn.commit()

def prepare_db():
    connect_settings_postgres = "dbname=postgres host=localhost user=postgres password=12345"
    connect_settings_ambience = "dbname=ambience host=localhost user=postgres password=12345"
    conn = psycopg2.connect(connect_settings_postgres)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = 'ambience'")
    exists = cur.fetchone()
    if not exists:
        conn.commit()
        conn.autocommit = True
        cur.execute('CREATE DATABASE ambience')
        conn = psycopg2.connect(connect_settings_ambience)
        create_table(conn)
        return conn
    else:
        conn = psycopg2.connect(connect_settings_ambience)
        create_table(conn)
        return conn

def int_to_bytes(x: int) -> bytes:
    return x.to_bytes((x.bit_length() + 7) // 8, 'big')

def init_index():
    global index
    if exists("./populated.index"):
        index = faiss.read_index("./populated.index")
        index_ivf = faiss.extract_index_ivf(index)
        index_ivf.nprobe = 16
    else:
        print("Index is not found! Exiting...")
        exit()

def check_if_image_id_exists(image_id):
    cursor = DB_img_points.cursor()
    cursor.execute("select exists(select 1 from img_points where image_id=%s)",[image_id])
    result = cursor.fetchone()
    return result[0]

def get_image_id_by_point_id(point_id):
    cursor = DB_img_points.cursor()
    cursor.execute("SELECT image_id FROM img_points WHERE point_id_range @> %s",[point_id])
    result = cursor.fetchone()
    if result is None:
        return None
    else:
        return result[0]

def get_point_ids(image_id):
    cursor = DB_img_points.cursor()
    cursor.execute("SELECT point_id_range FROM img_points WHERE image_id = %s",[image_id])
    result = cursor.fetchone()
    if result is None:
        return []
    else:
        return list(range(result[0].lower,result[0].upper))


def add_img_points(image_id,point_id_start,point_id_end):
    cursor = DB_img_points.cursor()
    cursor.execute("INSERT INTO img_points (image_id, point_id_range) VALUES(%s, %s)",[image_id, f'[{point_id_start},{point_id_end}]'])
    DB_img_points.commit()    

def add_keypoints(point_ids, kpts):
    keypoints_data = zip(point_ids,kpts)
    with DB_keypoints.begin(write=True, buffers=True) as txn:
        with txn.cursor() as curs:
            curs.putmulti(keypoints_data)

def add_descriptors(point_ids, descs):
    descs = [descriptor.tobytes() for descriptor in descs]
    descriptors_data = zip(point_ids, descs)
    with DB_descriptors.begin(write=True, buffers=True) as txn:
        with txn.cursor() as curs:
            curs.putmulti(descriptors_data)

def delete_img_points(image_id):
    cursor = DB_img_points.cursor()
    cursor.execute("DELETE FROM img_points WHERE image_id = %s",[image_id])
    DB_img_points.commit()

def delete_descriptors(point_ids):
    with DB_descriptors.begin(write=True, buffers=True) as txn:
        for point_id in point_ids:
            txn.delete(point_id)

def delete_keypoints(point_ids):
    with DB_keypoints.begin(write=True, buffers=True) as txn:
        for point_id in point_ids:
            txn.delete(point_id)

def read_img_buffer(image_data):
    img = Image.open(io.BytesIO(image_data))
    if img.mode != 'L':
        img = img.convert('L')
    return img

def resize_img_to_threshold(img):
    width, height = img.size
    threshold = 3000*3000
    if height*width > threshold:
        k = sqrt(height*width/threshold)
        img = img.resize((round(width/k), round(height/k)),Image.Resampling.LANCZOS)
    return img

def get_kpts_and_descs_by_id(image_id):
    point_ids = get_point_ids(image_id)
    if len(point_ids)==0:
        return None, None
    point_ids = [int_to_bytes(x) for x in point_ids]
    kpts=np.zeros( (len(point_ids), 2), dtype=np.float32 )
    descs=np.zeros( (len(point_ids), dim), dtype=np.float32 )
    with DB_keypoints.begin(buffers=True) as txn:
        with txn.cursor() as curs:
            _kpts = curs.getmulti(point_ids)
            for i in range(len(_kpts)):
                kpts[i]=np.frombuffer(_kpts[i][1], dtype=np.float32)
    with DB_descriptors.begin(buffers=True) as txn:
        with txn.cursor() as curs:
            _descs = curs.getmulti(point_ids)
            for i in range(len(_descs)):
                descs[i]=np.frombuffer(_descs[i][1], dtype=np.float32)
    return kpts, descs

def get_features(image_buffer, mirrored=False):
    img = read_img_buffer(image_buffer)
    img = resize_img_to_threshold(img)
    img = np.array(img)
    img_hash = hash(img.data.tobytes())
    if img_hash in LRU_CACHE:
        return LRU_CACHE[img_hash]
    if mirrored:
        img = np.fliplr(img)
    kpts = get_keypoints(img)
    if len(kpts) == 0:
        return None
    with torch.no_grad():
        timg = K.image_to_tensor(img, False).float()/255.
        timg = timg.to(device)
        lafs = laf_from_opencv_SIFT_kpts(kpts, device=device)
        patches = KF.extract_patches_from_pyramid(timg, lafs, 32)
        B, N, CH, H, W = patches.size()
        descs = HardNet8(patches.view(B * N, CH, H, W)).view(B * N, -1).cpu().numpy()   
    kpts = np.float32([x.pt for x in kpts]).reshape(-1,2)
    LRU_CACHE[img_hash] = (kpts,descs)
    return kpts, descs

def verify_pydegensac(src_pts,dst_pts,th = 4,  n_iter = 2000):
    _, mask = cv2.findHomography(src_pts, dst_pts, ransacReprojThreshold=th, confidence=0.999, maxIters = n_iter,method=cv2.USAC_MAGSAC)
    return int(mask.sum())

def local_features_search(orig_keypoints,target_features, k, k_clusters, knn_min_matches, matching_threshold,
use_smnn_matching, smnn_match_threshold,use_ransac):
    D, I = index.search(target_features, k_clusters)
    D = D.flatten()
    I = I.flatten()
    # print(D)
    # print(I)
    search_res={}
    for i in range(len(I)):
        if D[i] < matching_threshold:
            point_id = int(I[i])
            image_id = get_image_id_by_point_id(point_id)
            if image_id in search_res:
                search_res[image_id]+=1
            else:
                search_res[image_id] = 1
    res=[{"image_id":img_id, "matches":int(matches)} for img_id,matches in sorted(search_res.items(), key=lambda item: item[1],reverse=True) if matches>=knn_min_matches]
    if use_smnn_matching:
        new_res = []
        for item in res:
            kpts, descs = get_kpts_and_descs_by_id(item["image_id"])
            dists, match_ids = KF.match_smnn(torch.from_numpy(target_features), torch.from_numpy(descs), smnn_match_threshold)
            if len(dists) != 0:
                if use_ransac:
                    if len(dists) > 3:
                        new_res.append({"image_id":item["image_id"],"matches":verify_pydegensac(orig_keypoints[match_ids[:,0]],kpts[match_ids[:,1]])})
                else:
                    new_res.append({"image_id":item["image_id"],"matches":len(dists)})
        res = sorted(new_res, key=lambda item: item["matches"], reverse=True)
    if k:
        return res[:k]
    return res


@app.get("/")
async def read_root():
    return {"Hello": "World"}

class Item_local_features_get_similar_images_by_id(BaseModel):
    image_id: int
    k: Union[str,int,None] = None
    k_clusters: Union[str,int,None] = None
    knn_min_matches: Union[str,int,None] = None
    matching_threshold: Union[str,float,None] = None
    use_smnn_matching: Union[str,int,None] = None
    smnn_match_threshold: Union[str,float,None] = None
    use_ransac: Union[str,int,None] = None

@app.post("/local_features_get_similar_images_by_id")
async def local_features_get_similar_images_by_id_handler(item: Item_local_features_get_similar_images_by_id):
    try:
        image_id = int(item.image_id)
        k=item.k
        k_clusters=item.k_clusters
        matching_threshold=item.matching_threshold
        knn_min_matches=item.knn_min_matches

        use_smnn_matching=item.use_smnn_matching
        smnn_match_threshold=item.smnn_match_threshold
        use_ransac=item.use_ransac

        if k:
            k = int(k)
        if k_clusters:
            k_clusters = int(k_clusters)
        else:
            k_clusters=5
            
        if knn_min_matches:
            knn_min_matches = int(knn_min_matches)
        else:
            knn_min_matches=1

        if matching_threshold:
            matching_threshold = float(matching_threshold)
        else:
            matching_threshold = 0.9  

        if use_smnn_matching:
            use_smnn_matching=int(use_smnn_matching) #can be string, using int() to later use in a if statement as truthy/falsy value

        if smnn_match_threshold:
            smnn_match_threshold=float(smnn_match_threshold)
        else:
            smnn_match_threshold=0.8

        if use_ransac:
            use_ransac=int(use_ransac) #can be string, using int() to later use in a if statement as truthy/falsy value


        kpts, descs = get_kpts_and_descs_by_id(image_id)
        if kpts is None:
            return []
        similar = local_features_search(kpts, descs, k, k_clusters, knn_min_matches, matching_threshold, use_smnn_matching, smnn_match_threshold, use_ransac)
        return similar
    except:
        raise HTTPException(
            status_code=500, detail="Error in local_features_get_similar_images_by_id_handler")

@app.post("/local_features_get_similar_images_by_image_buffer")
async def local_features_get_similar_images_by_image_buffer_handler(image: bytes = File(...), 
 k: Optional[str] = Form(None), k_clusters: Optional[str] = Form(None),
 knn_min_matches: Optional[str] = Form(None), matching_threshold: Optional[str] = Form(None),
 use_smnn_matching: Optional[str] = Form(None), smnn_match_threshold: Optional[str] = Form(None), use_ransac: Optional[str] = Form(None)):
    try:
        if k:
            k = int(k)
        if k_clusters:
            k_clusters = int(k_clusters)
        else:
            k_clusters=5

        if knn_min_matches:
            knn_min_matches = int(knn_min_matches)
        else:
            knn_min_matches=1

        if matching_threshold:
            matching_threshold = float(matching_threshold)
        else:
            matching_threshold = 0.9  

        if use_smnn_matching:
            use_smnn_matching=int(use_smnn_matching) #can be string, using int() to later use in a if statement as truthy/falsy value

        if smnn_match_threshold:
            smnn_match_threshold=float(smnn_match_threshold)
        else:
            smnn_match_threshold=0.8

        if use_ransac:
            use_ransac=int(use_ransac) #can be string, using int() to later use in a if statement as truthy/falsy value
        
        kpts, descs = get_features(image)
        similar = local_features_search(kpts, descs, k, k_clusters, knn_min_matches, matching_threshold, use_smnn_matching, smnn_match_threshold, use_ransac)
        return similar
    except RuntimeError:
        raise HTTPException(status_code=500, detail="Error in local_features_get_similar_images_by_image_buffer_handler")


@app.post("/calculate_local_features")
async def calculate_local_features_handler(image: bytes = File(...), image_id: str = Form(...)):
    try:
        global DATA_CHANGED_SINCE_LAST_SAVE, LAST_POINT_ID
        image_id = int(image_id)
        if check_if_image_id_exists(image_id):
            raise HTTPException(status_code=500, detail="Image with this id is already in the db")
        kpts,descs = get_features(image)
        if descs is None:
            raise HTTPException(status_code=500, detail="No descriptors for this image")
        start = LAST_POINT_ID
        end = LAST_POINT_ID + len(kpts) - 1
        LAST_POINT_ID+=len(kpts)
        add_img_points(image_id,start,end)
        point_ids = list(range(start,end+1))
        point_ids_bytes = [int_to_bytes(x) for x in point_ids]
        add_keypoints(point_ids_bytes, kpts)
        add_descriptors(point_ids_bytes, descs)
        index.add_with_ids(descs, np.int64(point_ids))
        DATA_CHANGED_SINCE_LAST_SAVE = True
        return Response(status_code=status.HTTP_200_OK)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Can't calculate local features")


class Item_delete_local_features(BaseModel):
    image_id: int

@app.post("/delete_local_features")
async def delete_local_features_handler(item: Item_delete_local_features):
    global DATA_CHANGED_SINCE_LAST_SAVE
    try:
        image_id = int(item.image_id)
        point_ids = get_point_ids(image_id)
        if len(point_ids) != 0:
            point_ids_bytes = [int_to_bytes(x) for x in point_ids]
            delete_img_points(image_id)
            delete_keypoints(point_ids_bytes)
            delete_descriptors(point_ids_bytes)
            index.remove_ids( np.int64(point_ids) )
            DATA_CHANGED_SINCE_LAST_SAVE = True
            return Response(status_code=status.HTTP_200_OK)
        else:
            raise HTTPException(status_code=500, detail="Image with this id is not found")
    except:
        raise HTTPException(status_code=500, detail="Can't delete local features")

def periodically_save_index(loop):
    global DATA_CHANGED_SINCE_LAST_SAVE, index
    if DATA_CHANGED_SINCE_LAST_SAVE:
        DATA_CHANGED_SINCE_LAST_SAVE=False
        faiss.write_index(index, "./populated.index")
    loop.call_later(10, periodically_save_index,loop)
    
main()