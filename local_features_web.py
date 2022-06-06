import uvicorn
if __name__ == '__main__':
    uvicorn.run('local_features_web:app', host='127.0.0.1', port=33333, log_level="info")

from typing import Optional, Union
import cv2
import torch
import numpy as np
import math
import kornia as K
import asyncio
import kornia.feature as KF
from os.path import exists
from fastapi import FastAPI, File, Form, HTTPException, Response, status
from pydantic import BaseModel
import faiss
import lmdb
import psycopg2
import pydegensac

dim = 128
index = None
DATA_CHANGED_SINCE_LAST_SAVE = False

from kornia_moons import feature
laf_from_opencv_SIFT_kpts = feature.laf_from_opencv_SIFT_kpts

detector = cv2.SIFT_create(nfeatures=200)
device = "cuda" if torch.cuda.is_available() else "cpu"
HardNet8 = KF.HardNet8(True).eval().to(device)
POINT_ID = 0
FIND_MIRRORED = True


def prepare_db():
    conn = psycopg2.connect("dbname=postgres host=localhost user=postgres password=12345")
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = 'ambience'")
    exists = cur.fetchone()
    conn.commit()
    if not exists:
        conn.autocommit = True
        cur.execute('CREATE DATABASE ambience')
        conn = psycopg2.connect("dbname=ambience host=localhost user=postgres password=12345")
        cur = conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS img_points (
            image_id integer NOT NULL UNIQUE,
            point_id_range int4range NOT NULL)""")
        cur.execute('CREATE INDEX point_id_range_gist_index ON img_points USING GIST (point_id_range);')
        conn.commit()
        return conn
    else:
        conn = psycopg2.connect("dbname=ambience host=localhost user=postgres password=12345")
        return conn

DB_img_points = prepare_db()
DB_keypoints = lmdb.open('./keypoints.lmdb',map_size=6* 1000 * 1_000_000) #6gb
DB_descriptors = lmdb.open('./descriptors.lmdb',map_size=120 * 1000 * 1_000_000) #120gb


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
    cursor.execute("INSERT INTO img_points (image_id, point_id_range) VALUES %s",[image_id, f'[{point_id_start},{point_id_end}]'])
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

def read_img_file(image_data):
    return np.fromstring(image_data, np.uint8)

def resize_img_to_threshold(img):
    height, width = img.shape
    threshold = 3000*3000
    if height*width > threshold:
        k = math.sqrt(height*width/(threshold))
        img = cv2.resize(img, (round(width/k), round(height/k)), interpolation=cv2.INTER_LINEAR)
    return img


def preprocess_image(image_buffer):
    img = cv2.imdecode(read_img_file(image_buffer), 0)
    img = resize_img_to_threshold(img)
    return img

def get_kpts_and_descs_by_id(image_id):
    point_ids = get_point_ids(image_id)
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

def get_features(img, mirrored=False):
    if mirrored:
        img = cv2.flip(img, 1)
    kpts = detector.detect(img, None)
    if len(kpts) == 0:
        return None
    with torch.no_grad():
            timg = K.image_to_tensor(img, False).float()/255.
            timg = timg.to(device)
            lafs = laf_from_opencv_SIFT_kpts(kpts, device=device)
            patches = KF.extract_patches_from_pyramid(timg, lafs, 32)
            B, N, CH, H, W = patches.size()
            descs = HardNet8(patches.view(B * N, CH, H, W)).view(B * N, -1).detach().cpu().numpy()   
    kpts = np.float32([x.pt for x in kpts]).reshape(-1,2)
    return kpts, descs

def verify_pydegensac(src_pts,dst_pts,th = 4,  n_iter = 2000):
    _, mask = pydegensac.findHomography(src_pts, dst_pts, th, 0.999, n_iter)
    return int(mask.sum())

def local_features_search(orig_keypoints,target_features, k, k_clusters, min_matches, matching_threshold,
use_snn_matching, snn_match_threshold,use_ransac):
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
    res=[{"image_id":img_id, "matches":int(matches)} for img_id,matches in sorted(search_res.items(), key=lambda item: item[1],reverse=True) if matches>=min_matches]
    if use_snn_matching:
        new_res = []
        for item in res:
            kpts, descs = get_kpts_and_descs_by_id(item["image_id"])
            dists, match_ids = KF.match_snn(torch.from_numpy(target_features), torch.from_numpy(descs), snn_match_threshold)
            if use_ransac:
                if len(dists) > 3:
                    new_res.append({"image_id":item["image_id"],"matches":verify_pydegensac(orig_keypoints[match_ids[:,0]],kpts[match_ids[:,1]])})
            else:
                new_res.append({"image_id":item["image_id"],"matches":len(dists)})
        res = sorted(new_res, key=lambda item: item["matches"], reverse=True)
    if k:
        return res[:k]
    return res

app = FastAPI()
@app.get("/")
async def read_root():
    return {"Hello": "World"}

class Item_image_id(BaseModel):
    image_id: int
    k: Union[str,int,None] = None
    k_clusters: Union[str,int,None] = None
    min_matches: Union[str,int,None] = None
    matching_threshold: Union[str,float,None] = None
    use_snn_matching: Union[str,int,None] = None
    snn_match_threshold: Union[str,float,None] = None
    use_ransac: Union[str,int,None] = None

@app.post("/local_features_get_similar_images_by_id")
async def local_features_get_similar_images_by_id_handler(item: Item_image_id):
    try:
        image_id = int(item.image_id)
        k=item.k
        k_clusters=item.k_clusters
        matching_threshold=item.matching_threshold
        min_matches=item.min_matches

        use_snn_matching=item.use_snn_matching
        snn_match_threshold=item.snn_match_threshold
        use_ransac=item.use_ransac

        if k:
            k = int(k)
        if k_clusters:
            k_clusters = int(k_clusters)
        else:
            k_clusters=5
            
        if min_matches:
            min_matches = int(min_matches)
        else:
            min_matches=1

        if matching_threshold:
            matching_threshold = float(matching_threshold)
        else:
            matching_threshold = 0.9  

        if use_snn_matching:
            use_snn_matching=int(use_snn_matching)

        if snn_match_threshold:
            snn_match_threshold=float(snn_match_threshold)
        else:
            snn_match_threshold=0.8

        if use_ransac:
            use_ransac=int(use_ransac)


        kpts, descs = get_kpts_and_descs_by_id(image_id)
        similar = local_features_search(kpts, descs, k, k_clusters, min_matches, matching_threshold, use_snn_matching, snn_match_threshold, use_ransac)
        return similar
    except:
        raise HTTPException(
            status_code=500, detail="Error in local_features_get_similar_images_by_id_handler")

@app.post("/local_features_get_similar_images_by_image_buffer")
async def local_features_get_similar_images_by_image_buffer_handler(image: bytes = File(...), 
 k: Optional[str] = Form(None), k_clusters: Optional[str] = Form(None),
 min_matches: Optional[str] = Form(None), matching_threshold: Optional[str] = Form(None),
 use_snn_matching: Optional[str] = Form(None), snn_match_threshold: Optional[str] = Form(None), use_ransac: Optional[str] = Form(None)):
    try:
        if k:
            k = int(k)
        if k_clusters:
            k_clusters = int(k_clusters)
        else:
            k_clusters=5

        if min_matches:
            min_matches = int(min_matches)
        else:
            min_matches=1

        if matching_threshold:
            matching_threshold = float(matching_threshold)
        else:
            matching_threshold = 0.9  

        if use_snn_matching:
            use_snn_matching=int(use_snn_matching)

        if snn_match_threshold:
            snn_match_threshold=float(snn_match_threshold)
        else:
            snn_match_threshold=0.8

        if use_ransac:
            use_ransac=int(use_ransac)

        
        kpts, descs = get_features(preprocess_image(image))
        similar = local_features_search(kpts, descs, k, k_clusters, min_matches, matching_threshold, use_snn_matching, snn_match_threshold, use_ransac)
        return similar
    except RuntimeError:
        raise HTTPException(status_code=500, detail="Error in local_features_get_similar_images_by_image_buffer_handler")


@app.post("/calculate_local_features")
async def calculate_local_features_handler(image: bytes = File(...), image_id: str = Form(...)):
    try:
        global POINT_ID
        image_id = int(image_id)
        kpts,descs = get_features(preprocess_image(image))
        if descs is None:
            raise HTTPException(status_code=500, detail="No descriptors for this image")
        start = POINT_ID
        end = POINT_ID + len(kpts)
        POINT_ID+=len(kpts)+1
        add_img_points(image_id,start,end)
        point_ids = list(range(start,end+1))
        point_ids_bytes = [int_to_bytes(x) for x in point_ids]
        add_keypoints(point_ids_bytes, kpts)
        add_descriptors(point_ids_bytes, descs)
        index.add_with_ids(descs, np.int64(point_ids))
        return Response(status_code=status.HTTP_200_OK)
    except:
        raise HTTPException(status_code=500, detail="Can't calculate local features")


class Item(BaseModel):
    image_id: int

@app.post("/delete_local_features")
async def delete_local_features_handler(item: Item):
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

print(__name__)
if __name__ == 'local_features_web':
    init_index()
    loop = asyncio.get_event_loop()
    loop.call_later(10, periodically_save_index,loop)
