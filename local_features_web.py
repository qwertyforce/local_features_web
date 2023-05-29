import uvicorn
if __name__ == '__main__':
    uvicorn.run('local_features_web:app', host='127.0.0.1', port=33333, log_level="info")
    exit()

import traceback
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
from kornia_moons import feature
from PIL import Image
import io 
from lru import LRU
from tqdm import tqdm

from modules.psql_ops import prepare_db 
from modules.byte_ops import int_to_bytes
from modules.lmdb_ops import get_dbs, get_last_point_id

dim = 128
index = None
DATA_CHANGED_SINCE_LAST_SAVE = False
laf_from_opencv_SIFT_kpts = feature.laf_from_opencv_SIFT_kpts
device = "cuda" if torch.cuda.is_available() else "cpu"
HardNet8 = KF.HardNet8(True).eval().to(device)

FIND_SPARSE_KEYPOINTS = True
N_KEYPOINTS = 200
from modules import keypoint_ops
keypoint_ops.init(FIND_SPARSE_KEYPOINTS, N_KEYPOINTS)

LRU_CACHE = LRU(100)
#FIND_MIRRORED = True
app = FastAPI()

def main():
    global DB_img_points, DB_keypoints, DB_descriptors, LAST_POINT_ID
    DB_img_points = prepare_db()
    DB_keypoints, DB_descriptors = get_dbs()

    init_index()
    LAST_POINT_ID = get_last_point_id()+1
    loop = asyncio.get_event_loop()
    loop.call_later(10, periodically_save_index,loop)

def init_index():
    global index
    if exists("./data/populated.index"):
        index = faiss.read_index("./data/populated.index")
        index_ivf = faiss.extract_index_ivf(index)
        index_ivf.nprobe = 16
    else:
        print("Index is not found!")
        print("Creating empty index")
        import subprocess
        try:
            subprocess.call(['python3', 'add_to_index.py'])
        except:
            pass
        try:                                                 #one of these should exist
            subprocess.call(['python', 'add_to_index.py']) 
        except:
            pass

def check_if_image_id_exists(image_id):
    cursor = DB_img_points.cursor()
    cursor.execute("select exists(select 1 from img_points where image_id=%s)",[image_id])
    result = cursor.fetchone()
    return result[0]

def get_image_id_and_file_name_by_point_id(point_id):
    cursor = DB_img_points.cursor()
    cursor.execute("SELECT image_id, file_name FROM img_points WHERE point_id_range @> %s",[point_id])
    result = cursor.fetchone()
    if result is None:
        return None
    else:
        return result

def get_point_ids_by_image_id(image_id):
    cursor = DB_img_points.cursor()
    cursor.execute("SELECT point_id_range FROM img_points WHERE image_id = %s",[image_id])
    result = cursor.fetchone()
    if result is None:
        return []
    else:
        return list(range(result[0].lower,result[0].upper))

def get_point_ids_by_filename(file_name):
    cursor = DB_img_points.cursor()
    cursor.execute("SELECT point_id_range FROM img_points WHERE file_name = %s",[file_name])
    result = cursor.fetchone()
    if result is None:
        return []
    else:
        return list(range(result[0].lower,result[0].upper))
    
def add_img_points(image_id, file_name, point_id_start,point_id_end):
    cursor = DB_img_points.cursor()
    cursor.execute("INSERT INTO img_points (image_id, file_name, point_id_range) VALUES(%s, %s, %s)",[image_id, file_name, f'[{point_id_start},{point_id_end}]'])
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

def delete_img_points_by_image_id(image_id):
    cursor = DB_img_points.cursor()
    cursor.execute("DELETE FROM img_points WHERE image_id = %s",[image_id])
    DB_img_points.commit()

def delete_img_points_by_filename(file_name):
    cursor = DB_img_points.cursor()
    cursor.execute("DELETE FROM img_points WHERE file_name = %s",[file_name])
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
    point_ids = get_point_ids_by_image_id(image_id)
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
    kpts = keypoint_ops.get_keypoints(img)
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

def verify_ransac(src_pts,dst_pts,th = 4,  n_iter = 2000):
    _, mask = cv2.findHomography(src_pts, dst_pts, ransacReprojThreshold=th, confidence=0.999, maxIters = n_iter,method=cv2.USAC_MAGSAC)
    return int(mask.sum())

def local_features_search(orig_keypoints,target_features, k, k_clusters, knn_min_matches, matching_threshold,
use_smnn_matching, smnn_match_threshold,use_ransac):
    D, I = index.search(target_features, k_clusters)
    D = D.flatten()
    I = I.flatten()
    # print(D)
    # print(I)
    res={}
    for i in range(len(I)):
        if D[i] < matching_threshold:
            point_id = int(I[i])
            image_id, file_name = get_image_id_and_file_name_by_point_id(point_id)
            if image_id in res:
                res[image_id][0]+=1
            else:
                res[image_id] = [1,file_name]

    res=[{"image_id":img_id, "file_name":val[1], "matches":int(val[0])} for img_id, val in res.items() if val[0] >= knn_min_matches]
    res.sort(key=lambda item: item["matches"],reverse=True)
    if use_smnn_matching:
        new_res = []
        target_features = torch.from_numpy(target_features).to(device)
        for item in tqdm(res):
            kpts, descs = get_kpts_and_descs_by_id(item["image_id"])
            dists, match_ids = KF.match_smnn(target_features, torch.from_numpy(descs).to(device), smnn_match_threshold)
            if len(dists) != 0:
                match_ids = match_ids.cpu()
                if use_ransac:
                    if len(dists) > 3:
                        new_res.append({"image_id":item["image_id"], "file_name":item["file_name"], "matches":verify_ransac(orig_keypoints[match_ids[:,0]],kpts[match_ids[:,1]])})
                else:
                    new_res.append({"image_id":item["image_id"],"file_name":item["file_name"], "matches":len(dists)})
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
        traceback.print_exc()
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
    except:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error in local_features_get_similar_images_by_image_buffer_handler")


@app.post("/calculate_local_features")
async def calculate_local_features_handler(image: bytes = File(...), image_id: str = Form(...)):
    try:
        global DATA_CHANGED_SINCE_LAST_SAVE, LAST_POINT_ID
        image_id = int(image_id)
        if check_if_image_id_exists(image_id):
            return Response(content="Image with the same id is already in the db", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, media_type="text/plain")
        kpts,descs = get_features(image)
        if descs is None:
            return Response(content="No descriptors for this image", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, media_type="text/plain")
        start = LAST_POINT_ID
        end = LAST_POINT_ID + len(kpts) - 1
        LAST_POINT_ID+=len(kpts)
        add_img_points(image_id,f"{image_id}.online",start,end)
        point_ids = list(range(start,end+1))
        point_ids_bytes = [int_to_bytes(x) for x in point_ids]
        add_keypoints(point_ids_bytes, kpts)
        add_descriptors(point_ids_bytes, descs)
        index.add_with_ids(descs, np.int64(point_ids))
        DATA_CHANGED_SINCE_LAST_SAVE = True
        return Response(status_code=status.HTTP_200_OK)
    except:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Can't calculate local features")


class Item_delete_local_features(BaseModel):
    image_id: Union[int ,None] = None
    file_name: Union[None,str] = None

@app.post("/delete_local_features")
async def delete_local_features_handler(item: Item_delete_local_features):
    global DATA_CHANGED_SINCE_LAST_SAVE
    try:
        if item.file_name:
            file_name = item.file_name
            point_ids = get_point_ids_by_filename(file_name)
        else:
            image_id = item.image_id
            point_ids = get_point_ids_by_image_id(image_id)
        # print(point_ids)
        if len(point_ids) != 0:
            point_ids_bytes = [int_to_bytes(x) for x in point_ids]

            if item.file_name:
                delete_img_points_by_filename(file_name)
            else:
                delete_img_points_by_image_id(image_id)

            delete_keypoints(point_ids_bytes)
            delete_descriptors(point_ids_bytes)
            index.remove_ids( np.int64(point_ids) )
            DATA_CHANGED_SINCE_LAST_SAVE = True
            return Response(status_code=status.HTTP_200_OK)
        else:
            raise HTTPException(status_code=500, detail="Image with this id is not found")
    except:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Can't delete local features")

def periodically_save_index(loop):
    global DATA_CHANGED_SINCE_LAST_SAVE, index
    if DATA_CHANGED_SINCE_LAST_SAVE:
        DATA_CHANGED_SINCE_LAST_SAVE=False
        faiss.write_index(index, "./data/populated.index")
    loop.call_later(10, periodically_save_index,loop)
    
main()