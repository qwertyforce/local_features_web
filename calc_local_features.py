import cv2
import torch
from os import listdir
import math
import lmdb
from tqdm import tqdm
import numpy as np
import kornia as K
import kornia.feature as KF
from kornia_moons import feature
laf_from_opencv_SIFT_kpts = feature.laf_from_opencv_SIFT_kpts

detector = cv2.SIFT_create(nfeatures=200)
device = "cuda" if torch.cuda.is_available() else "cpu"
HardNet8 = KF.HardNet8(True).eval().to(device)

import psycopg2
import psycopg2.extras


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
DB_keypoints = lmdb.open('./keypoints.lmdb',map_size=150*1_000_000 * 1000) #500gb
DB_descriptors = lmdb.open('./descriptors.lmdb',map_size=150*1_000_000 * 1000) #500gb


def check_if_exists_by_id(id):
    cur = DB_img_points.cursor()
    cur.execute("select exists(select 1 from img_points where image_id=%s)",[id])
    result = cur.fetchone()
    return result[0]

def resize_img_to_threshold(img):
    height, width = img.shape
    threshold = 3000*3000
    if height*width > threshold:
        k = math.sqrt(height*width/(threshold))
        img = cv2.resize(img, (round(width/k), round(height/k)), interpolation=cv2.INTER_LINEAR)
    return img

def get_features(img):
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
    return kpts, descs


def calc_features(file_name):
    try:
        file_id=int(file_name[:file_name.index('.')])
        img_path=IMAGE_PATH+"/"+file_name
        query_image=cv2.imread(img_path,0)
        if query_image is None:
            print(f"can't read {file_name}")
            return None
        img_keypoints_decscs = get_features(query_image)
        if img_keypoints_decscs is None:
            return None    
        kpts,descs = img_keypoints_decscs
        # print(file_name)
        return (file_id,kpts,descs)
    except:
        print(f"error in {file_name}")
        return None

def int_to_bytes(x: int) -> bytes:
    return x.to_bytes((x.bit_length() + 7) // 8, 'big')


IMAGE_PATH="./../images"
file_names=listdir(IMAGE_PATH)
print(f"images in {IMAGE_PATH} = {len(file_names)}")
new_images=[]

for file_name in tqdm(file_names):
    file_id=int(file_name[:file_name.index('.')])
    if check_if_exists_by_id(file_id):
        continue
    new_images.append(file_name)

print(f"new images = {len(new_images)}")
new_images=[new_images[i:i + 10000] for i in range(0, len(new_images), 10000)]

LAST_POINT_ID = 1
for batch in tqdm(new_images):
    data=[calc_features(file_name) for file_name in tqdm(batch)]
    data= [i for i in data if i] #remove None's
    print("pushing data to db")
    img_points_data = []
    keypoints_data = []
    descriptors_data = []
    for el in data:
        img_points_data.append( (el[0],f'[{LAST_POINT_ID},{LAST_POINT_ID + len(el[1])}]') ) # image_id point_id_start poind_id_end
        _point_id = LAST_POINT_ID
        for keypoint,descriptor in zip(el[1],el[2]):
            _point_id_bytes = int_to_bytes(_point_id)
            descriptors_data.append( (_point_id_bytes, descriptor.tobytes()) )
            keypoints_data.append( (_point_id_bytes, np.float32([keypoint.pt[0], keypoint.pt[1]]).tobytes()) )
            _point_id+=1

        LAST_POINT_ID+=len(el[1])+1
    
    cur = DB_img_points.cursor()
    insert_query = "INSERT INTO img_points (image_id, point_id_range) VALUES %s"
    psycopg2.extras.execute_values(cur, insert_query, img_points_data, template=None, page_size=100)
    DB_img_points.commit()

    with DB_keypoints.begin(write=True, buffers=True) as txn:
        with txn.cursor() as curs:
            curs.putmulti(keypoints_data)

    with DB_descriptors.begin(write=True, buffers=True) as txn:
        with txn.cursor() as curs:
            curs.putmulti(descriptors_data)
    