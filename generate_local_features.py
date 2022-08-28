import cv2
from math import sqrt
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import numpy as np
from numba import jit
from numba.core import types
from numba.typed import Dict

to_tensor = ToTensor()
FIND_SPARSE_KEYPOINTS = True
N_KEYPOINTS = 200
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
    
class KeyPointSimple:
    def __init__(self, pt, size, angle, response):
        self.pt = pt
        self.size = size
        self.angle = angle
        self.response = response

def resize_img_to_threshold(img):
    width, height = img.size
    threshold = 3000*3000
    if height*width > threshold:
        k = sqrt(height*width/threshold)
        img = img.resize((round(width/k), round(height/k)),Image.Resampling.LANCZOS)
    return img

def read_img_file(f):
    img = Image.open(f)
    if img.mode != 'L':
        img = img.convert('L')
    return img

class InferenceDataset(Dataset):
        def __init__(self, images, IMAGE_PATH):
            self.images = images
            self.IMAGE_PATH = IMAGE_PATH

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            file_name = self.images[idx]
            image_id = int(file_name[:file_name.index('.')])
            img_path = self.IMAGE_PATH+"/"+file_name
            try:
                img = read_img_file(img_path)
                img = resize_img_to_threshold(img)
                img = np.array(img)
                kpts = get_keypoints(img)
                if kpts is None or len(kpts) == 0:
                    return None
                img = to_tensor(img)
                img = img.unsqueeze(0)
                kpts = [KeyPointSimple(x.pt,x.size, x.angle, x.response) for x in kpts]
                return (image_id, img, kpts)
            except:
                print(f"error reading {img_path}")

def collate_wrapper(batch):
    batch = [el for el in batch if el] #remove None
    if len(batch) == 0:
        return [],[],[]
    ids, images, kpts = zip(*batch)
    return ids, images, kpts


if __name__ == '__main__': #entry point 
    import torch
    from os import listdir
    import lmdb
    from tqdm import tqdm
    # import kornia as K
    import kornia.feature as KF
    from kornia_moons import feature
    import psycopg2
    import psycopg2.extras
    import argparse
    torch.multiprocessing.set_start_method('spawn') # to avoid problems when trying to fork process where torch is imported (CUDA problems)

    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str,nargs='?', default="/media/user/8498FD5A98FD4B66/logos/fips_images")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--prefetch_factor', type=int, default=1)

    args = parser.parse_args()
    IMAGE_PATH = args.image_path
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    PREFETCH_FACTOR = args.prefetch_factor
    
    laf_from_opencv_SIFT_kpts = feature.laf_from_opencv_SIFT_kpts
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"USING {device}")
    HardNet8 = KF.HardNet8(True).eval().to(device)

    def int_to_bytes(x: int) -> bytes:
        return x.to_bytes((x.bit_length() + 7) // 8, 'big')

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

    def check_if_exists_by_id(id):
        cur = DB_img_points.cursor()
        cur.execute("select exists(select 1 from img_points where image_id=%s)",[id])
        result = cur.fetchone()
        return result[0]

    def get_features(img, kpts):
        with torch.no_grad():
            #timg = K.image_to_tensor(img, False).float()/255.
            timg = img.to(device)
            lafs = laf_from_opencv_SIFT_kpts(kpts, device=device)
            patches = KF.extract_patches_from_pyramid(timg, lafs, 32)
            B, N, CH, H, W = patches.size()
            descs = HardNet8(patches.view(B * N, CH, H, W)).view(B * N, -1).cpu().numpy()   
        return descs

    def push_data(id_kpts_descs):
        global LAST_POINT_ID
        if len(id_kpts_descs) == 0:
            return
        img_points_data = []
        keypoints_data = []
        descriptors_data = []
        for el in id_kpts_descs:
            img_points_data.append( (el[0],f'[{LAST_POINT_ID},{LAST_POINT_ID + len(el[1])-1}]') ) # image_id point_id_start poind_id_end
            _point_id = LAST_POINT_ID
            for keypoint,descriptor in zip(el[1],el[2]):
                _point_id_bytes = int_to_bytes(_point_id)
                descriptors_data.append( (_point_id_bytes, descriptor.tobytes()) )
                keypoints_data.append( (_point_id_bytes, np.float32([keypoint.pt[0], keypoint.pt[1]]).tobytes()) )
                _point_id+=1
            LAST_POINT_ID+=len(el[1])

        print("pushing data to postgres")
        cur = DB_img_points.cursor()
        insert_query = "INSERT INTO img_points (image_id, point_id_range) VALUES %s"
        psycopg2.extras.execute_values(cur, insert_query, img_points_data, template=None, page_size=100)
        DB_img_points.commit()
        
        print("pushing data to DB_keypoints")
        with DB_keypoints.begin(write=True, buffers=True) as txn:
            with txn.cursor() as curs:
                curs.putmulti(keypoints_data)
                
        print("pushing data to DB_descriptors")
        with DB_descriptors.begin(write=True, buffers=True) as txn:
            with txn.cursor() as curs:
                curs.putmulti(descriptors_data)

    def int_from_bytes(xbytes: bytes) -> int:
        return int.from_bytes(xbytes, 'big')

    DB_img_points = prepare_db()
    DB_keypoints = lmdb.open('./keypoints.lmdb',map_size=1*1_000_000 * 1000) #1gb
    DB_descriptors = lmdb.open('./descriptors.lmdb',map_size=5*1_000_000 * 1000) #5gb
    file_names=listdir(IMAGE_PATH)
    print(f"images in {IMAGE_PATH} = {len(file_names)}")

    new_images=[]
    for file_name in tqdm(file_names):
        file_id=int(file_name[:file_name.index('.')])
        if check_if_exists_by_id(file_id):
            continue
        new_images.append(file_name)
    print(f"new images = {len(new_images)}")
    if len(new_images) == 0:
        exit()

    LAST_POINT_ID = 1
    if DB_keypoints.stat()["entries"] != DB_descriptors.stat()["entries"]:
        print('DB_keypoints.stat()["entries"] != DB_descriptors.stat()["entries"]')
        exit()

    with DB_keypoints.begin(buffers=True) as txn:
        with txn.cursor() as curs:
            for key in tqdm(curs.iternext(keys=True, values=False)):
                key = int_from_bytes(key)
                LAST_POINT_ID = max(LAST_POINT_ID,key)
                
    if LAST_POINT_ID != 1: # LAST_POINT_ID is taken by last point in db
        LAST_POINT_ID+=1

    infer_images = InferenceDataset(new_images,IMAGE_PATH)
    dataloader = torch.utils.data.DataLoader(infer_images, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, collate_fn=collate_wrapper)
    id_kpts_descs = []

    for batch_ids, batch_images, batch_kpts in tqdm(dataloader):
        if len(batch_ids) == 0 :
            continue
        for id, image, kpts in zip(batch_ids, batch_images, batch_kpts):
            descs = get_features(image, kpts)
            id_kpts_descs.append((id,kpts,descs))
        if len(id_kpts_descs)>=256:
            push_data(id_kpts_descs)
            id_kpts_descs=[]
    push_data(id_kpts_descs)