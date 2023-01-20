from math import sqrt
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import numpy as np

to_tensor = ToTensor()
FIND_SPARSE_KEYPOINTS = True
N_KEYPOINTS = 200
from modules import keypoint_ops
keypoint_ops.init(FIND_SPARSE_KEYPOINTS, N_KEYPOINTS)

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
            img_path = self.IMAGE_PATH+"/"+file_name
            try:
                img = read_img_file(img_path)
                img = resize_img_to_threshold(img)
                img = np.array(img)
                kpts = keypoint_ops.get_keypoints(img)
                if kpts is None or len(kpts) == 0:
                    return None
                img = to_tensor(img)
                img = img.unsqueeze(0)
                kpts = [KeyPointSimple(x.pt, x.size, x.angle, x.response) for x in kpts]
                return (file_name, img, kpts)
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
    from tqdm import tqdm
    import kornia.feature as KF
    from kornia_moons import feature
    import psycopg2
    import psycopg2.extras

    from modules.psql_ops import prepare_db
    from modules.byte_ops import int_to_bytes
    from modules.lmdb_ops import get_dbs, get_last_point_id

    import argparse
    torch.multiprocessing.set_start_method('spawn') # to avoid problems when trying to fork process where torch is imported (CUDA problems)

    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str,nargs='?', default="./../test_images")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--prefetch_factor', type=int, default=1)
    parser.add_argument('--use_int_filenames_as_id',choices=[0,1], type=int, default=0)

    args = parser.parse_args()
    IMAGE_PATH = args.image_path
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    PREFETCH_FACTOR = args.prefetch_factor
    USE_INT_FILENAMES = args.use_int_filenames_as_id

    laf_from_opencv_SIFT_kpts = feature.laf_from_opencv_SIFT_kpts
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"USING {device}")
    HardNet8 = KF.HardNet8(True).eval().to(device)

    def check_if_exists_by_filename(file_name):
        cur = DB_img_points.cursor()
        if USE_INT_FILENAMES:
            image_id = int(file_name[:file_name.index(".")])
            cur.execute("select exists(select 1 from img_points where image_id=%s)",[image_id])
            result = cur.fetchone()
            return result[0]
        else:
            cur.execute("select exists(select 1 from img_points where file_name=%s)",[file_name])
            result = cur.fetchone()
            return result[0]

    def get_last_image_id():
        cur = DB_img_points.cursor()
        cur.execute("select max(image_id) from img_points")
        result = cur.fetchone()
        return result[0]

    def get_features(img, kpts):
        with torch.no_grad():
            #timg = K.image_to_tensor(img, False).float()/255.
            img = img.to(device)
            lafs = laf_from_opencv_SIFT_kpts(kpts, device=device)
            patches = KF.extract_patches_from_pyramid(img, lafs, 32)
            B, N, CH, H, W = patches.size()
            descs = HardNet8(patches.view(B * N, CH, H, W)).view(B * N, -1).cpu().numpy()   
        return descs

    def push_data(id_kpts_descs):
        global LAST_POINT_ID, LAST_IMAGE_ID
        if len(id_kpts_descs) == 0:
            return
        img_points_data = []
        keypoints_data = []
        descriptors_data = []
        for el in id_kpts_descs:
            file_name = el[0]

            if USE_INT_FILENAMES:
                image_id = int(el[0][:el[0].index(".")])
            else:
                image_id = LAST_IMAGE_ID
                LAST_IMAGE_ID+=1

            img_points_data.append( (image_id, file_name, f'[{LAST_POINT_ID},{LAST_POINT_ID + len(el[1])-1}]') ) # image_id point_id_start poind_id_end
            _point_id = LAST_POINT_ID
            for keypoint,descriptor in zip(el[1],el[2]):
                _point_id_bytes = int_to_bytes(_point_id)
                descriptors_data.append( (_point_id_bytes, descriptor.tobytes()) )
                keypoints_data.append( (_point_id_bytes, np.float32([keypoint.pt[0], keypoint.pt[1]]).tobytes()) )
                _point_id+=1
            LAST_POINT_ID+=len(el[1])

        print("pushing data to postgres")
        cur = DB_img_points.cursor()
        insert_query = "INSERT INTO img_points (image_id, file_name, point_id_range) VALUES %s"
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

    DB_img_points = prepare_db()
    DB_keypoints, DB_descriptors = get_dbs()

    file_names=listdir(IMAGE_PATH)
    print(f"images in {IMAGE_PATH} = {len(file_names)}")
    new_images=[]
    for file_name in tqdm(file_names):
        if check_if_exists_by_filename(file_name):
            continue
        new_images.append(file_name)
    print(f"new images = {len(new_images)}")
    if len(new_images) == 0:
        exit()

    if USE_INT_FILENAMES == 0:
        LAST_IMAGE_ID = get_last_image_id()
        if LAST_IMAGE_ID:
            LAST_IMAGE_ID+=1
        else:
            LAST_IMAGE_ID=1
    
    LAST_POINT_ID = get_last_point_id()+1

    infer_images = InferenceDataset(new_images,IMAGE_PATH)
    dataloader = torch.utils.data.DataLoader(infer_images, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, collate_fn=collate_wrapper)
    id_kpts_descs = []

    for batch_ids, batch_images, batch_kpts in tqdm(dataloader):
        if len(batch_ids) == 0 :
            continue
        for id, image, kpts in zip(batch_ids, batch_images, batch_kpts):
            descs = get_features(image, kpts)
            id_kpts_descs.append((id,kpts,descs))
        if len(id_kpts_descs)>=256:    #push with big batches, to increase throughput
            push_data(id_kpts_descs)
            id_kpts_descs=[]
    push_data(id_kpts_descs)