import lmdb
from .byte_ops import int_from_bytes

DB_keypoints = lmdb.open('./data/keypoints.lmdb',map_size=1*1_000_000 * 1000) #1gb
DB_descriptors = lmdb.open('./data/descriptors.lmdb',map_size=5*1_000_000 * 1000) #5gb

if DB_keypoints.stat()["entries"] != DB_descriptors.stat()["entries"]:
    print('DB_keypoints.stat()["entries"] != DB_descriptors.stat()["entries"]')
    exit()

def get_dbs():
    return DB_keypoints, DB_descriptors

def get_last_point_id():
    with DB_keypoints.begin(buffers=True) as txn:
        with txn.cursor() as curs:
            curs.last()
            return int_from_bytes(curs.key())  # returns 0 if empty