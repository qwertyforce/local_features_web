from tqdm import tqdm
import numpy as np
import lmdb
import faiss
dim = 128
DB_descriptors = lmdb.open("descriptors.lmdb", readonly=True)
index = faiss.read_index("./trained.index")

def int_from_bytes(xbytes: bytes) -> int:
    return int.from_bytes(xbytes, 'big')

def get_all_data_iterator(batch_size=1000000):
    with DB_descriptors.begin(buffers=True) as txn:
        with txn.cursor() as curs:
            temp_ids = np.zeros(batch_size,np.int64)
            temp_descriptors = np.zeros((batch_size,dim),np.float32)
            retrieved = 0
            for data in curs.iternext(keys=True, values=True):
                temp_ids[retrieved] = int_from_bytes(data[0])
                temp_descriptors[retrieved] = np.frombuffer(data[1],dtype=np.float32)
                retrieved+=1
                if retrieved == batch_size:
                    retrieved=0
                    yield temp_ids, temp_descriptors
            if retrieved != 0:
                yield temp_ids[:retrieved], temp_descriptors[:retrieved]

for ids, descriptors in tqdm(get_all_data_iterator(1000000)):
    index.add_with_ids(descriptors,ids)
faiss.write_index(index,"./populated.index")