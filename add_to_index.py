from tqdm import tqdm
import numpy as np
import lmdb
import faiss
dim = 128
DB_descriptors = lmdb.open("descriptors.lmdb", readonly=True)
index = faiss.read_index("./trained.index")
USE_GPU = True

if USE_GPU:
    index_ivf = faiss.extract_index_ivf(index)
    quantizer = index_ivf.quantizer
    quantizer_gpu = faiss.index_cpu_to_all_gpus(quantizer)
    index_ivf.quantizer = quantizer_gpu

def int_from_bytes(xbytes: bytes) -> int:
    return int.from_bytes(xbytes, 'big')

def get_all_data_iterator(batch_size=1_000_000):
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

for ids, descriptors in tqdm(get_all_data_iterator(1_000_000)):
    index.add_with_ids(descriptors,ids)
    
if USE_GPU:
    index_ivf.quantizer = quantizer
    del quantizer_gpu 
faiss.write_index(index,"./populated.index")