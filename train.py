from tqdm import tqdm
import numpy as np
import lmdb
import math
import faiss
from os import remove

dim = 128
DB_descriptors = lmdb.open("./data/descriptors.lmdb", readonly=True)
entries = DB_descriptors.stat()['entries']
nlist = int(math.sqrt(entries))
print(f"entries = {entries}")
print(f"nlist = {nlist}")
index = faiss.index_factory(dim,f"OPQ64,IVF{nlist},PQ64",faiss.METRIC_L2)
batch_size= min(entries,50_000_000)

features = np.memmap('train.mmap', dtype='float32', mode='w+', shape=(batch_size, dim))
#features = np.zeros((batch_size, dim),dtype='float32')
def get_data():
    with DB_descriptors.begin(buffers=True) as txn:
        with txn.cursor() as curs:
            retrieved = 0
            for data in tqdm(curs.iternext(keys=False, values=True),total=batch_size):
                if retrieved == batch_size:
                    return
                features[retrieved] = np.frombuffer(data,dtype=np.float32)
                retrieved+=1
get_data()
print("Training......")
from timeit import default_timer as timer
start = timer()
index.train(features)
end = timer()
print(end - start)
del features
remove("train.mmap")
faiss.write_index(index,"./data/trained.index")