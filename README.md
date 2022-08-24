# local_features_web
Faiss + Pytorch + FastAPI + LMDB + PostgreSQL <br>
Uses HardNet8 + DoG + MAGSAC++ (OpenCV) <br>
Supported operations: add new image, delete image, find similar images by image file, find similar images by image id <br>

You should install torch yourself https://pytorch.org/get-started/locally/.  
```bash
apt install python3-dev
apt install libpq-dev
pip3 install -r requirements.txt
```

then install PostgreSQL <br>
after that:
```bash
sudo -i -u postgres
psql
ALTER USER postgres PASSWORD '12345';
```

```generate_local_features.py ./path_to_img_folder``` -> generates features  
```train.py``` -> trains index  
```add_to_index.py``` -> adds features from lmdb to index  
```local_features_web.py``` -> web microservice  
 
