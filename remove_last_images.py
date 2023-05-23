import psycopg2
import lmdb
from tqdm import tqdm

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
        conn.autocommit = True
        cur.execute('CREATE DATABASE ambience')
        conn = psycopg2.connect(connect_settings_ambience)
        create_table(conn)
        return conn
    else:
        conn = psycopg2.connect(connect_settings_ambience)
        create_table(conn)
        return conn

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


DB_img_points = prepare_db()
DB_keypoints = lmdb.open('./data/keypoints.lmdb',map_size=10 * 1000 * 1_000_000) #6gb
DB_descriptors = lmdb.open('./data/descriptors.lmdb',map_size=120 * 1000 * 1_000_000) #120gb

def get_point_ids(image_id):
    cursor = DB_img_points.cursor()
    cursor.execute("SELECT point_id_range FROM img_points WHERE image_id = %s",[image_id])
    result = cursor.fetchone()
    if result is None:
        return []
    else:
        return list(range(result[0].lower,result[0].upper))

def get_last_inserted_image():
    cursor = DB_img_points.cursor()
    cursor.execute("SELECT MAX(lower(point_id_range)) FROM img_points")
    result = cursor.fetchone()
    last_image_id = result[0]
    cursor.execute("SELECT image_id FROM (SELECT image_id,MAX(lower(point_id_range)) as mmax FROM img_points GROUP BY image_id )t1 WHERE mmax = %s",[last_image_id])
    result = cursor.fetchone()
    return result[0]

def int_to_bytes(x: int) -> bytes:
    return x.to_bytes((x.bit_length() + 7) // 8, 'big')

# print(get_last_inserted_image())

for i in tqdm(range(11000)):
    image_id = get_last_inserted_image()
    point_ids = get_point_ids(image_id)
    point_ids_bytes = [int_to_bytes(x) for x in point_ids]
    delete_img_points(image_id)
    delete_keypoints(point_ids_bytes)
    delete_descriptors(point_ids_bytes)