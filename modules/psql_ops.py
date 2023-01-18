import psycopg2
import psycopg2.extras

def create_table(conn):
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS img_points (
            image_id integer NOT NULL UNIQUE,
            file_name text NOT NULL UNIQUE,
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

