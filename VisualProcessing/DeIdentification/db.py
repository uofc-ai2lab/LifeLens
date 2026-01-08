import secrets
import string
import psycopg2

DATABASE="capstone"
HOST="localhost"
PORT="5432"

# Database Connection
conn = psycopg2.connect(
    database=DATABASE,
    host=HOST,
    port=PORT
)
cursor = conn.cursor()

def generate_image_key(length=20):
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def insert_source_image(image_key, image_id, path, sha256):
    cursor.execute("""
        INSERT INTO source_images (image_key, image_id, file_path, sha256)
        VALUES (%s, %s, %s, %s)
    """, (image_key, image_id, path, sha256))
    conn.commit()
    # conn.close()

