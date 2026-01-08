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

import hashlib
from psycopg2 import Binary
from pathlib import Path

def insert_source_image(image_key, image_id=None, path=None, image_bytes=None, sha256=None):
    """Insert an image into `source_images`.

    Provide either `path` or `image_bytes`. If `sha256` is not provided it will be computed.
    Returns the inserted row id.
    """
    if image_bytes is None and path is None:
        raise ValueError("Provide either `path` or `image_bytes`")

    if image_bytes is None:
        with open(path, "rb") as f:
            image_bytes = f.read()

    if sha256 is None:
        sha256 = hashlib.sha256(image_bytes).hexdigest()

    cursor.execute(
        """
        INSERT INTO source_images (image_key, image_id, image_data, sha256)
        VALUES (%s, %s, %s, %s)
        RETURNING id
        """,
        (image_key, image_id, Binary(image_bytes), sha256)
    )
    conn.commit()
    return cursor.fetchone()[0]


def fetch_image_bytes_by_key(image_key):
    """Return a dict with `image_id`, `image_bytes`, `sha256`, `saved_at` for `image_key` or None."""
    cursor.execute(
        "SELECT image_id, image_data, sha256, saved_at FROM source_images WHERE image_key = %s",
        (image_key,)
    )
    row = cursor.fetchone()
    if not row:
        return None
    image_id, image_data, sha256, saved_at = row
    return {"image_id": image_id, "image_bytes": bytes(image_data), "sha256": sha256, "saved_at": saved_at}


def write_image_to_file(image_key, out_path):
    data = fetch_image_bytes_by_key(image_key)
    if data is None:
        raise KeyError(f"image_key {image_key} not found")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(data["image_bytes"])
    return str(out_path)


def list_source_images(limit=100):
    cursor.execute("SELECT image_key, image_id, sha256, saved_at FROM source_images ORDER BY saved_at DESC LIMIT %s", (limit,))
    return cursor.fetchall()


def table_exists(table_name):
    """Return True if `public.table_name` exists."""
    cursor.execute("SELECT to_regclass(%s)", (f"public.{table_name}",))
    res = cursor.fetchone()
    return res and res[0] is not None


def create_tables_from_sql(sql_path):
    """Execute the SQL statements in `sql_path` to create tables.

    This safely splits statements so multiple semicolon-separated statements are executed.
    """
    sql_path = Path(sql_path)
    if not sql_path.exists():
        raise FileNotFoundError(sql_path)
    sql = sql_path.read_text()
    # Execute each statement individually to avoid issues with multi-statement execution
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    for stmt in statements:
        cursor.execute(stmt)
    conn.commit()


def ensure_tables_exist(sql_path="database/table_creation.sql"):
    """Check for required tables and create them using `sql_path` if any are missing.

    Returns list of created table names (or empty list if nothing was created).
    """
    required = ["source_images", "deidentified_images"]
    missing = [t for t in required if not table_exists(t)]
    if not missing:
        return []
    # Create tables
    create_tables_from_sql(sql_path)
    return missing

