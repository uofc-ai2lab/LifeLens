from __future__ import annotations

import io
# import os
import secrets
import string
import typing as t
# from datetime import datetime
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor
from PIL import Image

# --- Configuration helpers -------------------------------------------------

DEFAULT_KEY_LENGTH = 20
MAX_KEY_GENERATION_ATTEMPTS = 10
DATABASE="capstone"
HOST="localhost"
PORT="5432"


# def get_db_params_from_env() -> dict:
#     """Gather DB connection params from environment variables."""
#     return {
#         "host": os.environ.get("PGHOST", "localhost"),
#         "port": int(os.environ.get("PGPORT", 5432)),
#         "dbname": os.environ.get("PGDATABASE", "postgres"),
#         "user": os.environ.get("PGUSER", "postgres"),
#         "password": os.environ.get("PGPASSWORD", ""),
#     }


# --- Table creation --------------------------------------------------------

CREATE_BLURRED_IMAGES_SQL = """
CREATE TABLE IF NOT EXISTS blurred_images (
    key CHAR(20) PRIMARY KEY,
    filename TEXT,
    mime_type TEXT,
    width INTEGER,
    height INTEGER,
    image BYTEA NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);
"""

CREATE_SOURCE_IMAGES_SQL = """
CREATE TABLE IF NOT EXISTS source_images (
    id SERIAL PRIMARY KEY,
    blurred_key CHAR(20) NOT NULL REFERENCES blurred_images(key) ON DELETE CASCADE,
    filename TEXT,
    mime_type TEXT,
    width INTEGER,
    height INTEGER,
    image BYTEA NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);
"""


def create_tables(conn: psycopg2.extensions.connection) -> None:
    """Create tables if they don't exist."""
    with conn.cursor() as cur:
        cur.execute(CREATE_BLURRED_IMAGES_SQL)
        cur.execute(CREATE_SOURCE_IMAGES_SQL)
    conn.commit()


# --- Key generation --------------------------------------------------------


def _random_key(length: int = DEFAULT_KEY_LENGTH) -> str:
    alphabet = string.ascii_letters
    return "".join(secrets.choice(alphabet) for _ in range(length))


def generate_unique_blurred_key(conn: psycopg2.extensions.connection, length: int = DEFAULT_KEY_LENGTH) -> str:
    """Generate a unique random key and ensure it doesn't already exist in DB."""
    for attempt in range(MAX_KEY_GENERATION_ATTEMPTS):
        key = _random_key(length)
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM blurred_images WHERE key = %s", (key,))
            if cur.fetchone() is None:
                return key
    raise RuntimeError("Unable to generate a unique key after several attempts")


# --- Image helpers --------------------------------------------------------


def image_to_bytes(img: Image.Image, fmt: str | None = None) -> bytes:
    """Encode PIL Image to bytes. If fmt is None, use PNG."""
    bio = io.BytesIO()
    fmt = fmt or "PNG"
    img.save(bio, format=fmt)
    return bio.getvalue()


def read_file_bytes(path: Path) -> bytes:
    return path.read_bytes()


def get_image_size_from_bytes(raw: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(raw)) as im:
        return im.width, im.height


# --- Save / retrieve operations -------------------------------------------


def save_blurred_image(conn: psycopg2.extensions.connection, filename: str, image_bytes: bytes, mime_type: str | None = None) -> str:
    """Save a blurred image and return the generated 20-letter key."""
    key = generate_unique_blurred_key(conn)
    width, height = get_image_size_from_bytes(image_bytes)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO blurred_images (key, filename, mime_type, width, height, image) VALUES (%s, %s, %s, %s, %s, %s)",
            (key, filename, mime_type, width, height, psycopg2.Binary(image_bytes)),
        )
    conn.commit()
    return key


def save_source_image(conn: psycopg2.extensions.connection, blurred_key: str, filename: str, image_bytes: bytes, mime_type: str | None = None) -> int:
    """Save a source image that references a blurred image by key. Returns the new source image id."""
    width, height = get_image_size_from_bytes(image_bytes)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO source_images (blurred_key, filename, mime_type, width, height, image) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
            (blurred_key, filename, mime_type, width, height, psycopg2.Binary(image_bytes)),
        )
        new_id = cur.fetchone()[0]
    conn.commit()
    return new_id


def get_blurred_image(conn: psycopg2.extensions.connection, key: str) -> dict | None:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT key, filename, mime_type, width, height, image, created_at FROM blurred_images WHERE key = %s", (key,))
        row = cur.fetchone()
        if row is None:
            return None
        # convert memoryview to bytes if needed
        if isinstance(row["image"], memoryview):
            row["image"] = bytes(row["image"])
        return dict(row)


def list_blurred_images(conn: psycopg2.extensions.connection) -> list[dict]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT key, filename, mime_type, width, height, created_at FROM blurred_images ORDER BY created_at DESC LIMIT 100")
        return [dict(r) for r in cur.fetchall()]



def _connect() -> psycopg2.extensions.connection:
    # params = get_db_params_from_env()
    return psycopg2.connect(
        database=DATABASE,
        host=HOST,
        port=PORT
    )


def main():
    project_root = Path(__file__).parent
    sample_dir = project_root / "test_data" / "input"
    if not sample_dir.exists():
        print("No sample images found under test_data/input/. Create some to run the example.")
        return

    candidates = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
    if not candidates:
        print("No .jpg/.png images found in test_data/input/ to use for example.")
        return

    sample = candidates[0]
    sample_bytes = read_file_bytes(sample)

    with _connect() as conn:
        # create_tables(conn)
        print("Tables ensured.")
        key = save_blurred_image(conn, sample.name, sample_bytes, mime_type=None)
        print(f"Saved blurred image with key: {key}")
        src_id = save_source_image(conn, key, f"src_{sample.name}", sample_bytes, mime_type=None)
        print(f"Saved source image with id: {src_id}")

        row = get_blurred_image(conn, key)
        if row:
            print(f"Retrieved blurred image '{row['filename']}' ({row['width']}x{row['height']})")


if __name__ == "__main__":
    main()
