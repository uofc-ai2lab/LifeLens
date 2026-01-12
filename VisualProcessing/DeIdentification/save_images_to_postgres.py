import os
import hashlib
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

def create_db_tables():
    create_source_table = """
    CREATE TABLE IF NOT EXISTS source_images (
        id          SERIAL PRIMARY KEY,
        image_key   CHAR(20) UNIQUE NOT NULL,
        image_id    TEXT,
        image_data  BYTEA NOT NULL,
        sha256      TEXT,
        saved_at    TIMESTAMP DEFAULT NOW()
    );
    """

    create_deidentified_table = """
    CREATE TABLE IF NOT EXISTS deidentified_images (
        id          SERIAL PRIMARY KEY,
        image_key   CHAR(20) NOT NULL,
        image_id    TEXT,
        image_data  BYTEA NOT NULL,
        sha256      TEXT,
        saved_at    TIMESTAMP DEFAULT NOW(),
        CONSTRAINT fk_source_image_key
            FOREIGN KEY (image_key)
            REFERENCES source_images(image_key)
            ON DELETE CASCADE
    );
    """
    cursor.execute(create_source_table)
    cursor.execute(create_deidentified_table)
    conn.commit()
    print("Tables checked/created successfully.")

def generate_image_key(length=20):
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def compute_sha256(image_data):
    sha256_hash = hashlib.sha256()
    sha256_hash.update(image_data)
    return sha256_hash.hexdigest()

def save_images_to_db(source_folder, blurred_folder):
    try:
        # Get list of source images
        source_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))
        and not f.startswith('.')                     
        and f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'))]
        
        for source_file in source_files:
            source_path = os.path.join(source_folder, source_file)
            
            # Read source image data
            with open(source_path, 'rb') as f:
                source_data = f.read()
            
            # Compute SHA256
            source_sha256 = compute_sha256(source_data)
            
            # Generate unique image_key
            image_key = generate_image_key()
            
            # Insert into source_images
            insert_source = """
                INSERT INTO source_images (image_key, image_id, image_data, sha256)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(insert_source, (image_key, source_file, source_data, source_sha256))
            
            # Determine blurred filename (assuming convention: original_name_anonymized.ext)
            name, ext = os.path.splitext(source_file)
            blurred_file = f"{name}_anonymized{ext}"
            blurred_path = os.path.join(blurred_folder, blurred_file)
            
            if os.path.exists(blurred_path):
                # Read blurred image data
                with open(blurred_path, 'rb') as f:
                    blurred_data = f.read()
                
                # Compute SHA256
                blurred_sha256 = compute_sha256(blurred_data)
                
                # Insert into deidentified_images with same image_key
                insert_blurred ="""
                    INSERT INTO deidentified_images (image_key, image_id, image_data, sha256)
                    VALUES (%s, %s, %s, %s)
                """
                cursor.execute(insert_blurred, (image_key, blurred_file, blurred_data, blurred_sha256))
            else:
                print(f"Warning: Blurred image not found for {source_file}: {blurred_file}")
        
        # Commit changes
        conn.commit()
        print("Images saved successfully.")
    
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error: {e}")
    
    finally:
        if conn:
            cursor.close()
            conn.close()

def main():
    source_folder = "./test_data/input"
    blurred_folder = "./test_data/output"  
    create_db_tables()
    save_images_to_db(source_folder, blurred_folder)

if __name__ == "__main__":
    main()