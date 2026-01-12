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

# User Input
key = input("Input a distinct image key:  ")

# Query the source image data
cursor.execute("SELECT image_data FROM source_images WHERE image_key = %s", (key,))
source_binary_data = cursor.fetchone()[0]

cursor.execute("SELECT image_id FROM source_images WHERE image_key = %s", (key,))
name = cursor.fetchone()[0]

# Query the blurred image data
cursor.execute("SELECT image_data FROM source_images WHERE image_key = %s", (key,))
blurred_binary_data = cursor.fetchone()[0]

cursor.close()
conn.close()

# Save to a new file
path = f"./test_data/postgres_load/{name}"
with open(path, "wb") as f:
    f.write(source_binary_data)

# Can also open directly with Pillow
# from PIL import Image
# import io
# image = Image.open(io.BytesIO(binary_data))
# image.show()
