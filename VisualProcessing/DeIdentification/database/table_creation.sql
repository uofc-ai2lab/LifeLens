-- Initial Input Frames
CREATE TABLE source_images (
    id SERIAL PRIMARY KEY,
    image_key CHAR(20) UNIQUE NOT NULL,
    image_id TEXT,
    image_data BYTEA NOT NULL,
    sha256 TEXT,
    saved_at TIMESTAMP DEFAULT NOW()
);

-- DeIdentified Frames
CREATE TABLE deidentified_images (
    id SERIAL PRIMARY KEY,
    image_key CHAR(20) NOT NULL,
    image_id TEXT,
    image_data BYTEA NOT NULL,
    saved_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (image_key) REFERENCES source_images(image_key)
);

-- drop table source_images, deidentified_images
