import cv2
import numpy as np
from pathlib import Path
from cryptography.fernet import Fernet
from config.logger import Logger
from config.video_settings import IMAGE_ENC_KEY
log = Logger("[video][anonymization]")

def encrypt_image(image: np.ndarray, fernet: Fernet, suffix: str = ".jpg") -> bytes:
    """Encode an image array and encrypt it with a Fernet key.

    Args:
        image: Image data as a NumPy array.
        fernet: Fernet instance used to encrypt the encoded image.
        suffix: File extension used to select encoding format (".png" or ".jpg").

    Returns:
        str: Encrypted bytes produced by Fernet.

    Raises:
        ValueError: If the input is not a valid NumPy image or encoding fails.
    """
    if not isinstance(image, np.ndarray):
        log.error("Input must be a numpy array")
        raise ValueError("Input must be a numpy array")

    if image.size == 0:
        log.error("Image array is empty")
        raise ValueError("Image array is empty")

    try:
        img_format = ".png" if suffix == ".png" else ".jpg"

        if img_format == ".jpg":
            success, encoded = cv2.imencode(img_format, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            success, encoded = cv2.imencode(img_format, image)

        if not success:
            raise ValueError(f"Failed to encode image to {img_format}")

        encrypted = fernet.encrypt(encoded.tobytes())
        return encrypted

    except Exception as e:
        log.error(f"Encryption failed: {e}")
        raise RuntimeError("Encryption failed") from e


def run_anonymize_image(image_path: Path) -> bytes:
    """Runs image anonymization on the specified file.

    Args:
        image_path: Path to the input image file.

    Returns:
        bytes: Encrypted bytes of the encoded image.

    Raises:
        ValueError: If the encryption key is missing or invalid.
        FileNotFoundError: If the image cannot be read.
    """
    try:
        fernet = Fernet(IMAGE_ENC_KEY.encode())
    except Exception as e:
        raise ValueError("Invalid IMAGE_ENC_KEY") from e

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image '{image_path}'")

    return encrypt_image(frame, fernet, image_path.suffix.lower())