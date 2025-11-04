import os
import sys
from pathlib import Path
import dotenv
dotenv.load_dotenv()

def download_models():
    """Download pyannote models for offline use."""
    
    # Get token from environment
    token = os.getenv('HUGGING_FACE_TOKEN')
    
    if not token:
        print("=" * 70)
        print("HUGGING FACE TOKEN REQUIRED")
        print("=" * 70)
        print("\nBefore proceeding, you must:")
        print("1. Accept conditions at: https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("2. Accept conditions at: https://huggingface.co/pyannote/segmentation-3.0")
        print("3. Get your token from: https://huggingface.co/settings/tokens")
        print("\n" + "=" * 70 + "\n")
        
        sys.exit(1)
    
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub is not installed.")
        print("Install it with: pip install huggingface_hub")
        sys.exit(1)
    
    # Set cache directory based on storing location from .env
    cache_dir = Path(os.getenv('PYANNOTE_CACHE_DIR', './pyannote_models'))
    cache_dir.mkdir(exist_ok=True)
    
    print(f"\nDownloading models to: {cache_dir.absolute()}\n")
    
    models = [
        "pyannote/speaker-diarization-3.1",
        "pyannote/segmentation-3.0",
    ]
    
    for model_id in models:
        print(f"Downloading {model_id}...")
        try:
            snapshot_download(
                repo_id=model_id,
                cache_dir=str(cache_dir),
                token=token,
                resume_download=True
            )
            print(f" Successfully downloaded {model_id}\n")
        except Exception as e:
            print(f" Error downloading {model_id}: {e}")
            print("\nPossible issues:")
            print("- Invalid token")
            print("- Haven't accepted user conditions on Hugging Face")
            print("- Network connection issues")
            sys.exit(1)
    
    print("=" * 70)
    print("SUCCESS! All models downloaded.")
    print("=" * 70)
    print(f"\nModels location: {cache_dir.absolute()}")
    print("\nYou can now use pyannote offline with:")
    print(f'  pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", cache_dir="{cache_dir}")')
    print("\nFor fully offline usage, set: USE_OFFLINE_MODELS=1 in the environment file.")

if __name__ == "__main__":
    download_models()
