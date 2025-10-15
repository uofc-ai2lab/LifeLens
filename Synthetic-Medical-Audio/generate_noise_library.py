#!/usr/bin/env python3
"""
Downloads realistic EMS background noises (especially helicopter blades)
from Freesound.org using the Freesound API.

Outputs a structured noise library for use in synthetic medical audio generation.

Requirements:
    pip install requests pydub

Before running:
    1. Get a Freesound API key: https://freesound.org/help/developers/
    2. Set it as an environment variable or add it to an .env file: FREESOUND_API_KEY=your_key_here
"""

import os
import time
import requests
from pathlib import Path
from urllib.parse import quote
from pydub import AudioSegment


# Configuration
FREESOUND_API_KEY = os.getenv("FREESOUND_API_KEY")
if not FREESOUND_API_KEY:
    raise EnvironmentError(
        "Please set your Freesound API key as FREESOUND_API_KEY environment variable."
    )

BASE_URL = "https://freesound.org/apiv2"
OUTPUT_DIR = Path("noise_library")
MAX_RESULTS_PER_QUERY = 10  # Avoid overwhelming rate limits
MIN_DURATION = 5.0          # Seconds
MAX_DURATION = 60.0         # Seconds


def search_and_download(query: str, subdir: str, license_filter: str = "creative commons") -> None:
    """
    Search Freesound for a query, filter by license/duration, and download WAV files.
    """
    out_path = OUTPUT_DIR / subdir
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\n🔍 Searching for: '{query}' → {out_path}")

    params = {
        "query": quote(query),
        # "filter": f"duration:[{MIN_DURATION} TO {MAX_DURATION}]"",
        "fields": "id,name,duration,previews,license",
        # "page_size": MAX_RESULTS_PER_QUERY,
    }

    # headers
    headers = {
        "Authorization": f"Token {FREESOUND_API_KEY}"
    }

    try:
        response = requests.get(f"{BASE_URL}/search/text/", params=params, headers=headers)
        print(f"  🔗 Request URL: {response.url}")
        response.raise_for_status()
        results = response.json()
        print(results)
    except requests.RequestException as e:
        print(f"❌ API request failed: {e}")
        return

    downloaded = 0
    for sound in results.get("results", []):
        if downloaded >= MAX_RESULTS_PER_QUERY:
            break

        preview_url = sound.get("previews", {}).get("preview-lq-mp3")
        if not preview_url:
            continue

        # Derive safe filename
        name = "".join(c if c.isalnum() else "_" for c in sound["name"])
        filepath = out_path / f"{name}.mp3"

        if filepath.exists():
            print(f"  ⏩ Already exists: {filepath.name}")
            downloaded += 1
            continue

        try:
            print(f"  ⬇️  Downloading: {sound['name']} ({sound['duration']:.1f}s)")
            audio_resp = requests.get(preview_url)
            audio_resp.raise_for_status()

            filepath.write_bytes(audio_resp.content)

            # Convert to WAV (required by audiomentations)
            wav_path = filepath.with_suffix(".wav")
            audio = AudioSegment.from_mp3(filepath)
            audio = audio.set_frame_rate(16000).set_channels(1)  # Mono, 16kHz
            audio.export(wav_path, format="wav")
            filepath.unlink()  # Remove MP3

            downloaded += 1
            time.sleep(0.5)  # Be kind to the API

        except Exception as e:
            print(f"    ❌ Failed to process {sound['name']}: {e}")

    print(f"✅ Downloaded {downloaded} files to {subdir}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Prioritize helicopter sounds (critical for air-med scenarios)
    search_and_download("helicopterRaw_30sec", "helicopter")
    search_and_download("air ambulance", "helicopter")

    # Other EMS-relevant noises
    search_and_download("ambulance siren", "sirens")
    search_and_download("sw_DP2011_HornsSiren_1", "traffic")
    search_and_download("crowd emergency", "crowd")
    search_and_download("emergency room hospital", "equipment")
    search_and_download("oxygen machine", "equipment")

    print(f"\nNoise library ready at: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()