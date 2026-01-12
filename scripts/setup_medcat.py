from pathlib import Path
import sys, time
import urllib.request

MEDCAT_DATA_DIR = Path("data/data_p3.2")
MEDCAT_DATA_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "medmen_wstatus_2021_oct.zip":
        "https://cogstack-medcat-example-models.s3.eu-west-2.amazonaws.com/medcat-example-models/medmen_wstatus_2021_oct.zip",
    "pt_notes.csv":
        "https://raw.githubusercontent.com/CogStack/MedCATtutorials/main/notebooks/introductory/data/pt_notes.csv",
}

def download(url, dest):
    if dest.exists():
        print(f"[SKIP] {dest.name} already exists")
        return

    print(f"[DOWNLOADING] {dest.name}")

    start_time = time.time()
    last_time = start_time

    def report(block_num, block_size, total_size):
        nonlocal last_time

        downloaded = block_num * block_size
        elapsed = time.time() - start_time

        if elapsed == 0:
            return

        speed = downloaded / elapsed  # bytes/sec
        percent = min(downloaded / total_size, 1.0)

        bar_len = 40
        filled = int(bar_len * percent)
        bar = "=" * filled + " " * (bar_len - filled)

        def fmt_bytes(n):
            for unit in ["B", "KB", "MB", "GB"]:
                if n < 1024:
                    return f"{n:.2f}{unit}"
                n /= 1024
            return f"{n:.2f}TB"

        remaining = total_size - downloaded
        eta = remaining / speed if speed > 0 else 0
        mins, secs = divmod(int(eta), 60)

        sys.stdout.write(
            f"\r{dest.name[:25]:25} "
            f"{int(percent * 100):3}%"
            f"[{bar}] "
            f"{fmt_bytes(downloaded):>8} "
            f"{fmt_bytes(speed)}/s "
            f"ETA {mins}m {secs:02d}s"
        )
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest, reporthook=report)
        print(f"\n[DONE] {dest.name}")
    except KeyboardInterrupt:
        print("\n[ABORTED] Download cancelled by user.")
        if dest.exists():
            dest.unlink()
            
def main():
    print("Setting up MedCAT resources...\n")
    for name, url in FILES.items():
        download(url, MEDCAT_DATA_DIR / name)
    print("\nMedCAT setup complete.")

if __name__ == "__main__":
    main()