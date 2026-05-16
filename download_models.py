"""
Run this ONCE before starting app.py to pre-download all required models.
Handles flaky connections with automatic resume and retry.

Usage:
    python download_models.py
"""

import time
from huggingface_hub import snapshot_download

MODELS = [
    ("microsoft/Florence-2-large",           "Florence-2 vision model"),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0",  "TinyLlama text model"),
]

MAX_RETRIES = 10
RETRY_DELAY = 5  # seconds


def download_with_retry(repo_id: str, label: str):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"\n[{label}] Download attempt {attempt}/{MAX_RETRIES}...")
            snapshot_download(
                repo_id=repo_id,
                resume_download=True,   # picks up from where it left off
            )
            print(f"[{label}] Done.")
            return True
        except Exception as e:
            print(f"[{label}] Attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                print(f"  Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"[{label}] All {MAX_RETRIES} attempts failed. Check your connection.")
                return False


if __name__ == "__main__":
    print("=" * 55)
    print("  Model Pre-Downloader")
    print("  Run this before app.py to avoid mid-request freezes")
    print("=" * 55)

    all_ok = True
    for repo_id, label in MODELS:
        ok = download_with_retry(repo_id, label)
        if not ok:
            all_ok = False

    print("\n" + "=" * 55)
    if all_ok:
        print("  All models cached. You can now run: python app.py")
    else:
        print("  Some downloads failed. Re-run this script to retry.")
    print("=" * 55)
