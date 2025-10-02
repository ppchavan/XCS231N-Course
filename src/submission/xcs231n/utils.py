import os
import sys
import subprocess

def download_from_gdrive(target_dir, model_weight, url):

    # Ensure gdown is available
    try:
        import gdown  # type: ignore
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown  # type: ignore

    os.makedirs(target_dir, exist_ok=True)

    output_path = os.path.join(target_dir, model_weight)

    # Only download if file doesn't already exist
    if not os.path.exists(output_path):
        # Download (fuzzy=True lets gdown parse various Drive URL formats)
        print(f"Downloading from: {url}")
        gdown.download(url=url, output=output_path, quiet=False, fuzzy=True)

        # Verify
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"Saved to {output_path} ({os.path.getsize(output_path)} bytes)")
        else:
            raise RuntimeError("Download failed or produced an empty file.")
    else:
        print(f"Model file already exists at {output_path}")