import os
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GDRIVE_FILE_ID = "1uCnJ3LrsBHOeQoJDoe4Yg8H32VuQJodv"
EXPECTED_TRAIN = 209
EXPECTED_TEST = 101


def download_dataset():
    datasets_dir = ROOT / "datasets"
    if datasets_dir.exists():
        print("datasets/ already exists, skipping download.")
        return

    zip_path = ROOT / "datasets.zip"
    before_dirs = set(p.name for p in ROOT.iterdir() if p.is_dir())

    print(f"Downloading from Google Drive (id={GDRIVE_FILE_ID})...")
    try:
        subprocess.run(
            ["gdown", "--id", GDRIVE_FILE_ID, "-O", str(zip_path)],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"ERROR: gdown failed ({e.returncode}). Make sure gdown is installed.")
        raise

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(ROOT)

    after_dirs = set(p.name for p in ROOT.iterdir() if p.is_dir())
    new_dirs = after_dirs - before_dirs
    for name in new_dirs:
        extracted = ROOT / name
        if (extracted / "train").is_dir() or (extracted / "test_release").is_dir():
            print(f"Renaming {name} -> datasets")
            extracted.rename(datasets_dir)
            break

    zip_path.unlink(missing_ok=True)
    print("Download complete.")


def install_requirements():
    req = ROOT / "requirements.txt"
    if not req.exists():
        return
    print("Installing requirements...")
    try:
        subprocess.run(["pip", "install", "-q", "-r", str(req)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"WARN: pip install returned {e.returncode}")


def chmod_scripts():
    for name in ("train.sh", "train-v2.sh", "train-runpod.sh"):
        p = ROOT / name
        if p.exists():
            os.chmod(p, 0o755)


def configure_git():
    print("Configuring git globals...")
    for cmd in (
        ["git", "config", "--global", "user.name", "Rayhan"],
        ["git", "config", "--global", "user.email", "rayhanhaqi@github.com"],
        ["git", "config", "--global", "credential.helper", "store"],
    ):
        subprocess.run(cmd, check=False)
    print("Git configured.")


def sanity_check():
    train_dir = ROOT / "datasets" / "train"
    test_dir = ROOT / "datasets" / "test_release"
    ids_json = ROOT / "datasets" / "test_image_name_to_ids.json"

    if train_dir.exists():
        n_train = sum(1 for p in train_dir.iterdir() if p.is_dir())
        print(f"Found {n_train} train folders (expected {EXPECTED_TRAIN}).")
    else:
        print(f"MISSING: {train_dir}")

    if test_dir.exists():
        n_test = sum(1 for p in test_dir.glob("*.tif"))
        print(f"Found {n_test} test images (expected {EXPECTED_TEST}).")
    else:
        print(f"MISSING: {test_dir}")

    if ids_json.exists():
        print(f"Found {ids_json.name}.")
    else:
        print(f"MISSING: {ids_json}")


def main():
    print("=== HW3 Setup ===")
    download_dataset()
    install_requirements()
    configure_git()
    chmod_scripts()
    sanity_check()
    print("Setup complete.")


if __name__ == "__main__":
    main()
