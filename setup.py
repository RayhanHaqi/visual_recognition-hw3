import os
import subprocess
import tarfile
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

    archive_path = ROOT / "datasets.tar"
    if not archive_path.exists():
        print(f"Downloading from Google Drive (id={GDRIVE_FILE_ID})...")
        try:
            subprocess.run(
                ["gdown", f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", "-O", str(archive_path)],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"ERROR: gdown failed ({e.returncode}). Make sure gdown is installed.")
            raise
    else:
        print("Archive already downloaded, skipping.")

    tmp_dir = ROOT / "datasets_tmp"
    if tmp_dir.exists():
        import shutil
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()

    print("Extracting...")
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(tmp_dir)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as t:
            t.extractall(tmp_dir)
    else:
        tmp_dir.rmdir()
        archive_path.unlink(missing_ok=True)
        print(f"ERROR: unknown archive format, deleted bad archive. Re-run setup.")
        raise SystemExit(1)

    train_src = tmp_dir / "train"
    test_src = tmp_dir / "test_release"
    ids_src = tmp_dir / "test_image_name_to_ids.json"

    for child in tmp_dir.iterdir():
        d = child if child.is_dir() else child.parent
        if (tmp_dir / child.name / "train").is_dir():
            train_src = tmp_dir / child.name / "train"
            test_src = tmp_dir / child.name / "test_release"
            ids_src = tmp_dir / child.name / "test_image_name_to_ids.json"
            break

    if not train_src.is_dir():
        print("ERROR: could not find train/ in extracted archive")
        import shutil
        shutil.rmtree(tmp_dir)
        raise SystemExit(1)

    datasets_dir.mkdir()
    train_src.rename(datasets_dir / "train")
    test_src.rename(datasets_dir / "test_release")
    if ids_src.exists():
        ids_src.rename(datasets_dir / "test_image_name_to_ids.json")
    import shutil
    shutil.rmtree(tmp_dir)
    archive_path.unlink(missing_ok=True)
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
    for name in ("train.sh", "train-v2.sh", "train-runpod.sh", "train-runpod-v2.sh"):
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

    print("Setting up git LFS for submission ZIPs...")
    subprocess.run(["apt-get", "update", "-qq"], check=False)
    subprocess.run(["apt-get", "install", "-qq", "-y", "git-lfs"], check=False)
    subprocess.run(["git", "lfs", "install"], cwd=str(ROOT), check=False)
    subprocess.run(["git", "lfs", "track", "submission/*.zip"], cwd=str(ROOT), check=False)

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
    install_requirements()
    download_dataset()
    configure_git()
    chmod_scripts()
    sanity_check()
    print("Setup complete.")


if __name__ == "__main__":
    main()
