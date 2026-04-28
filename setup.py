import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED_TRAIN = 209
EXPECTED_TEST = 101


def rename_typo_folder():
    typo = ROOT / "datesets"
    correct = ROOT / "datasets"
    if typo.exists() and not correct.exists():
        print(f"Renaming {typo.name} -> {correct.name}")
        typo.rename(correct)
    elif typo.exists() and correct.exists():
        print(f"WARN: both '{typo.name}' and '{correct.name}' exist. Manual review needed.")


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
    for name in ("train.sh", "train-runpod.sh"):
        p = ROOT / name
        if p.exists():
            os.chmod(p, 0o755)


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
    rename_typo_folder()
    install_requirements()
    chmod_scripts()
    sanity_check()
    print("Setup complete.")


if __name__ == "__main__":
    main()
