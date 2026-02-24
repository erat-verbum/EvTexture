import os
import requests
from pathlib import Path
from tqdm import tqdm

MODEL_URLS = {
    "REDS": "https://github.com/DachunKai/EvTexture/releases/download/v0.0/EvTexture_REDS_BIx4.pth",
    "Vimeo90K": "https://github.com/DachunKai/EvTexture/releases/download/v0.0/EvTexture_Vimeo90K_BIx4.pth",
}

MODEL_FILENAMES = {
    "REDS": "EvTexture_REDS_BIx4.pth",
    "Vimeo90K": "EvTexture_Vimeo90K_BIx4.pth",
}

EXPECTED_SIZES = {
    "REDS": 71_518_689,
    "Vimeo90K": 71_518_689,
}


def get_model_dir():
    repo_root = Path(__file__).parent.parent
    model_dir = repo_root / "experiments" / "pretrained_models" / "EvTexture"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def get_model_path(model_name: str) -> Path:
    model_dir = get_model_dir()
    return model_dir / MODEL_FILENAMES[model_name]


def download_model(model_name: str, force: bool = False, verbose: bool = True) -> Path:
    model_path = get_model_path(model_name)

    if model_path.exists() and not force:
        if verbose:
            print(f"Model already exists at: {model_path}")
        return model_path

    url = MODEL_URLS[model_name]
    expected_size = EXPECTED_SIZES[model_name]

    if verbose:
        print(f"Downloading {model_name} model from: {url}")
        print(
            f"This will download approximately {expected_size / (1024 * 1024):.1f} MB"
        )

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with (
        open(model_path, "wb") as f,
        tqdm(
            desc=f"Downloading {model_name}",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    actual_size = model_path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"Downloaded file size {actual_size} does not match expected {expected_size}. "
            f"Download may be corrupted."
        )

    if verbose:
        print(f"Model saved to: {model_path}")

    return model_path


def ensure_model(
    model_name: str, auto_download: bool = False, verbose: bool = True
) -> Path:
    model_path = get_model_path(model_name)

    if model_path.exists():
        return model_path

    if auto_download:
        return download_model(model_name, verbose=verbose)

    url = MODEL_URLS[model_name]
    raise FileNotFoundError(
        f"Model file not found: {model_path}\n"
        f"To download automatically, run with --download flag\n"
        f"Or manually download from: {url}\n"
        f"and place it at: {model_path}"
    )
