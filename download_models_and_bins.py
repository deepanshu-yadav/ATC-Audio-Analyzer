"""
Setup and asset downloader for CrispASR ATC Transcription Pipeline.

Downloads:
1. CrispASR binaries (CPU, CUDA, Vulkan, etc.) for Windows and Linux.
2. Speech recognition model (Parakeet-v3 GGUF for ATC).
3. Silero Voice Activity Detection (VAD) model.
"""

import os
import sys
import shutil
import zipfile
import tarfile
import urllib.request
import argparse
from pathlib import Path

CRISPASR_VERSION = "v0.8.39"
GITHUB_RELEASE_BASE = f"https://github.com/CrispStrobe/CrispASR/releases/download/{CRISPASR_VERSION}"

# Model URLs
PARAKEET_MODEL_URL = "https://huggingface.co/pronoobie/Parakeet-v3-For_ATC/resolve/main/speech-model.gguf"
SILERO_VAD_URL = "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin"

# Mapping of platform & device options to CrispASR release archives
BINARY_TARGETS = {
    "windows": {
        "cpu": {
            "archive": "crispasr-windows-x86_64-cpu.zip",
            "dest_subdir": "cpu",
            "description": "Windows x86_64 CPU (Modern AVX/AVX2)",
        },
        "cpu-legacy": {
            "archive": "crispasr-windows-x86_64-cpu-legacy.zip",
            "dest_subdir": "cpu",
            "description": "Windows x86_64 CPU Legacy (older processors)",
        },
        "cuda": {
            "archive": "crispasr-windows-x86_64-cuda.zip",
            "dest_subdir": "cuda",
            "description": "Windows x86_64 CUDA (NVIDIA GPU, CUDA 12)",
        },
        "cuda13": {
            "archive": "crispasr-windows-x86_64-cuda13.zip",
            "dest_subdir": "cuda",
            "description": "Windows x86_64 CUDA 13 (NVIDIA GPU)",
        },
        "vulkan": {
            "archive": "crispasr-windows-x86_64-vulkan.zip",
            "dest_subdir": "vulkan",
            "description": "Windows x86_64 Vulkan (AMD/Intel/NVIDIA GPU)",
        },
    },
    "linux": {
        "cpu": {
            "archive": "crispasr-linux-x86_64.tar.gz",
            "dest_subdir": "cpu",
            "description": "Linux x86_64 CPU (Modern AVX2)",
        },
        "cpu-avx512": {
            "archive": "crispasr-linux-x86_64-avx512.tar.gz",
            "dest_subdir": "cpu",
            "description": "Linux x86_64 CPU (AVX-512)",
        },
        "cpu-legacy": {
            "archive": "crispasr-linux-x86_64-cpu-legacy.tar.gz",
            "dest_subdir": "cpu",
            "description": "Linux x86_64 CPU Legacy",
        },
        "cuda": {
            "archive": "crispasr-linux-x86_64-cuda.tar.gz",
            "dest_subdir": "cuda",
            "description": "Linux x86_64 CUDA (NVIDIA GPU, CUDA 12)",
        },
        "cuda13": {
            "archive": "crispasr-linux-x86_64-cuda13.tar.gz",
            "dest_subdir": "cuda",
            "description": "Linux x86_64 CUDA 13 (NVIDIA GPU)",
        },
        "vulkan": {
            "archive": "crispasr-linux-x86_64-vulkan.tar.gz",
            "dest_subdir": "vulkan",
            "description": "Linux x86_64 Vulkan (AMD/Intel/NVIDIA GPU)",
        },
        "arm64": {
            "archive": "crispasr-linux-arm64.tar.gz",
            "dest_subdir": "cpu",
            "description": "Linux ARM64 / AArch64",
        },
    },
    "darwin": {
        "arm64": {
            "archive": "crispasr-macos-arm64.tar.gz",
            "dest_subdir": "cpu",
            "description": "macOS Apple Silicon (M1/M2/M3/M4)",
        },
        "x86_64": {
            "archive": "crispasr-macos-x86_64.tar.gz",
            "dest_subdir": "cpu",
            "description": "macOS Intel x86_64",
        },
    }
}


def get_project_dir() -> Path:
    return Path(__file__).resolve().parent


# Backward compatibility alias
get_backend_dir = get_project_dir


def detect_platform() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    elif sys.platform.startswith("linux"):
        return "linux"
    elif sys.platform.startswith("darwin"):
        return "darwin"
    return "linux"


def download_file(url: str, dest_path: Path, description: str = ""):
    """Download a file with real-time terminal progress."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".download")

    print(f"\n[Downloading] {description or dest_path.name}")
    print(f"  URL: {url}")
    print(f"  Destination: {dest_path}")

    headers = {"User-Agent": "CrispASR-Downloader/1.0"}
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024 * 1024  # 1 MB buffer
            downloaded = 0

            with open(temp_path, "wb") as f:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    f.write(buffer)
                    if total_size > 0:
                        percent = downloaded * 100 / total_size
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        sys.stdout.write(f"\r  Progress: {mb_done:.1f}/{mb_total:.1f} MB ({percent:.1f}%)")
                    else:
                        mb_done = downloaded / (1024 * 1024)
                        sys.stdout.write(f"\r  Downloaded: {mb_done:.1f} MB")
                    sys.stdout.flush()

        print()  # newline
        if temp_path.exists():
            if dest_path.exists():
                dest_path.unlink()
            temp_path.rename(dest_path)
            print(f"  [OK] Saved: {dest_path} ({dest_path.stat().st_size / (1024*1024):.2f} MB)")
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Download failed for {url}: {e}") from e


def extract_archive(archive_path: Path, target_dir: Path):
    """Extract .zip or .tar.gz archive directly into target directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [Extracting] {archive_path.name} -> {target_dir}")

    temp_extract = target_dir / "_extract_tmp"
    if temp_extract.exists():
        shutil.rmtree(temp_extract, ignore_errors=True)
    temp_extract.mkdir(parents=True, exist_ok=True)

    try:
        if archive_path.name.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(temp_extract)
        elif archive_path.name.endswith((".tar.gz", ".tgz")):
            with tarfile.open(archive_path, "r:gz") as tf:
                tf.extractall(temp_extract)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path.name}")

        # Flatten single subdirectory wrapper if present
        items = list(temp_extract.iterdir())
        if len(items) == 1 and items[0].is_dir():
            source_dir = items[0]
        else:
            source_dir = temp_extract

        for item in source_dir.iterdir():
            dest = target_dir / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest, ignore_errors=True)
                else:
                    dest.unlink()
            shutil.move(str(item), str(target_dir))

        shutil.rmtree(temp_extract, ignore_errors=True)
    except Exception as e:
        shutil.rmtree(temp_extract, ignore_errors=True)
        raise e

    # Mark binary executable on Linux/Unix
    if not sys.platform.startswith("win"):
        for binary_name in ["crispasr", "crispasr.bin", "crispasr-quantize"]:
            bin_path = target_dir / binary_name
            if bin_path.exists():
                bin_path.chmod(0o755)

    print(f"  [OK] Extracted to {target_dir}")


def download_binary(platform_key: str, device_key: str, project_dir: Path, force: bool = False):
    """Download and extract CrispASR binary for the given platform and device."""
    platform_map = BINARY_TARGETS.get(platform_key)
    if not platform_map:
        raise ValueError(f"Unsupported platform '{platform_key}'. Options: {list(BINARY_TARGETS.keys())}")

    device_info = platform_map.get(device_key)
    if not device_info:
        raise ValueError(f"Unsupported device '{device_key}' for platform '{platform_key}'. Options: {list(platform_map.keys())}")

    archive_name = device_info["archive"]
    dest_subdir = device_info["dest_subdir"]
    description = device_info["description"]
    url = f"{GITHUB_RELEASE_BASE}/{archive_name}"

    bin_dir = project_dir / "bin" / dest_subdir
    bin_dir.mkdir(parents=True, exist_ok=True)

    exe_candidates = ["crispasr.exe", "crispasr"]
    has_binary = any((bin_dir / c).is_file() for c in exe_candidates)

    if has_binary and not force:
        print(f"[Skip] Binary already present in {bin_dir} ({description}). Use --force to re-download.")
        return

    temp_archive = project_dir / "bin" / archive_name
    try:
        download_file(url, temp_archive, description=f"CrispASR binary ({description})")
        extract_archive(temp_archive, bin_dir)
    finally:
        if temp_archive.exists():
            temp_archive.unlink()


def download_speech_model(project_dir: Path, force: bool = False):
    """Download Parakeet-v3 GGUF speech model for ATC."""
    model_dir = project_dir / "models" / "ggufs"
    model_file = model_dir / "speech-model.gguf"

    if model_file.exists() and not force:
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"[Skip] Speech model already present at {model_file} ({size_mb:.1f} MB). Use --force to re-download.")
        return

    download_file(PARAKEET_MODEL_URL, model_file, description="Parakeet-v3 ATC Speech Model (GGUF)")


def download_vad_model(project_dir: Path, force: bool = False):
    """Download Silero VAD model."""
    vad_dir = project_dir / "models" / "vad"
    vad_file = vad_dir / "ggml-silero-v6.2.0.bin"

    if vad_file.exists() and not force:
        size_kb = vad_file.stat().st_size / 1024
        print(f"[Skip] VAD model already present at {vad_file} ({size_kb:.1f} KB). Use --force to re-download.")
        return

    download_file(SILERO_VAD_URL, vad_file, description="Silero VAD v6.2.0 (ggml)")


def main():
    parser = argparse.ArgumentParser(description="Download binaries and models for ATC Speech Transcription")
    parser.add_argument(
        "--platform",
        choices=["auto", "windows", "linux", "darwin"],
        default="auto",
        help="Target operating system platform (default: auto-detect)",
    )
    parser.add_argument(
        "--device",
        choices=["all", "cpu", "cpu-legacy", "cuda", "cuda13", "vulkan", "arm64"],
        default="cpu",
        help="CrispASR target hardware device (default: cpu)",
    )
    parser.add_argument(
        "--only-model",
        action="store_true",
        help="Only download the Parakeet-v3 speech model and Silero VAD",
    )
    parser.add_argument(
        "--only-binary",
        action="store_true",
        help="Only download the CrispASR binaries",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing downloaded models and binaries",
    )

    args = parser.parse_args()
    project_dir = get_project_dir()

    plat = detect_platform() if args.platform == "auto" else args.platform
    print(f"=== ATC Transcription Setup & Downloader ===")
    print(f"Platform: {plat}")
    print(f"Project directory: {project_dir}")

    # Handle binary download
    if not args.only_model:
        if args.device == "all":
            devices_to_download = list(BINARY_TARGETS.get(plat, {}).keys())
        else:
            devices_to_download = [args.device]

        for d in devices_to_download:
            try:
                download_binary(plat, d, project_dir, force=args.force)
            except Exception as e:
                print(f"[Warning] Could not setup binary for device '{d}': {e}")

    # Handle model downloads
    if not args.only_binary:
        download_speech_model(project_dir, force=args.force)
        download_vad_model(project_dir, force=args.force)

    print("\n[Done] Setup complete! You can now run 'python app.py' or 'python run_transcription.py <audio_file>'.")


if __name__ == "__main__":
    main()
