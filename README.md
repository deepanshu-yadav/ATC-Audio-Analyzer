# Air Traffic Control (ATC) Audio Transcription Pipeline

An end-to-end Air Traffic Control (ATC) audio transcription system featuring audio pre-processing (high-pass filter, spectral noise reduction), voice activity detection (VAD), intelligent multi-processing chunking, and high-performance inference via **CrispASR** (`ggml`/`gguf`).

Includes a **FastAPI backend**, an interactive **web application UI**, and a **CLI transcription tool** supporting both **CPU** and **NVIDIA CUDA GPU** acceleration on Windows and Linux.

---

## Features

- **ATC Speech Recognition**: Uses [Parakeet-v3-For_ATC GGUF](https://huggingface.co/pronoobie/Parakeet-v3-For_ATC), fine-tuned for noisy ATC aviation communications.
- **Noise Filtering & Reduction**: Integrated high-pass filter (>100 Hz) and spectral noise reduction to handle radio background noise, carrier hiss, and cockpit interference.
- **Silero VAD**: Automatic speech detection and silence trimming using [Silero VAD v6.2.0](https://huggingface.co/ggml-org/whisper-vad).
- **Parallel Chunking**: Overlapping chunk segmentation with multiprocessing for long audio recordings (> 120s) with boundary stitching.
- **Hardware Acceleration**: Auto-detects and supports both **CPU** (OpenBLAS/AVX) and **CUDA** (NVIDIA GPU).
- **Web Interface & API**: FastAPI backend with SQLite transcription history, real-time audio playback comparison (Original vs. Denoised), and full JSON/text export.

---

## Directory Structure

```text
backend/
├── app.py                         # FastAPI web server and REST API
├── run_transcription.py           # CLI runner for transcription with auto-chunking
├── transcribe_atc.py              # Core transcription pipeline and CrispASR wrapper
├── audio_chunker.py               # Multiprocessing overlapping chunk processor
├── device_utils.py                # Hardware & binary auto-discovery (CPU/CUDA)
├── database.py                    # SQLite database for transcription records
├── download_models_and_bins.py    # Automatic downloader for models & binaries
├── requirements_transcribe.txt    # Python dependencies
├── bin/
│   ├── cpu/                       # CrispASR CPU executable & OpenBLAS DLLs
│   └── cuda/                      # CrispASR CUDA executable & runtime DLLs
├── models/
│   ├── ggufs/                     # speech-model.gguf (Parakeet-v3)
│   └── vad/                       # ggml-silero-v6.2.0.bin (Silero VAD)
├── output/                        # Saved filtered, denoised, and split wavs
└── static/                        # Web dashboard UI (HTML, CSS, JS)
```

---

## Prerequisites

1. **Python**: 3.9 – 3.12 recommended.
2. **FFmpeg / Libsndfile**: Ensure audio codecs are available on system PATH for audio reading (`librosa`, `soundfile`).
3. **GPU Drivers (Optional)**: NVIDIA driver + CUDA Toolkit if using CUDA acceleration.

---

## Installation & Setup

### 1. Set Up Python Virtual Environment

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements_transcribe.txt
```

---

### 2. Download Models and CrispASR Binaries

We provide an automated setup script [`download_models_and_bins.py`](file:///c:/Users/DEEPANSHU/Desktop/workspace/atc/backend/download_models_and_bins.py) that fetches both the speech recognition model, Silero VAD, and the appropriate CrispASR binaries.

#### Option A: Automatic Setup (Recommended)

```bash
# Download speech model + Silero VAD + CPU binary (default)
python download_models_and_bins.py --device cpu

# Or download with NVIDIA CUDA GPU binary
python download_models_and_bins.py --device cuda

# Or download only the models (if you already have binaries)
python download_models_and_bins.py --only-model
```

#### Option B: Manual Setup

If you prefer to download components manually:

1. **ATC Speech Model (GGUF)**:
   - **Repository**: [pronoobie/Parakeet-v3-For_ATC](https://huggingface.co/pronoobie/Parakeet-v3-For_ATC)
   - **File**: [`speech-model.gguf`](https://huggingface.co/pronoobie/Parakeet-v3-For_ATC/blob/main/speech-model.gguf)
   - Place into: `backend/models/ggufs/speech-model.gguf`

2. **Silero VAD Model**:
   - **Download**: [ggml-silero-v6.2.0.bin](https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin)
   - Place into: `backend/models/vad/ggml-silero-v6.2.0.bin`

3. **CrispASR Release Binaries (v0.8.39)**:
   - **Releases Page**: [CrispASR Releases v0.8.39](https://github.com/CrispStrobe/CrispASR/releases/tag/v0.8.39)
   - Extract into `backend/bin/cpu/` or `backend/bin/cuda/`:

   | OS / Target | Recommended Release Archive | Extract Destination |
   | :--- | :--- | :--- |
   | **Windows (CPU)** | `crispasr-windows-x86_64-cpu.zip` | `backend/bin/cpu/` |
   | **Windows (Legacy CPU)** | `crispasr-windows-x86_64-cpu-legacy.zip` | `backend/bin/cpu/` |
   | **Windows (CUDA / NVIDIA)** | `crispasr-windows-x86_64-cuda.zip` | `backend/bin/cuda/` |
   | **Windows (CUDA 13)** | `crispasr-windows-x86_64-cuda13.zip` | `backend/bin/cuda/` |
   | **Linux (CPU)** | `crispasr-linux-x86_64.tar.gz` | `backend/bin/cpu/` |
   | **Linux (CUDA)** | `crispasr-linux-x86_64-cuda.tar.gz` | `backend/bin/cuda/` |
   | **macOS (Apple Silicon)** | `crispasr-macos-arm64.tar.gz` | `backend/bin/cpu/` |

> [!NOTE]
> Make sure the folder contains `crispasr.exe` (or `crispasr` on Linux) and any accompanying `.dll` or `.so` libraries (such as `openblas.dll` or CUDA DLLs).

---

## Running the Application

### 1. Web Application (UI + API)

To launch the FastAPI server with web dashboard:

```bash
python app.py
```

The server starts at `http://127.0.0.1:8000`:
- **Web Dashboard**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Swagger API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **System Capabilities**: [http://127.0.0.1:8000/api/system](http://127.0.0.1:8000/api/system)

#### Web UI Features:
- Upload MP3, WAV, FLAC, or OGG ATC recordings.
- Select compute device: `auto`, `cuda`, or `cpu`.
- Toggle High-Pass Filtering and Noise Reduction with customizable strength.
- Play original vs. cleaned audio directly in the browser.
- View segmented transcriptions with timestamps and confidence scores.
- Search and review past transcription history stored in SQLite.

---

### 2. Command-Line Transcription (CLI)

You can transcribe audio files directly via command line using [`run_transcription.py`](file:///c:/Users/DEEPANSHU/Desktop/workspace/atc/backend/run_transcription.py):

```bash
# Transcribe single file (auto-detects CPU/CUDA and chunking)
python run_transcription.py path/to/atc_audio.mp3

# Specify output JSON path
python run_transcription.py sample.wav --output output/result.json

# Force CUDA GPU
python run_transcription.py sample.wav --device cuda

# Tune parallel chunking parameters
python run_transcription.py sample.wav --chunk-duration 15.0 --overlap 1.5 --num-workers 4

# Skip noise reduction (if audio is already clean)
python run_transcription.py sample.wav --no-noise-reduce
```

---

## How It Works

1. **Audio Ingestion & Normalization**: The audio is loaded via `librosa`, converted to 16 kHz mono.
2. **Noise Preprocessing**:
   - Butterworth high-pass filter (>100 Hz) eliminates low-frequency rumbling.
   - Non-stationary spectral noise gating cleans background radio hiss.
3. **Chunking & Multi-Processing**:
   - Audio longer than 120 seconds is split into overlapping chunks (e.g. 10s–20s duration with 1.0s overlap).
   - Audio chunks are processed across CPU worker pools.
4. **CrispASR Engine**:
   - The standalone `crispasr` binary performs beam-search speech decoding against `speech-model.gguf`.
   - Silence intervals are managed using Silero VAD.
5. **Deduplication & Merging**:
   - Overlapping segments are deduplicated based on timestamp intervals and text matching.
