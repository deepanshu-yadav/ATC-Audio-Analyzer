"""
ATC Audio Transcription Pipeline (Lightweight crispasr.exe Version)
Pipeline with Noise Removal (Filter + Spectral NR) and crispasr.exe Wrapper
"""

import os
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Tuple, Optional, Union
import json
from pathlib import Path
import time
import argparse
import subprocess
import re
import datetime
import device_utils


class AudioTranscriptionPipeline:
    """
    Complete audio transcription pipeline for noisy ATC recordings
    Includes: Filter + Noise Removal -> crispasr (CPU/CUDA, Windows/Linux) Wrapper
    """

    def __init__(self,
                 exe_path: Optional[str] = None,
                 model_path: Optional[str] = None,
                 sample_rate: int = 16000,
                 use_gpu: Optional[bool] = None,
                 device: str = "auto",
                 vad_model_path: Optional[str] = None,      # NEW
                 # Keep other parameters for compatibility, but ignored
                 whisper_model: Optional[str] = None,
                 parakeet_model_dir: Optional[str] = None,
                 backend: Optional[str] = None):
        """
        Initialize the transcription pipeline.

        Args:
            exe_path (Optional[str]): Path to the crispasr executable. Auto-discovered if None.
            model_path (Optional[str]): Path to the GGUF model. Auto-discovered if None.
            sample_rate (int): Target audio sample rate.
            use_gpu (Optional[bool]): Whether to enable GPU (CUDA). Auto-detected if None.
            device (str): Requested compute device ('auto', 'cuda', or 'cpu').
            vad_model_path (Optional[str]): Path to the Silero VAD .bin. Auto-discovered
                in models/vad/ if None.
        """
        self.sample_rate = sample_rate
        self.backend = "crispasr"

        # Determine target device
        target_device = device or "auto"
        if use_gpu is True:
            target_device = "cuda"
        elif use_gpu is False:
            target_device = "cpu"

        # Discover binary if not explicitly provided or invalid
        if not exe_path or not os.path.exists(exe_path):
            resolved_exe, resolved_device, resolved_gpu = device_utils.find_crispasr_binary(device=target_device)
            self.exe_path = resolved_exe
            self.device = resolved_device
            self.use_gpu = resolved_gpu if use_gpu is None else use_gpu
        else:
            self.exe_path = os.path.abspath(exe_path)
            if use_gpu is not None:
                self.use_gpu = use_gpu
                self.device = "cuda" if use_gpu else "cpu"
            else:
                self.use_gpu = "cuda" in self.exe_path.lower()
                self.device = "cuda" if self.use_gpu else "cpu"

        # Discover model if not provided or invalid
        if whisper_model and os.path.exists(whisper_model) and whisper_model.endswith(".gguf"):
            self.model_path = os.path.abspath(whisper_model)
        elif model_path and os.path.exists(model_path):
            self.model_path = os.path.abspath(model_path)
        else:
            self.model_path = device_utils.find_model_path(model_path)

        # Discover VAD model (models/vad/ggml-silero-v6.2.0.bin)
        self.vad_model_path = self._find_vad_model_path(vad_model_path)
        if self.vad_model_path is None:
            print("Warning: Silero VAD model not found in models/vad/. "
                  "crispasr will fall back to auto-download in ~/.cache/crispasr.")

        print(f"Initializing CrispASR pipeline...")
        print(f"  Platform: {device_utils.get_platform_name()}")
        print(f"  Device: {self.device} (GPU enabled: {self.use_gpu})")
        print(f"  Executable: {self.exe_path}")
        print(f"  Model: {self.model_path}")
        print(f"  VAD model: {self.vad_model_path or 'auto (crispasr default)'}")

    def _find_vad_model_path(self, vad_model_path: Optional[str] = None) -> Optional[str]:
        """
        Locate the Silero VAD model (ggml-silero-v6.2.0.bin) without relying on
        the default ~/.cache/crispasr location.

        Search order:
          1. Explicit path passed in (absolute, or relative to cwd / this file)
          2. CRISPASR_VAD_MODEL environment variable
          3. <this_file_dir>/models/vad/ggml-silero-v6.2.0.bin
          4. <this_file_dir>/../models/vad/ggml-silero-v6.2.0.bin
          5. <cwd>/models/vad/ggml-silero-v6.2.0.bin

        Returns:
            Absolute path if found, otherwise None.
        """
        filename = "ggml-silero-v6.2.0.bin"
        base_dir = os.path.dirname(os.path.abspath(__file__))

        candidates = []

        # 1. Explicit argument
        if vad_model_path:
            candidates.append(vad_model_path)  # as given (absolute or cwd-relative)
            if not os.path.isabs(vad_model_path):
                candidates.append(os.path.join(base_dir, vad_model_path))

        # 2. Environment variable
        env_path = os.environ.get("CRISPASR_VAD_MODEL")
        if env_path:
            candidates.append(env_path)

        # 3-5. Relative default locations
        candidates.extend([
            os.path.join(base_dir, "models", "vad", filename),
            os.path.join(base_dir, "..", "models", "vad", filename),
            os.path.join(os.getcwd(), "models", "vad", filename),
        ])

        for path in candidates:
            if path and os.path.isfile(path):
                return os.path.abspath(path)

        return None

    def _load_vad_model(self):
        """Deprecated: VAD is handled internally by crispasr.exe"""
        return None, (None, None, None, None, None)

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and resample to target sample rate

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        print(f"Loading audio from: {audio_path}")
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        return audio, sr

    def apply_vad(self,
                  audio: np.ndarray,
                  threshold: float = 0.5,
                  min_speech_duration_ms: int = 250,
                  min_silence_duration_ms: int = 100) -> List[Dict]:
        """Deprecated: VAD is handled internally by crispasr.exe"""
        print("VAD is now handled internally by crispasr.exe. Returning empty segment list.")
        return []

    def remove_noise(self,
                     audio: np.ndarray,
                     stationary: bool = True,
                     prop_decrease: float = 1.0) -> np.ndarray:
        """
        Remove noise from audio using spectral gating

        Args:
            audio: Input audio signal
            stationary: Use stationary or non-stationary noise reduction
            prop_decrease: Proportion of noise to reduce (0-1)

        Returns:
            Denoised audio signal
        """
        print("Removing noise...")

        try:
            import noisereduce as nr

            # Apply noise reduction
            denoised_audio = nr.reduce_noise(
                y=audio,
                sr=self.sample_rate,
                stationary=stationary,
                prop_decrease=prop_decrease
            )

            return denoised_audio
        except ImportError:
            print("Warning: noisereduce not installed. Skipping noise removal.")
            print("Install with: pip install noisereduce")
            return audio

    def apply_deepfilternet(self, audio: np.ndarray) -> np.ndarray:
        """
        Deprecated/Not recommended under lightweight env, but kept for compatibility.
        """
        print("Skipping DeepFilterNet (unsupported in lightweight env).")
        return audio

    def apply_bandpass_filter(self,
                              audio: np.ndarray,
                              lowcut: float = 300.0,
                              highcut: float = 3400.0,
                              order: int = 5) -> np.ndarray:
        """
        Apply Bandpass Filter (Butterworth) to keep frequencies relevant for human speech.
        """
        try:
            from scipy.signal import butter, lfilter

            nyq = 0.5 * self.sample_rate

            # Determine filter type based on bounds
            use_low = (lowcut > 0)
            use_high = (highcut < nyq)

            if not use_low and not use_high:
                print("Skipping filter: both lowcut and highcut are outside bounds.")
                return audio

            if use_low and use_high:
                print(f"Applying Bandpass Filter ({lowcut}Hz - {highcut}Hz)...")
                low = lowcut / nyq
                high = highcut / nyq
                b, a = butter(order, [low, high], btype='band')
            elif use_low:
                print(f"Applying High-pass Filter (>{lowcut}Hz)...")
                low = lowcut / nyq
                b, a = butter(order, low, btype='high')
            else:
                print(f"Applying Low-pass Filter (<{highcut}Hz)...")
                high = highcut / nyq
                b, a = butter(order, high, btype='low')

            filtered_audio = lfilter(b, a, audio)
            return filtered_audio
        except ImportError:
            print("Warning: scipy not installed. Skipping bandpass filter.")
            return audio
        except Exception as e:
            print(f"Error applying bandpass filter: {e}")
            return audio

    def extract_speech_segments(self,
                                audio: np.ndarray,
                                speech_timestamps: List[Dict]) -> np.ndarray:
        """Deprecated: VAD is handled internally by crispasr.exe"""
        return audio

    def transcribe(self,
                   audio: Union[np.ndarray, str],
                   language: str = "en",
                   task: str = "transcribe") -> Dict:
        """
        Transcribe audio using crispasr.exe subprocess wrapper

        Args:
            audio: Numpy array of audio data, or path to audio file
            language: Language code (default: 'en')
            task: Action (ignored)

        Returns:
            Transcription result dictionary
        """
        import tempfile
        import re

        temp_wav_path = None
        if isinstance(audio, np.ndarray):
            # Write numpy array to a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_wav_path = tmp.name
            sf.write(temp_wav_path, audio, self.sample_rate)
            audio_file_path = temp_wav_path
        else:
            audio_file_path = audio

        try:
            # Build command with optimizations: beam search, forced English language,
            # and domain-specific hotwords
            hotwords = ""  # e.g. "callsign1,waypoint1"

            cmd = [
                self.exe_path,
                "-m", self.model_path,
                "-f", audio_file_path,
                "--vad",
            ]

            # Use the local VAD model so nothing is downloaded to ~/.cache/crispasr
            if self.vad_model_path:
                cmd += ["-vm", self.vad_model_path]

            cmd += [
                "--flush-after", "1",
                "-osrt",
                "-bs", "5",         # Beam search size 5
                "-l", "en",         # Force English language to prevent auto-detect errors
            ]

            # Only pass hotwords when non-empty (bias callsigns and waypoints)
            if hotwords:
                cmd += ["--hotwords", hotwords]

            # Explicitly pass -ng if running in CPU mode
            if not self.use_gpu:
                cmd.append("-ng")

            print(f"Running backend command: {' '.join(cmd)}")

            # Execute subprocess with environment configured for libraries (openblas.dll, .so)
            subproc_env = device_utils.get_subprocess_env(self.exe_path)
            process_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
                env=subproc_env,
                check=False
            )

            stdout = process_result.stdout
            stderr = process_result.stderr

            if process_result.returncode != 0:
                print(f"Warning: crispasr.exe exited with return code {process_result.returncode}")
                print(f"Stderr:\n{stderr}")

            # Try to read the SRT file if generated
            srt_path_1 = audio_file_path + ".srt"
            srt_path_2 = os.path.splitext(audio_file_path)[0] + ".srt"

            segments = []
            srt_found = False

            for srt_path in [srt_path_1, srt_path_2]:
                if os.path.exists(srt_path):
                    print(f"Parsing timestamps from generated SRT: {srt_path}")
                    segments = self._parse_srt_file(srt_path)
                    try:
                        os.remove(srt_path)
                    except Exception as e:
                        print(f"Warning: Could not remove SRT file {srt_path}: {e}")
                    if segments:
                        srt_found = True
                        break

            # If no SRT file on disk, but stdout contains SRT formatting
            if not srt_found and "-->" in stdout:
                print("Parsing SRT content directly from stdout...")
                segments = self._parse_srt_content(stdout)
                if segments:
                    srt_found = True

            if srt_found:
                text_lines = [seg["text"] for seg in segments]
                full_text = " ".join(text_lines).strip()
            else:
                # Parse text lines from stdout
                text_lines = []
                stdout_lines = stdout.splitlines()
                for line in stdout_lines:
                    line_strip = line.strip()
                    if line_strip and not self._is_log_line(line_strip):
                        # Clean timestamp prefix if present in the line (e.g. from stdout)
                        cleaned_line = re.sub(r'^\[\d{2}:\d{2}:\d{2}[,\.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,\.]\d{3}\]\s*', '', line_strip)
                        text_lines.append(cleaned_line.strip())

                full_text = " ".join(text_lines).strip()

                # Fallback 1: If SRT was not generated/found but VAD info is in stderr logs,
                # parse VAD segments from stderr and pair them with stdout text lines
                print("No SRT file found. Attempting to parse VAD segments from stderr logs...")
                vad_segments = self._parse_vad_from_stderr(stderr)

                if vad_segments and text_lines:
                    print(f"Found {len(vad_segments)} VAD segments and {len(text_lines)} transcribed text lines.")
                    # Pair them up
                    for i in range(min(len(vad_segments), len(text_lines))):
                        segments.append({
                            "start": vad_segments[i]["start"],
                            "end": vad_segments[i]["end"],
                            "text": text_lines[i]
                        })
                    # If we have leftover text lines, combine them into the last segment
                    if len(text_lines) > len(vad_segments):
                        extra_text = " ".join(text_lines[len(vad_segments):])
                        if segments:
                            segments[-1]["text"] += " " + extra_text
                elif vad_segments and full_text:
                    segments.append({
                        "start": vad_segments[0]["start"],
                        "end": vad_segments[-1]["end"],
                        "text": full_text
                    })

            # Fallback 2: General fallback (single global segment)
            if not segments and full_text:
                segments.append({
                    "start": 0.0,
                    "end": 0.0,
                    "text": full_text
                })

            # Split long segments to improve readability
            segments = self._split_long_segments(segments, max_words=10)

            return {
                "text": full_text,
                "segments": segments,
                "language": language
            }

        finally:
            if temp_wav_path and os.path.exists(temp_wav_path):
                try:
                    os.remove(temp_wav_path)
                except Exception as e:
                    print(f"Warning: Could not remove temp file {temp_wav_path}: {e}")

    def _parse_srt_file(self, srt_path: str) -> List[Dict]:
        try:
            with open(srt_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return self._parse_srt_content(content)
        except Exception as e:
            print(f"Warning: Failed to parse SRT file {srt_path}: {e}")
            return []

    def _parse_srt_content(self, content: str) -> List[Dict]:
        try:
            segments = []
            content = content.replace('\r\n', '\n')

            # Match SRT blocks
            srt_pattern = re.compile(
                r'(?:\d+)\n(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\n(.*?)(?=\n\s*\n|\n\d+\n|\Z)',
                re.DOTALL
            )

            matches = srt_pattern.findall(content)
            for start_str, end_str, text_content in matches:
                start_sec = self._parse_timestamp_to_seconds(start_str)
                end_sec = self._parse_timestamp_to_seconds(end_str)
                clean_lines = [line.strip() for line in text_content.strip().splitlines()]
                clean_text = " ".join([l for l in clean_lines if l]).strip()

                if clean_text:
                    segments.append({
                        "start": start_sec,
                        "end": end_sec,
                        "text": clean_text
                    })
            return segments
        except Exception as e:
            print(f"Warning: Failed to parse SRT content: {e}")
            return []

    def _split_long_segments(self, segments: List[Dict], max_words: int = 10) -> List[Dict]:
        new_segments = []
        for seg in segments:
            text = seg["text"].strip()
            words = text.split()
            if len(words) <= max_words:
                new_segments.append(seg)
                continue

            start = seg["start"]
            end = seg["end"]
            duration = end - start
            num_words = len(words)

            if duration <= 0:
                new_segments.append(seg)
                continue

            for i in range(0, num_words, max_words):
                chunk_words = words[i: i + max_words]
                chunk_text = " ".join(chunk_words)

                chunk_start = start + (i / num_words) * duration
                chunk_end = start + (min(i + max_words, num_words) / num_words) * duration

                new_segments.append({
                    "start": round(chunk_start, 2),
                    "end": round(chunk_end, 2),
                    "text": chunk_text
                })
        return new_segments

    def _parse_vad_from_stderr(self, stderr: str) -> List[Dict]:
        vad_segments = []
        pattern = re.compile(
            r'VAD segment\s*\d+:\s*start\s*=\s*([\d\.]+),\s*end\s*=\s*([\d\.]+)'
        )
        for line in stderr.splitlines():
            match = pattern.search(line)
            if match:
                start_val = float(match.group(1))
                end_val = float(match.group(2))
                vad_segments.append({
                    "start": start_val,
                    "end": end_val
                })
        return vad_segments

    def _parse_timestamp_to_seconds(self, ts_str: str) -> float:
        try:
            ts_str = ts_str.replace(',', '.')
            parts = ts_str.split(':')
            if len(parts) == 3:
                h, m, s = parts
                return float(h) * 3600 + float(m) * 60 + float(s)
            elif len(parts) == 2:
                m, s = parts
                return float(m) * 60 + float(s)
            return float(ts_str)
        except Exception:
            return 0.0

    def _is_log_line(self, line: str) -> bool:
        log_prefixes = [
            "system_info:", "whisper_init", "model", "main:", "whisper_",
            "llama_", "ggml_", "load_", "warning:", "error:", "info:"
        ]
        line_lower = line.lower()
        for prefix in log_prefixes:
            if line_lower.startswith(prefix):
                return True
        if all(c in '-=_* \t' for c in line):
            return True
        return False

    def _transcribe_chunk(self,
                          chunk_wav_path: str,
                          chunk_id: int,
                          start_time_offset: float,
                          end_time_offset: float,
                          language: str = "en") -> Dict:
        """
        Transcribe a single audio chunk and adjust segment timestamps to the
        global timeline.

        Args:
            chunk_wav_path (str): Path to the temporary WAV file for this chunk.
            chunk_id (int): 0-based chunk index.
            start_time_offset (float): Where this chunk starts in the full audio (seconds).
            end_time_offset (float): Where this chunk ends in the full audio (seconds).
            language (str): Language code.

        Returns:
            Dict with keys: chunk_id, start_time, end_time, text, segments.
        """
        result = self.transcribe(chunk_wav_path, language=language)

        # Shift every segment timestamp so it refers to the global timeline
        shifted_segments = []
        for seg in result.get("segments", []):
            shifted_segments.append({
                "start": round(seg["start"] + start_time_offset, 2),
                "end": round(seg["end"] + start_time_offset, 2),
                "text": seg["text"]
            })

        return {
            "chunk_id": chunk_id,
            "start_time": start_time_offset,
            "end_time": end_time_offset,
            "text": result.get("text", ""),
            "segments": shifted_segments,
        }

    @staticmethod
    def _merge_chunk_results(chunk_results: List[Dict],
                             overlap_duration: float) -> Dict:
        """
        Merge transcription results from overlapping chunks into a single
        coherent transcript.  Segments whose *start* falls inside the overlap
        region of a previous chunk are dropped to avoid duplicated text.

        Args:
            chunk_results (List[Dict]): Sorted list of per-chunk results.
            overlap_duration (float): Overlap duration in seconds.

        Returns:
            Dict with merged text and segments.
        """
        if not chunk_results:
            return {"text": "", "segments": []}

        merged_segments: List[Dict] = []
        merged_text_parts: List[str] = []

        for i, cr in enumerate(chunk_results):
            if i == 0:
                # First chunk — keep everything
                merged_segments.extend(cr.get("segments", []))
                if cr.get("text"):
                    merged_text_parts.append(cr["text"])
                continue

            # For subsequent chunks, drop segments that start before the
            # effective boundary (i.e. inside the overlap zone of the
            # previous chunk).
            effective_start = cr["start_time"] + overlap_duration
            kept_texts: List[str] = []
            for seg in cr.get("segments", []):
                if seg["start"] >= effective_start - 0.5:  # 0.5s tolerance
                    merged_segments.append(seg)
                    kept_texts.append(seg["text"])

            # If no segments survived the filter but there is raw text,
            # include the whole chunk text (fallback)
            if not kept_texts and cr.get("text"):
                merged_text_parts.append(cr["text"])
            elif kept_texts:
                merged_text_parts.append(" ".join(kept_texts))

        return {
            "text": " ".join(merged_text_parts).strip(),
            "segments": merged_segments,
        }

    def process_audio(self,
                      audio_path: str,
                      apply_vad: bool = True,  # Ignored, handled by exe
                      apply_noise_removal: bool = True,
                      vad_threshold: float = 0.5,
                      noise_reduction: float = 0.75,
                      language: str = "en",
                      save_intermediate: bool = False,
                      output_dir: Optional[str] = None,
                      chunk_duration: float = 30.0,
                      overlap_duration: float = 5.0) -> Dict:
        """
        Complete pipeline: Load -> Filter & Denoise -> Chunk -> Parallel
        Transcribe (crispasr.exe) -> Merge.

        When the audio is longer than 2 minutes the denoised signal is split
        into overlapping chunks which are transcribed *concurrently* (each
        chunk spawns its own crispasr.exe subprocess via ThreadPoolExecutor).
        Results are then merged with overlap-aware deduplication.

        For audio shorter than 2 minutes the full file is transcribed in a
        single pass (no chunking overhead).
        """
        import multiprocessing
        from concurrent.futures import ThreadPoolExecutor, as_completed

        CHUNKING_THRESHOLD_SEC = 120.0  # 2 minutes

        start_time = time.time()

        # Load audio
        audio, sr = self.load_audio(audio_path)
        audio_duration_sec = len(audio) / self.sample_rate

        # Generate timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        filtered_audio = audio
        denoised_audio = audio

        filtered_full_path = None
        filtered_full_path_rel = None
        denoised_full_path = None
        denoised_full_path_rel = None
        intermediate_chunks = []

        # Apply Global Noise Removal (Bandpass + Spectral Gating)
        if apply_noise_removal:
            print("Applying global noise removal...")
            # 1. Bandpass Filter (Widened: 100Hz - 8000Hz)
            filtered_audio = self.apply_bandpass_filter(audio, lowcut=100.0, highcut=8000.0)

            if save_intermediate and output_dir:
                os.makedirs(output_dir, exist_ok=True)
                filtered_path = os.path.join(output_dir, f"filtered_full_{timestamp}.wav")
                sf.write(filtered_path, filtered_audio, self.sample_rate)
                filtered_full_path = os.path.abspath(filtered_path)
                filtered_full_path_rel = os.path.relpath(filtered_path).replace("\\", "/")
                print(f"Saved global filtered audio: {filtered_full_path}")

            # 2. Spectral Gating (noisereduce)
            denoised_audio = self.remove_noise(filtered_audio.copy(), prop_decrease=noise_reduction)

            if save_intermediate and output_dir:
                denoised_path = os.path.join(output_dir, f"denoised_full_{timestamp}.wav")
                sf.write(denoised_path, denoised_audio, self.sample_rate)
                denoised_full_path = os.path.abspath(denoised_path)
                denoised_full_path_rel = os.path.relpath(denoised_path).replace("\\", "/")
                print(f"Saved global denoised audio: {denoised_full_path}")

        # ── Decide whether to chunk ──────────────────────────────────────
        use_chunking = audio_duration_sec > CHUNKING_THRESHOLD_SEC

        if use_chunking:
            # Determine optimal num_workers (threads — each one spawns a
            # crispasr.exe subprocess so actual CPU parallelism comes from
            # the OS process scheduler, not the GIL).
            cpu_count = multiprocessing.cpu_count()
            num_workers = min(cpu_count, 4)
            if audio_duration_sec <= 300:
                num_workers = min(num_workers, 2)

            print(f"\n{'='*60}")
            print(f"Parallel chunking ENABLED  (audio={audio_duration_sec:.1f}s > {CHUNKING_THRESHOLD_SEC:.0f}s)")
            print(f"  Chunk duration : {chunk_duration}s")
            print(f"  Overlap        : {overlap_duration}s")
            print(f"  Workers        : {num_workers}")
            print(f"{'='*60}\n")

            # ── Split into overlapping chunks ────────────────────────────
            import tempfile
            chunk_samples = int(chunk_duration * self.sample_rate)
            step_samples = int((chunk_duration - overlap_duration) * self.sample_rate)
            total_samples = len(denoised_audio)

            chunk_infos: List[Dict] = []  # (chunk_id, tmp_path, start_sec, end_sec)
            chunk_idx = 0
            start_sample = 0

            while start_sample < total_samples:
                end_sample = min(start_sample + chunk_samples, total_samples)
                chunk_audio = denoised_audio[start_sample:end_sample]

                # Write chunk to a temp WAV
                tmp = tempfile.NamedTemporaryFile(
                    suffix=f"_chunk{chunk_idx}.wav", delete=False
                )
                tmp_path = tmp.name
                tmp.close()
                sf.write(tmp_path, chunk_audio, self.sample_rate)

                start_sec = start_sample / self.sample_rate
                end_sec = end_sample / self.sample_rate

                chunk_infos.append({
                    "chunk_id": chunk_idx,
                    "tmp_path": tmp_path,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                })

                print(f"  Chunk {chunk_idx}: {start_sec:.2f}s – {end_sec:.2f}s  ({tmp_path})")

                if end_sample >= total_samples:
                    break
                start_sample += step_samples
                chunk_idx += 1

            print(f"\nCreated {len(chunk_infos)} chunks — launching {num_workers} parallel transcriptions …\n")

            # ── Transcribe chunks in parallel ────────────────────────────
            chunk_results: List[Dict] = [None] * len(chunk_infos)  # type: ignore[list-item]

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_idx = {}
                for ci in chunk_infos:
                    fut = executor.submit(
                        self._transcribe_chunk,
                        chunk_wav_path=ci["tmp_path"],
                        chunk_id=ci["chunk_id"],
                        start_time_offset=ci["start_sec"],
                        end_time_offset=ci["end_sec"],
                        language=language,
                    )
                    future_to_idx[fut] = ci["chunk_id"]

                for fut in as_completed(future_to_idx):
                    idx = future_to_idx[fut]
                    try:
                        chunk_results[idx] = fut.result()
                        print(f"  ✓ Chunk {idx} transcribed")
                    except Exception as exc:
                        print(f"  ✗ Chunk {idx} FAILED: {exc}")
                        chunk_results[idx] = {
                            "chunk_id": idx,
                            "start_time": chunk_infos[idx]["start_sec"],
                            "end_time": chunk_infos[idx]["end_sec"],
                            "text": "",
                            "segments": [],
                        }

            # Clean up temp chunk files
            for ci in chunk_infos:
                try:
                    os.remove(ci["tmp_path"])
                except OSError:
                    pass

            # ── Merge results ────────────────────────────────────────────
            print("\nMerging chunk transcriptions …")
            merged = self._merge_chunk_results(chunk_results, overlap_duration)
            result = {
                "text": merged["text"],
                "segments": merged["segments"],
                "language": language,
            }

        else:
            # ── Short audio: single-pass transcription ───────────────────
            print(f"\nSingle-pass transcription (audio={audio_duration_sec:.1f}s ≤ {CHUNKING_THRESHOLD_SEC:.0f}s)")
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_wav_path = tmp.name
            try:
                sf.write(temp_wav_path, denoised_audio, self.sample_rate)
                result = self.transcribe(temp_wav_path, language=language)
            finally:
                if os.path.exists(temp_wav_path):
                    try:
                        os.remove(temp_wav_path)
                    except Exception as e:
                        print(f"Warning: Could not remove temp file {temp_wav_path}: {e}")

        # ── Save intermediate files in overlapping chunks ────────────────
        if save_intermediate and output_dir:
            os.makedirs(output_dir, exist_ok=True)

            step_duration_sec = chunk_duration - overlap_duration

            chunk_samples = int(chunk_duration * self.sample_rate)
            step_samples = int(step_duration_sec * self.sample_rate)
            total_samples = len(filtered_audio)

            chunk_idx = 0
            while True:
                start_sample = chunk_idx * step_samples
                end_sample = start_sample + chunk_samples

                if start_sample >= total_samples:
                    break

                end_sample = min(end_sample, total_samples)
                chunk_num = chunk_idx + 1

                # Slice and save filtered chunk
                chunk_filtered = filtered_audio[start_sample:end_sample]
                filtered_chunk_path = os.path.join(output_dir, f"filtered_chunk_{chunk_num}_{timestamp}.wav")
                sf.write(filtered_chunk_path, chunk_filtered, self.sample_rate)

                # Slice and save denoised chunk
                chunk_denoised = denoised_audio[start_sample:end_sample]
                denoised_chunk_path = os.path.join(output_dir, f"denoised_chunk_{chunk_num}_{timestamp}.wav")
                sf.write(denoised_chunk_path, chunk_denoised, self.sample_rate)

                intermediate_chunks.append({
                    "chunk_num": chunk_num,
                    "start_time": round(start_sample / self.sample_rate, 2),
                    "end_time": round(end_sample / self.sample_rate, 2),
                    "filtered_path": os.path.abspath(filtered_chunk_path),
                    "filtered_path_rel": os.path.relpath(filtered_chunk_path).replace("\\", "/"),
                    "denoised_path": os.path.abspath(denoised_chunk_path),
                    "denoised_path_rel": os.path.relpath(denoised_chunk_path).replace("\\", "/")
                })

                if end_sample >= total_samples:
                    break

                chunk_idx += 1

            print(f"Saved {chunk_idx + 1} intermediate overlapping {chunk_duration:.0f}-second filtered and denoised chunks in {output_dir}")

        # Prepare output
        output = {
            "text": result["text"],
            "segments": result.get("segments", []),
            "language": result["language"],
            "processing_time_seconds": time.time() - start_time,
            "speech_duration_seconds": audio_duration_sec,
            "vad_enabled": True,
            "noise_removal_enabled": apply_noise_removal,
            "timestamp": timestamp,
            "chunking_enabled": use_chunking,
            "chunk_duration": chunk_duration if use_chunking else None,
            "overlap_duration": overlap_duration if use_chunking else None,
            "num_workers": num_workers if use_chunking else 1,
            "filtered_full_path": filtered_full_path,
            "filtered_full_path_rel": filtered_full_path_rel,
            "denoised_full_path": denoised_full_path,
            "denoised_full_path_rel": denoised_full_path_rel,
            "intermediate_chunks": intermediate_chunks
        }

        print(f"\nTranscription complete in {time.time() - start_time:.2f}s")
        print(f"Transcribed text: {result['text'][:100]}...")
        return output

    def process_audio_steps(self,
                            audio_path: str,
                            apply_bandpass: bool = False,
                            apply_spectral_noise_reduction: bool = False,
                            apply_deepfilternet: bool = False,
                            apply_vad: bool = False,
                            vad_threshold: float = 0.5,
                            noise_reduction_prop: float = 0.75,
                            output_dir: str = "output/testing") -> Dict:
        """
        Run the pipeline step-by-step for testing and return paths to the intermediate files.
        """
        os.makedirs(output_dir, exist_ok=True)
        results = {}

        # 1. Original
        audio, sr = self.load_audio(audio_path)
        orig_path = os.path.join(output_dir, "step_0_original.wav")
        sf.write(orig_path, audio, sr)
        results['original'] = orig_path

        current_audio = audio

        # 2. Bandpass Filter
        if apply_bandpass:
            current_audio = self.apply_bandpass_filter(current_audio, lowcut=100.0, highcut=8000.0)
            bp_path = os.path.join(output_dir, "step_1_bandpass.wav")
            sf.write(bp_path, current_audio, sr)
            results['bandpass'] = bp_path

        # 3. Spectral Noise Reduction
        if apply_spectral_noise_reduction:
            current_audio = self.remove_noise(current_audio, prop_decrease=noise_reduction_prop)
            snr_path = os.path.join(output_dir, "step_2_spectral_nr.wav")
            sf.write(snr_path, current_audio, sr)
            results['spectral_nr'] = snr_path

        # 4. DeepFilterNet (skipped)
        if apply_deepfilternet:
            print("Skipping DeepFilterNet step.")

        # Save final preprocessed
        final_processed_path = os.path.join(output_dir, "step_final_processed.wav")
        sf.write(final_processed_path, current_audio, sr)
        results['final_processed'] = final_processed_path

        # 5. Transcribe final processed audio
        transcription_result = self.transcribe(final_processed_path)
        results['transcription'] = transcription_result.get('text', '')

        return results

    def save_transcription(self, result: Dict, output_path: str):
        """Save transcription result to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved transcription to: {output_path}")


def main():
    """CLI usage of the pipeline"""
    parser = argparse.ArgumentParser(description="ATC Audio Transcription Pipeline (CrispASR CPU/CUDA)")
    parser.add_argument("audio_file", help="Path to input audio file")
    parser.add_argument("--model", default=None, help="Path to local GGUF model (auto-detected if omitted)")
    parser.add_argument("--vad_model", default=None, help="Path to Silero VAD .bin (auto-detected in models/vad/ if omitted)")
    parser.add_argument("--exe_path", default=None, help="Path to crispasr executable (auto-detected from bin/cpu or bin/cuda)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Compute device (auto, cuda, cpu)")
    parser.add_argument("--output_dir", default="output", help="Directory to save output")
    parser.add_argument("--no_noise_removal", action="store_true", help="Disable noise removal")
    parser.add_argument("--noise_reduction", type=float, default=0.75, help="Noise reduction amount (default: 0.75)")
    parser.add_argument("--language", default="en", help="Language code")

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = AudioTranscriptionPipeline(
        exe_path=args.exe_path,
        model_path=args.model,
        vad_model_path=args.vad_model,
        device=args.device,
        sample_rate=16000
    )

    os.makedirs(args.output_dir, exist_ok=True)

    result = pipeline.process_audio(
        audio_path=args.audio_file,
        apply_noise_removal=not args.no_noise_removal,
        noise_reduction=args.noise_reduction,
        language=args.language,
        save_intermediate=True,
        output_dir=args.output_dir
    )

    # Save transcription
    timestamp = result.get("timestamp", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_path = os.path.join(args.output_dir, f"{Path(args.audio_file).stem}_transcription_{timestamp}.json")
    pipeline.save_transcription(result, output_path)

    return result


if __name__ == "__main__":
    main()