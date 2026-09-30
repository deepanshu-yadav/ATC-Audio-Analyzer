"""
Script to run transcription on a single audio file with automatic chunking support
Automatically determines optimal chunking settings based on audio duration
"""

import argparse
import os
import json
import librosa
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from transcribe_atc import AudioTranscriptionPipeline
from audio_chunker import AudioChunker, ParallelAudioProcessor, ChunkConfig, AudioChunk


def get_audio_duration(audio_path: str, sample_rate: int = 16000) -> float:
    """
    Get the duration of an audio file in seconds.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
        
    Returns:
        Duration in seconds
    """
    try:
        duration = librosa.get_duration(path=audio_path)
        return duration
    except Exception as e:
        print(f"Warning: Could not get audio duration: {e}")
        # Fallback: load audio and calculate
        audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        return len(audio) / sr


def determine_chunking_config(audio_duration: float, num_workers: Optional[int] = None) -> Tuple[bool, float, float, int]:
    """
    Automatically determine optimal chunking configuration based on audio duration.
    
    Args:
        audio_duration: Duration of audio in seconds
        num_workers: Optional manual override for number of workers
        
    Returns:
        Tuple of (should_chunk, chunk_duration, overlap_duration, num_workers)
    """
    import multiprocessing
    
    # Determine if chunking should be enabled
    CHUNKING_THRESHOLD = 120  # 2 minutes
    should_chunk = audio_duration > CHUNKING_THRESHOLD
    
    if not should_chunk:
        return False, 0, 0, 1
    
    # Determine optimal chunk duration based on total duration
    if audio_duration <= 300:  # 5 minutes
        chunk_duration = 10.0
        overlap_duration = 1.0
    elif audio_duration <= 900:  # 15 minutes
        chunk_duration = 20.0
        overlap_duration = 2
    elif audio_duration <= 1800:  # 30 minutes
        chunk_duration = 30
        overlap_duration = 5
    else:  # > 30 minutes
        chunk_duration = 40
        overlap_duration = 7
    
    # Determine optimal number of workers
    if num_workers is None:
        cpu_count = multiprocessing.cpu_count()
        # Use up to CPU count, but cap at 8 for most cases
        num_workers = min(cpu_count, 4)
        
        # For very short files, use fewer workers
        if audio_duration <= 300:
            num_workers = min(num_workers, 2)
    print(f"✓ Chunking enabled: {should_chunk}, chunk duration: {chunk_duration}, overlap: {overlap_duration}, workers: {num_workers}")
    return should_chunk, chunk_duration, overlap_duration, num_workers



def transcribe_chunk(chunk: AudioChunk, pipeline_config: dict) -> dict:
    """
    Transcribe a single audio chunk.
    
    Args:
        chunk: AudioChunk to transcribe
        pipeline_config: Configuration for the transcription pipeline
        
    Returns:
        Dictionary with transcription results
    """
    # Initialize pipeline for this worker process
    if pipeline_config['backend'] == "parakeet":
        pipeline = AudioTranscriptionPipeline(
            backend="parakeet",
            parakeet_model_dir=pipeline_config['parakeet_model_dir'],
            sample_rate=pipeline_config['sample_rate'],
            use_gpu=False
        )
    else:
        pipeline = AudioTranscriptionPipeline(
            backend="whisper",
            whisper_model=pipeline_config.get('whisper_model', 'base'),
            sample_rate=pipeline_config['sample_rate'],
            use_gpu=None
        )
    
    # Transcribe the chunk audio directly (no VAD/noise removal per chunk)
    # The audio has already been preprocessed if needed
    result = pipeline.transcribe(
        chunk.audio_data,
        language=pipeline_config.get('language', 'en')
    )
    
    print(f"✓ Chunk {chunk.chunk_id}: {chunk.start_time:.2f}s - {chunk.end_time:.2f}s transcribed")
    
    return {
        'chunk_id': chunk.chunk_id,
        'start_time': chunk.start_time,
        'end_time': chunk.end_time,
        'text': result['text'],
        'segments': result.get('segments', []),
        'language': result.get('language', pipeline_config.get('language', 'en'))
    }


def merge_chunk_transcriptions(chunk_results: List[dict], overlap_duration: float) -> dict:
    """
    Merge transcriptions from overlapping chunks.
    
    Args:
        chunk_results: List of transcription results from each chunk
        overlap_duration: Duration of overlap in seconds
        
    Returns:
        Combined transcription result
    """
    if not chunk_results:
        return {'text': '', 'segments': [], 'num_chunks': 0}
    
    merged_text_parts = []
    merged_segments = []
    
    for i, chunk_result in enumerate(chunk_results):
        chunk_text = chunk_result['text'].strip()
        start_time = chunk_result['start_time']
        end_time = chunk_result['end_time']
        
        # For all chunks except the last one, calculate effective end time
        if i < len(chunk_results) - 1:
            effective_end = end_time - overlap_duration
            
            # Create segment with adjusted timestamp
            segment = {
                'start': start_time,
                'end': effective_end,
                'text': chunk_text,
                'chunk_id': chunk_result['chunk_id']
            }
        else:
            # Last chunk: use full duration
            segment = {
                'start': start_time,
                'end': end_time,
                'text': chunk_text,
                'chunk_id': chunk_result['chunk_id']
            }
        
        if chunk_text:  # Only add non-empty transcriptions
            merged_text_parts.append(chunk_text)
            merged_segments.append(segment)
    
    return {
        'text': ' '.join(merged_text_parts),
        'segments': merged_segments,
        'num_chunks': len(chunk_results)
    }


def transcribe_with_chunking(
    audio_path: str,
    pipeline_config: dict,
    chunk_duration: float,
    overlap_duration: float,
    num_workers: int,
    output_dir: str,
    apply_vad: bool,
    apply_noise_removal: bool
) -> dict:
    """
    Transcribe audio file using chunking and parallel processing.
    
    Args:
        audio_path: Path to audio file
        pipeline_config: Configuration for transcription pipeline
        chunk_duration: Duration of each chunk in seconds
        overlap_duration: Overlap between chunks in seconds
        num_workers: Number of parallel workers
        output_dir: Directory for intermediate files
        apply_vad: Whether to apply VAD preprocessing
        apply_noise_removal: Whether to apply noise removal preprocessing
        
    Returns:
        Complete transcription result
    """
    import time
    import librosa
    import soundfile as sf
    
    start_time = time.time()
    
    # Step 1: Load and preprocess audio if needed
    print(f"Loading audio: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=pipeline_config['sample_rate'], mono=True)
    
    # Apply preprocessing (VAD and noise removal) to the full audio first
    if apply_vad or apply_noise_removal:
        print("\nApplying preprocessing to full audio...")
        temp_pipeline = AudioTranscriptionPipeline(
            backend=pipeline_config['backend'],
            parakeet_model_dir=pipeline_config.get('parakeet_model_dir'),
            whisper_model=pipeline_config.get('whisper_model', 'base'),
            sample_rate=pipeline_config['sample_rate'],
            use_gpu=False if pipeline_config['backend'] == 'parakeet' else None
        )
        
        # Apply noise removal
        if apply_noise_removal:
            print("Applying noise removal...")
            audio = temp_pipeline.apply_bandpass_filter(audio, lowcut=100.0, highcut=8000.0)
            audio = temp_pipeline.remove_noise(audio, prop_decrease=0.75)
        
        # Apply VAD
        if apply_vad:
            print("Applying VAD...")
            speech_timestamps = temp_pipeline.apply_vad(audio)
            audio = temp_pipeline.extract_speech_segments(audio, speech_timestamps)
        
        # Save preprocessed audio for chunking
        preprocessed_path = os.path.join(output_dir, "preprocessed_for_chunking.wav")
        sf.write(preprocessed_path, audio, pipeline_config['sample_rate'])
        print(f"Saved preprocessed audio: {preprocessed_path}")
        audio_to_chunk = preprocessed_path
    else:
        audio_to_chunk = audio_path
    
    # Step 2: Configure chunking
    config = ChunkConfig(
        chunk_duration=chunk_duration,
        overlap_duration=overlap_duration,
        sample_rate=pipeline_config['sample_rate']
    )
    
    chunker = AudioChunker(config)
    processor = ParallelAudioProcessor(chunker, num_workers=num_workers)
    
    print(f"\n{'='*60}")
    print(f"Chunking Configuration:")
    print(f"  Chunk duration: {chunk_duration}s")
    print(f"  Overlap duration: {overlap_duration}s")
    print(f"  Sample rate: {pipeline_config['sample_rate']} Hz")
    print(f"  Workers: {num_workers}")
    print(f"{'='*60}\n")
    
    # Step 3: Process chunks in parallel
    results = processor.process_audio(
        audio_path=audio_to_chunk,
        processing_func=transcribe_chunk,
        pipeline_config=pipeline_config
    )
    
    # Step 4: Merge results
    print("\nMerging chunk transcriptions...")
    final_result = processor.combine_results(
        results,
        lambda chunks: merge_chunk_transcriptions(chunks, overlap_duration)
    )
    
    # Add metadata
    final_result['processing_time_seconds'] = time.time() - start_time
    final_result['vad_enabled'] = apply_vad
    final_result['noise_removal_enabled'] = apply_noise_removal
    final_result['chunking_enabled'] = True
    final_result['chunk_duration'] = chunk_duration
    final_result['overlap_duration'] = overlap_duration
    final_result['num_workers'] = num_workers
    final_result['language'] = pipeline_config.get('language', 'en')
    
    return final_result


def transcribe_audio(
    audio_file: str,
    parakeet_model_dir: str = "float16_onnx",
    output_dir: str = "output",
    backend: str = "parakeet",
    language: str = "en",
    enable_vad: bool = True,
    enable_noise_removal: bool = True,
    disable_chunking: bool = False,
    chunk_duration: Optional[float] = None,
    overlap_duration: Optional[float] = None,
    num_workers: Optional[int] = None
) -> dict:
    """
    Transcribe an audio file with automatic chunking optimization.
    
    Args:
        audio_file: Path to audio file to transcribe
        parakeet_model_dir: Path to Parakeet model directory (default: "float16_onnx")
        output_dir: Output directory for results (default: "output")
        backend: Transcription backend - "parakeet" or "whisper" (default: "parakeet")
        language: Language code (default: "en")
        enable_vad: Enable Voice Activity Detection (default: True)
        enable_noise_removal: Enable noise removal (default: True)
        disable_chunking: Force disable automatic chunking (default: False)
        chunk_duration: Override chunk duration in seconds (auto-detected by default)
        overlap_duration: Override overlap duration in seconds (auto-detected by default)
        num_workers: Override number of parallel workers (auto-detected by default)
        
    Returns:
        Dictionary containing transcription results
    """
    # Validate input file
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get audio duration
    print(f"Analyzing audio file: {audio_file}")
    audio_duration = get_audio_duration(audio_file)
    print(f"Audio duration: {audio_duration:.2f} seconds ({audio_duration/60:.2f} minutes)\n")
    
    # Determine chunking configuration automatically
    if disable_chunking:
        should_chunk = False
        chunk_duration_final = 0
        overlap_duration_final = 0
        num_workers_final = 1
        print("Chunking: DISABLED (forced by user)")
    else:
        should_chunk, chunk_duration_final, overlap_duration_final, num_workers_final = determine_chunking_config(
            audio_duration, 
            num_workers=num_workers
        )
        
        # Apply manual overrides if provided
        if chunk_duration is not None:
            chunk_duration_final = chunk_duration
        if overlap_duration is not None:
            overlap_duration_final = overlap_duration
        
        if should_chunk:
            print(f"Chunking: ENABLED (auto-detected)")
            print(f"  Chunk duration: {chunk_duration_final}s")
            print(f"  Overlap: {overlap_duration_final}s")
            print(f"  Workers: {num_workers_final}")
        else:
            print(f"Chunking: DISABLED (audio too short, < 2 minutes)")

    # If VAD is enabled, prefer VAD-driven segmentation for paragraphing
    if enable_vad:
        print("VAD enabled - using VAD-based segmentation (silence split threshold: 3s) for paragraphs.")
        # Initialize a pipeline instance for VAD + per-segment transcription
        temp_pipeline = AudioTranscriptionPipeline(
            backend=backend,
            parakeet_model_dir=parakeet_model_dir,
            sample_rate=16000,
            use_gpu=False if backend == 'parakeet' else None
        )

        # Load full audio (numpy array)
        audio, sr = temp_pipeline.load_audio(audio_file)

        # Optional preprocessing: noise removal & bandpass
        if enable_noise_removal:
            print("Applying bandpass & noise removal before VAD segmentation...")
            audio = temp_pipeline.apply_bandpass_filter(audio, lowcut=100.0, highcut=8000.0)
            audio = temp_pipeline.remove_noise(audio, prop_decrease=0.75)

        # Use VAD to detect speech segments, but treat silences >= 3000ms as paragraph breaks
        speech_timestamps = temp_pipeline.apply_vad(
            audio,
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=3000
        )

        # If VAD returned segments, transcribe each segment individually and keep timestamps
        if speech_timestamps:
            print(f"Transcribing {len(speech_timestamps)} VAD segments...")
            import time
            start_time = time.time()

            segments = []
            total_speech_samples = 0
            for i, seg in enumerate(speech_timestamps):
                s = int(seg['start'])
                e = int(seg['end'])
                total_speech_samples += (e - s)
                segment_audio = audio[s:e]

                # Transcribe segment
                seg_result = temp_pipeline.transcribe(segment_audio, language=language)
                seg_text = seg_result.get('text', '').strip()

                segments.append({
                    'start': round(s / sr, 2),
                    'end': round(e / sr, 2),
                    'text': seg_text,
                    'chunk_id': i  # Assign chunk_id based on index
                })

            full_text = "\n\n".join([g['text'] for g in segments]).strip()

            result = {
                'text': full_text,
                'segments': segments,
                'processing_time_seconds': time.time() - start_time,
                'speech_duration_seconds': round(total_speech_samples / sr, 2),
                'vad_enabled': True,
                'noise_removal_enabled': enable_noise_removal,
                'chunking_enabled': False,
                'num_chunks': len(segments),
                'chunk_duration': None,
                'overlap_duration': None,
                'num_workers': 1,
                'language': language
            }

            return result
        else:
            print("VAD detected no speech segments; falling back to normal chunking/transcription.")

    
    print(f"\nBackend: {backend}")
    print(f"VAD enabled: {enable_vad}")
    print(f"Noise removal enabled: {enable_noise_removal}")
    print()
    
    # Prepare pipeline configuration
    pipeline_config = {
        'backend': backend,
        'sample_rate': 16000,
        'language': language,
        'parakeet_model_dir': parakeet_model_dir,
        'whisper_model': 'base'
    }
    
    # Choose processing mode
    if should_chunk:
        # Process with chunking and multiprocessing
        result = transcribe_with_chunking(
            audio_path=audio_file,
            pipeline_config=pipeline_config,
            chunk_duration=chunk_duration_final,
            overlap_duration=overlap_duration_final,
            num_workers=num_workers_final,
            output_dir=output_dir,
            apply_vad=enable_vad,
            apply_noise_removal=enable_noise_removal
        )
    else:
        # Standard single-process mode
        if backend == "parakeet":
            pipeline = AudioTranscriptionPipeline(
                backend="parakeet",
                parakeet_model_dir=parakeet_model_dir,
                sample_rate=16000,
                use_gpu=False
            )
        else:
            pipeline = AudioTranscriptionPipeline(
                backend="whisper",
                whisper_model="base",
                sample_rate=16000,
                use_gpu=None
            )
        
        result = pipeline.process_audio(
            audio_path=audio_file,
            apply_vad=enable_vad,
            apply_noise_removal=enable_noise_removal,
            language=language,
            save_intermediate=True,
            output_dir=output_dir
        )
    
    # Save transcription result
    output_filename = f"{Path(audio_file).stem}_transcription.json"
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved transcription to: {output_path}")
    
    # Display results
    print("\n" + "="*60)
    print("TRANSCRIPTION RESULT")
    print("="*60)
    print(f"\nText: {result['text']}")
    print(f"\nProcessing time: {result['processing_time_seconds']:.2f}s")
    print(f"Audio duration: {audio_duration:.2f}s")
    
    if should_chunk:
        print(f"\nChunking details:")
        print(f"  Total chunks processed: {result['num_chunks']}")
        print(f"  Chunk duration: {result['chunk_duration']}s")
        print(f"  Overlap: {result['overlap_duration']}s")
        print(f"  Workers used: {result['num_workers']}")
    
    if result.get('speech_duration_seconds'):
        print(f"Speech duration: {result['speech_duration_seconds']:.2f}s")
    
    print(f"\nVAD enabled: {result['vad_enabled']}")
    print(f"Noise removal enabled: {result['noise_removal_enabled']}")
    print("="*60)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run ATC Transcription with automatic chunking optimization"
    )
    parser.add_argument("audio_file", help="Path to audio file to transcribe")
    parser.add_argument("--parakeet_model_dir", default="float16_onnx", 
                       help="Path to Parakeet model directory (default: float16_onnx)")
    parser.add_argument("--output_dir", default="output", 
                       help="Output directory for results (default: output)")
    parser.add_argument("--no_vad", action="store_true", 
                       help="Disable Voice Activity Detection")
    parser.add_argument("--no_noise_removal", action="store_true", 
                       help="Disable noise removal")
    parser.add_argument("--language", default="en", 
                       help="Language code (default: en)")
    parser.add_argument("--backend", default="parakeet", choices=["parakeet", "whisper"],
                       help="Transcription backend (default: parakeet)")
    
    # Chunking options (manual overrides for automatic settings)
    parser.add_argument("--disable_chunking", action="store_true",
                       help="Force disable automatic chunking (process entire file at once)")
    parser.add_argument("--chunk_duration", type=float, default=None,
                       help="Override chunk duration in seconds (auto-detected by default)")
    parser.add_argument("--overlap_duration", type=float, default=None,
                       help="Override overlap duration in seconds (auto-detected by default)")
    parser.add_argument("--num_workers", type=int, default=None,
                       help="Override number of parallel workers (auto-detected by default)")
    
    args = parser.parse_args()
    
    # Call the main transcribe_audio function with args
    try:
        result = transcribe_audio(
            audio_file=args.audio_file,
            parakeet_model_dir=args.parakeet_model_dir,
            output_dir=args.output_dir,
            backend=args.backend,
            language=args.language,
            enable_vad=not args.no_vad,
            enable_noise_removal=not args.no_noise_removal,
            disable_chunking=args.disable_chunking,
            chunk_duration=args.chunk_duration,
            overlap_duration=args.overlap_duration,
            num_workers=args.num_workers
        )
        return 0
    except Exception as e:
        print(f"Error during transcription: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Example usage when called directly (not as CLI)
    import sys
    
    # Check if we're being run with command-line arguments
    if len(sys.argv) > 1:
        # CLI mode
        exit(main())
    else:
        # Direct call mode - test with sample2.mp3
        print("Running in test mode with sample2.mp3\n")
        try:
            result = transcribe_audio("sample2.mp3")
            print("\n✅ Transcription completed successfully!")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
