"""
Audio Chunking Module with Multiprocessing Support

This module provides functionality to split audio files into overlapping chunks
and process them in parallel using multiprocessing.
"""

import os
import numpy as np
from typing import List, Tuple, Callable, Any, Optional
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
import librosa
import soundfile as sf


@dataclass
class ChunkConfig:
    """Configuration for audio chunking."""
    chunk_duration: float  # Duration of each chunk in seconds
    overlap_duration: float  # Overlap duration between chunks in seconds
    sample_rate: int = 16000  # Sample rate for audio processing


@dataclass
class AudioChunk:
    """Represents a single audio chunk with metadata."""
    audio_data: np.ndarray
    chunk_id: int
    start_time: float
    end_time: float
    sample_rate: int


class AudioChunker:
    """
    Handles splitting audio files into overlapping chunks and parallel processing.
    """
    
    def __init__(self, config: ChunkConfig):
        """
        Initialize the AudioChunker.
        
        Args:
            config: Configuration for chunking parameters
        """
        self.config = config
        
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        audio_data, sr = librosa.load(audio_path, sr=self.config.sample_rate)
        return audio_data, sr
    
    def split_into_chunks(self, audio_data: np.ndarray, sample_rate: int) -> List[AudioChunk]:
        """
        Split audio into overlapping chunks.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            List of AudioChunk objects
        """
        chunk_samples = int(self.config.chunk_duration * sample_rate)
        overlap_samples = int(self.config.overlap_duration * sample_rate)
        step_samples = chunk_samples - overlap_samples
        
        total_samples = len(audio_data)
        chunks = []
        chunk_id = 0
        
        start_sample = 0
        while start_sample < total_samples:
            end_sample = min(start_sample + chunk_samples, total_samples)
            
            # Extract chunk
            chunk_audio = audio_data[start_sample:end_sample]
            
            # Pad last chunk if necessary
            if len(chunk_audio) < chunk_samples and start_sample > 0:
                # Only pad if it's not the last chunk and too small
                padding = chunk_samples - len(chunk_audio)
                chunk_audio = np.pad(chunk_audio, (0, padding), mode='constant')
            
            # Create chunk object
            chunk = AudioChunk(
                audio_data=chunk_audio,
                chunk_id=chunk_id,
                start_time=start_sample / sample_rate,
                end_time=end_sample / sample_rate,
                sample_rate=sample_rate
            )
            chunks.append(chunk)
            
            chunk_id += 1
            start_sample += step_samples
            
            # Break if we've reached the end
            if end_sample >= total_samples:
                break
        
        return chunks
    
    def save_chunk(self, chunk: AudioChunk, output_dir: str) -> str:
        """
        Save a single chunk to disk (useful for debugging).
        
        Args:
            chunk: AudioChunk to save
            output_dir: Directory to save the chunk
            
        Returns:
            Path to saved chunk file
        """
        os.makedirs(output_dir, exist_ok=True)
        chunk_path = os.path.join(
            output_dir, 
            f"chunk_{chunk.chunk_id:04d}_{chunk.start_time:.2f}s-{chunk.end_time:.2f}s.wav"
        )
        sf.write(chunk_path, chunk.audio_data, chunk.sample_rate)
        return chunk_path


def process_chunk_wrapper(args: Tuple[AudioChunk, Callable, dict]) -> Tuple[int, Any]:
    """
    Wrapper function for multiprocessing to process a single chunk.
    
    Args:
        args: Tuple of (chunk, processing_function, kwargs)
        
    Returns:
        Tuple of (chunk_id, result)
    """
    chunk, processing_func, kwargs = args
    result = processing_func(chunk, **kwargs)
    return chunk.chunk_id, result


class ParallelAudioProcessor:
    """
    Handles parallel processing of audio chunks using multiprocessing.
    """
    
    def __init__(self, chunker: AudioChunker, num_workers: Optional[int] = None):
        """
        Initialize the parallel processor.
        
        Args:
            chunker: AudioChunker instance
            num_workers: Number of worker processes (defaults to CPU count)
        """
        self.chunker = chunker
        self.num_workers = num_workers or cpu_count()
    
    def process_audio(
        self, 
        audio_path: str, 
        processing_func: Callable[[AudioChunk], Any],
        **kwargs
    ) -> List[Tuple[int, Any]]:
        """
        Process audio file in parallel chunks.
        
        Args:
            audio_path: Path to audio file
            processing_func: Function to apply to each chunk
                            Should accept AudioChunk as first argument
            **kwargs: Additional keyword arguments to pass to processing_func
            
        Returns:
            List of tuples (chunk_id, result) sorted by chunk_id
        """
        # Load and split audio
        audio_data, sample_rate = self.chunker.load_audio(audio_path)
        chunks = self.chunker.split_into_chunks(audio_data, sample_rate)
        
        print(f"Split audio into {len(chunks)} chunks")
        print(f"Processing with {self.num_workers} workers...")
        
        # Prepare arguments for multiprocessing
        process_args = [(chunk, processing_func, kwargs) for chunk in chunks]
        
        # Process chunks in parallel
        with Pool(processes=self.num_workers) as pool:
            results = pool.map(process_chunk_wrapper, process_args)
        
        # Sort results by chunk_id
        results.sort(key=lambda x: x[0])
        
        return results
    
    def combine_results(
        self, 
        results: List[Tuple[int, Any]], 
        combine_func: Callable[[List[Any]], Any]
    ) -> Any:
        """
        Combine results from all chunks.
        
        Args:
            results: List of (chunk_id, result) tuples
            combine_func: Function to combine all chunk results
            
        Returns:
            Combined result
        """
        chunk_results = [result for _, result in results]
        return combine_func(chunk_results)


# ==================== Example Usage Functions ====================

def example_processing_function(chunk: AudioChunk, **kwargs) -> dict:
    """
    Example processing function that extracts audio features.
    Replace this with your actual processing logic (e.g., transcription).
    
    Args:
        chunk: AudioChunk to process
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with processing results
    """
    # Example: Calculate some audio statistics
    audio = chunk.audio_data
    
    result = {
        'chunk_id': chunk.chunk_id,
        'start_time': chunk.start_time,
        'end_time': chunk.end_time,
        'duration': chunk.end_time - chunk.start_time,
        'rms_energy': np.sqrt(np.mean(audio**2)),
        'max_amplitude': np.max(np.abs(audio)),
        'mean_amplitude': np.mean(np.abs(audio)),
        # Add your custom processing here
        # For example, you could run transcription:
        # 'transcription': your_transcription_model(audio, chunk.sample_rate)
    }
    
    return result


def example_combine_function(chunk_results: List[dict]) -> dict:
    """
    Example function to combine results from all chunks.
    
    Args:
        chunk_results: List of results from each chunk
        
    Returns:
        Combined result dictionary
    """
    combined = {
        'total_chunks': len(chunk_results),
        'total_duration': chunk_results[-1]['end_time'] if chunk_results else 0,
        'chunk_results': chunk_results,
        'average_rms_energy': np.mean([r['rms_energy'] for r in chunk_results]),
        'max_amplitude_overall': max([r['max_amplitude'] for r in chunk_results]),
    }
    
    return combined


def main():
    """Example usage of the audio chunking and parallel processing system."""
    
    # Configuration
    config = ChunkConfig(
        chunk_duration=30.0,  # 30 seconds per chunk
        overlap_duration=5.0,  # 5 seconds overlap
        sample_rate=16000     # 16kHz sample rate
    )
    
    # Initialize chunker and processor
    chunker = AudioChunker(config)
    processor = ParallelAudioProcessor(chunker, num_workers=4)
    
    # Example audio file path - replace with your actual file
    audio_path = "path/to/your/audio.wav"
    
    # Process audio in parallel
    results = processor.process_audio(
        audio_path=audio_path,
        processing_func=example_processing_function,
        # Add any additional kwargs for your processing function here
    )
    
    # Combine results
    combined_result = processor.combine_results(results, example_combine_function)
    
    print("\n=== Processing Complete ===")
    print(f"Total chunks processed: {combined_result['total_chunks']}")
    print(f"Total duration: {combined_result['total_duration']:.2f} seconds")
    print(f"Average RMS energy: {combined_result['average_rms_energy']:.6f}")
    
    # Print individual chunk results
    print("\nChunk Details:")
    for chunk_result in combined_result['chunk_results']:
        print(f"  Chunk {chunk_result['chunk_id']}: "
              f"{chunk_result['start_time']:.2f}s - {chunk_result['end_time']:.2f}s, "
              f"RMS: {chunk_result['rms_energy']:.6f}")
    
    return combined_result


if __name__ == "__main__":
    main()
