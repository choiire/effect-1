"""
VAD-Gated Speech Volume Normalizer
Main processing pipeline that combines VAD and DRC for robust speech normalization.
"""

import numpy as np
import librosa
import soundfile as sf
from typing import List, Tuple, Optional
from tqdm import tqdm

from vad_module import RobustVAD
from drc_engine import DynamicRangeCompressor, SmoothCrossfader


class SpeechVolumeNormalizer:
    """
    Main class that combines VAD and DRC for speech volume normalization.
    Only applies compression to detected speech segments.
    """
    
    def __init__(self, 
                 # VAD parameters
                 vad_aggressiveness: int = 2,
                 vad_frame_duration_ms: int = 20,
                 vad_hangover_ms: int = 200,
                 # DRC parameters
                 threshold_db: float = -20.0,
                 ratio: float = 4.0,
                 attack_ms: float = 10.0,
                 release_ms: float = 100.0,
                 # Processing parameters
                 crossfade_ms: float = 10.0,
                 block_size: int = 4096):
        """
        Initialize the speech volume normalizer.
        
        Args:
            vad_aggressiveness: VAD aggressiveness (0-3)
            vad_frame_duration_ms: VAD frame duration in ms
            vad_hangover_ms: VAD hangover time in ms
            threshold_db: DRC threshold in dB
            ratio: DRC compression ratio
            attack_ms: DRC attack time in ms
            release_ms: DRC release time in ms
            crossfade_ms: Crossfade duration at segment boundaries in ms
            block_size: Audio processing block size
        """
        # Initialize VAD
        self.vad = RobustVAD(
            aggressiveness=vad_aggressiveness,
            frame_duration_ms=vad_frame_duration_ms,
            hangover_ms=vad_hangover_ms
        )
        
        # Initialize DRC
        self.compressor = DynamicRangeCompressor(
            threshold_db=threshold_db,
            ratio=ratio,
            attack_ms=attack_ms,
            release_ms=release_ms
        )
        
        # Processing parameters
        self.crossfade_ms = crossfade_ms
        self.block_size = block_size
        
    def _time_to_samples(self, time_sec: float, sample_rate: int) -> int:
        """Convert time in seconds to sample index."""
        return int(time_sec * sample_rate)
    
    def _is_in_speech_segment(self, sample_idx: int, speech_segments: List[Tuple[float, float]], 
                             sample_rate: int) -> bool:
        """Check if a sample index falls within any speech segment."""
        current_time = sample_idx / sample_rate
        
        for start_time, end_time in speech_segments:
            if start_time <= current_time <= end_time:
                return True
        return False
    
    def process_audio_file(self, input_path: str, output_path: str, 
                          show_progress: bool = True) -> dict:
        """
        Process an audio file with VAD-gated compression.
        
        Args:
            input_path: Path to input audio file
            output_path: Path to output audio file
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary with processing statistics
        """
        print(f"Loading audio file: {input_path}")
        
        # Load original audio
        audio, sample_rate = librosa.load(input_path, sr=None, mono=True)
        total_samples = len(audio)
        total_duration = total_samples / sample_rate
        
        print(f"Audio loaded: {total_duration:.2f}s, {sample_rate}Hz, {total_samples} samples")
        
        # Detect speech segments using VAD
        print("Detecting speech segments...")
        speech_segments = self.vad.detect_speech_segments(input_path)
        
        if not speech_segments:
            print("No speech segments detected. Copying original audio.")
            sf.write(output_path, audio, sample_rate)
            return {
                'total_duration': total_duration,
                'speech_duration': 0.0,
                'speech_ratio': 0.0,
                'segments_count': 0
            }
        
        # Calculate speech statistics
        speech_duration = sum(end - start for start, end in speech_segments)
        speech_ratio = speech_duration / total_duration
        
        print(f"Found {len(speech_segments)} speech segments")
        print(f"Speech duration: {speech_duration:.2f}s ({speech_ratio:.1%} of total)")
        
        # Prepare compressor
        self.compressor.prepare(sample_rate)
        
        # Calculate crossfade samples
        crossfade_samples = int(self.crossfade_ms * sample_rate / 1000)
        
        # Initialize output buffer
        output_audio = audio.copy()
        
        # Process each speech segment
        print("Processing speech segments...")
        
        if show_progress:
            segment_iterator = tqdm(speech_segments, desc="Processing segments")
        else:
            segment_iterator = speech_segments
        
        for segment_idx, (start_time, end_time) in enumerate(segment_iterator):
            # Convert times to sample indices
            start_sample = self._time_to_samples(start_time, sample_rate)
            end_sample = self._time_to_samples(end_time, sample_rate)
            
            # Ensure indices are within bounds
            start_sample = max(0, start_sample)
            end_sample = min(total_samples, end_sample)
            
            if start_sample >= end_sample:
                continue
            
            # Extract segment audio
            segment_audio = audio[start_sample:end_sample]
            
            # Apply compression to segment
            compressed_segment = self.compressor.process_audio(segment_audio, sample_rate)
            
            # Apply smooth crossfade at boundaries
            output_audio = SmoothCrossfader.smooth_transition(
                unprocessed=output_audio,
                processed=output_audio.copy(),  # We'll update this
                start_idx=start_sample,
                end_idx=end_sample,
                fade_samples=crossfade_samples
            )
            
            # Insert compressed segment with crossfade
            output_audio[start_sample:end_sample] = compressed_segment
            
            # Apply crossfade at boundaries
            if crossfade_samples > 0:
                # Fade in at start
                if start_sample > crossfade_samples:
                    fade_start = start_sample - crossfade_samples
                    fade_end = start_sample + crossfade_samples
                    fade_length = min(crossfade_samples * 2, end_sample - start_sample)
                    
                    if fade_length > 0:
                        fade_curve = np.linspace(0, 1, fade_length)
                        blend_start = start_sample
                        blend_end = start_sample + fade_length
                        
                        output_audio[blend_start:blend_end] = (
                            audio[blend_start:blend_end] * (1 - fade_curve) +
                            compressed_segment[:fade_length] * fade_curve
                        )
                
                # Fade out at end
                if end_sample < total_samples - crossfade_samples:
                    fade_length = min(crossfade_samples * 2, end_sample - start_sample)
                    
                    if fade_length > 0:
                        fade_curve = np.linspace(1, 0, fade_length)
                        blend_start = end_sample - fade_length
                        blend_end = end_sample
                        
                        compressed_end = compressed_segment[-(fade_length):]
                        output_audio[blend_start:blend_end] = (
                            compressed_end * fade_curve +
                            audio[blend_start:blend_end] * (1 - fade_curve)
                        )
        
        # Save processed audio
        print(f"Saving processed audio to: {output_path}")
        sf.write(output_path, output_audio, sample_rate)
        
        # Calculate processing statistics
        original_rms = np.sqrt(np.mean(audio**2))
        processed_rms = np.sqrt(np.mean(output_audio**2))
        
        stats = {
            'total_duration': total_duration,
            'speech_duration': speech_duration,
            'speech_ratio': speech_ratio,
            'segments_count': len(speech_segments),
            'original_rms': original_rms,
            'processed_rms': processed_rms,
            'rms_ratio': processed_rms / original_rms if original_rms > 0 else 1.0,
            'crossfade_ms': self.crossfade_ms
        }
        
        print(f"Processing complete!")
        print(f"Original RMS: {original_rms:.6f}")
        print(f"Processed RMS: {processed_rms:.6f}")
        print(f"RMS ratio: {stats['rms_ratio']:.3f}")
        
        return stats
    
    def analyze_audio(self, input_path: str) -> dict:
        """
        Analyze audio file without processing (for parameter tuning).
        
        Args:
            input_path: Path to input audio file
            
        Returns:
            Dictionary with analysis results
        """
        # Load audio
        audio, sample_rate = librosa.load(input_path, sr=None, mono=True)
        total_duration = len(audio) / sample_rate
        
        # Detect speech segments
        speech_segments = self.vad.detect_speech_segments(input_path)
        speech_duration = sum(end - start for start, end in speech_segments)
        
        # Calculate audio statistics
        rms = np.sqrt(np.mean(audio**2))
        peak = np.max(np.abs(audio))
        
        # Analyze speech segments
        speech_audio = []
        for start_time, end_time in speech_segments:
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            speech_audio.extend(audio[start_sample:end_sample])
        
        speech_rms = np.sqrt(np.mean(np.array(speech_audio)**2)) if speech_audio else 0.0
        speech_peak = np.max(np.abs(speech_audio)) if speech_audio else 0.0
        
        return {
            'total_duration': total_duration,
            'speech_duration': speech_duration,
            'speech_ratio': speech_duration / total_duration if total_duration > 0 else 0.0,
            'segments_count': len(speech_segments),
            'overall_rms': rms,
            'overall_peak': peak,
            'speech_rms': speech_rms,
            'speech_peak': speech_peak,
            'segments': speech_segments
        }


if __name__ == "__main__":
    # Test the normalizer
    normalizer = SpeechVolumeNormalizer(
        threshold_db=-20.0,
        ratio=4.0,
        vad_aggressiveness=2
    )
    
    # Analyze input file
    print("=== Audio Analysis ===")
    analysis = normalizer.analyze_audio("input.wav")
    
    print(f"Total duration: {analysis['total_duration']:.2f}s")
    print(f"Speech duration: {analysis['speech_duration']:.2f}s ({analysis['speech_ratio']:.1%})")
    print(f"Speech segments: {analysis['segments_count']}")
    print(f"Overall RMS: {analysis['overall_rms']:.6f}")
    print(f"Speech RMS: {analysis['speech_rms']:.6f}")
    
    # Process audio
    print("\n=== Processing Audio ===")
    stats = normalizer.process_audio_file("input.wav", "output_normalized.wav")
