"""
Dynamic Range Compression (DRC) Engine
Implements audio compression with RMS envelope following and configurable parameters.
"""

import numpy as np
from typing import Optional


class DynamicRangeCompressor:
    """
    Dynamic Range Compressor with RMS envelope following.
    Applies compression only when signal exceeds threshold.
    """
    
    def __init__(self, threshold_db: float = -20.0, ratio: float = 4.0, 
                 attack_ms: float = 10.0, release_ms: float = 100.0):
        """
        Initialize DRC with specified parameters.
        
        Args:
            threshold_db: Compression threshold in dB
            ratio: Compression ratio (e.g., 4:1 = 4.0)
            attack_ms: Attack time in milliseconds
            release_ms: Release time in milliseconds
        """
        self.threshold_db = threshold_db
        self.ratio = ratio
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        
        # Convert threshold to linear scale
        self.threshold_linear = self._db_to_linear(threshold_db)
        
        # State variables
        self.envelope = 0.0
        self.attack_coeff = 0.0
        self.release_coeff = 0.0
        self.sample_rate = None
        
    def _db_to_linear(self, db: float) -> float:
        """Convert dB to linear scale."""
        return 10.0 ** (db / 20.0)
    
    def _linear_to_db(self, linear: float) -> float:
        """Convert linear to dB scale."""
        return 20.0 * np.log10(max(linear, 1e-10))  # Avoid log(0)
    
    def prepare(self, sample_rate: int):
        """
        Prepare the compressor for processing at given sample rate.
        
        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        
        # Calculate attack and release coefficients
        # Formula: coeff = exp(-1.0 / (time_in_seconds * sample_rate))
        attack_time_sec = self.attack_ms / 1000.0
        release_time_sec = self.release_ms / 1000.0
        
        self.attack_coeff = np.exp(-1.0 / (attack_time_sec * sample_rate))
        self.release_coeff = np.exp(-1.0 / (release_time_sec * sample_rate))
        
        # Reset envelope
        self.envelope = 0.0
    
    def process_block(self, audio_block: np.ndarray) -> np.ndarray:
        """
        Process an audio block with dynamic range compression.
        
        Args:
            audio_block: Input audio block as numpy array
            
        Returns:
            Compressed audio block
        """
        if self.sample_rate is None:
            raise ValueError("Must call prepare() with sample rate before processing")
        
        output = np.zeros_like(audio_block)
        
        for i in range(len(audio_block)):
            # Calculate input power (RMS detection)
            input_sample = audio_block[i]
            input_power = input_sample * input_sample
            
            # Update envelope with attack/release
            if input_power > self.envelope:
                # Attack: signal is increasing
                self.envelope = input_power + self.attack_coeff * (self.envelope - input_power)
            else:
                # Release: signal is decreasing
                self.envelope = input_power + self.release_coeff * (self.envelope - input_power)
            
            # Convert envelope to dB for gain calculation
            envelope_rms = np.sqrt(max(self.envelope, 1e-10))
            envelope_db = self._linear_to_db(envelope_rms)
            
            # Calculate gain reduction
            if envelope_db > self.threshold_db:
                # Apply compression
                gain_reduction_db = (envelope_db - self.threshold_db) * (1.0/self.ratio - 1.0)
                gain_linear = self._db_to_linear(gain_reduction_db)
            else:
                # No compression needed
                gain_linear = 1.0
            
            # Apply gain to input sample
            output[i] = input_sample * gain_linear
        
        return output
    
    def process_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Process entire audio array with compression.
        
        Args:
            audio: Input audio as numpy array
            sample_rate: Audio sample rate in Hz
            
        Returns:
            Compressed audio array
        """
        self.prepare(sample_rate)
        return self.process_block(audio)


class SmoothCrossfader:
    """
    Utility class for smooth crossfading between processed and unprocessed audio.
    """
    
    @staticmethod
    def linear_crossfade(audio1: np.ndarray, audio2: np.ndarray, 
                        crossfade_samples: int) -> np.ndarray:
        """
        Apply linear crossfade between two audio signals.
        
        Args:
            audio1: First audio signal
            audio2: Second audio signal  
            crossfade_samples: Number of samples for crossfade
            
        Returns:
            Crossfaded audio
        """
        if len(audio1) != len(audio2):
            raise ValueError("Audio signals must have same length")
        
        if crossfade_samples <= 0:
            return audio2
        
        crossfade_samples = min(crossfade_samples, len(audio1))
        result = audio2.copy()
        
        # Create fade curve (0 to 1)
        fade_curve = np.linspace(0, 1, crossfade_samples)
        
        # Apply crossfade at the beginning
        result[:crossfade_samples] = (audio1[:crossfade_samples] * (1 - fade_curve) + 
                                     audio2[:crossfade_samples] * fade_curve)
        
        return result
    
    @staticmethod
    def smooth_transition(unprocessed: np.ndarray, processed: np.ndarray,
                         start_idx: int, end_idx: int, 
                         fade_samples: int = 441) -> np.ndarray:
        """
        Create smooth transition between unprocessed and processed audio.
        
        Args:
            unprocessed: Original audio
            processed: Processed audio
            start_idx: Start index of processed segment
            end_idx: End index of processed segment
            fade_samples: Number of samples for fade in/out
            
        Returns:
            Audio with smooth transitions
        """
        result = unprocessed.copy()
        
        # Ensure indices are within bounds
        start_idx = max(0, start_idx)
        end_idx = min(len(unprocessed), end_idx)
        fade_samples = min(fade_samples, (end_idx - start_idx) // 2)
        
        if start_idx >= end_idx or fade_samples <= 0:
            return result
        
        # Copy processed segment
        result[start_idx:end_idx] = processed[start_idx:end_idx]
        
        # Fade in at the beginning
        if start_idx > 0 and fade_samples > 0:
            fade_start = max(0, start_idx - fade_samples)
            fade_end = start_idx + fade_samples
            
            if fade_end <= end_idx:
                fade_length = fade_end - fade_start
                fade_curve = np.linspace(0, 1, fade_length)
                
                result[fade_start:fade_end] = (unprocessed[fade_start:fade_end] * (1 - fade_curve) +
                                             processed[fade_start:fade_end] * fade_curve)
        
        # Fade out at the end
        if end_idx < len(unprocessed) and fade_samples > 0:
            fade_start = max(start_idx, end_idx - fade_samples)
            fade_end = min(len(unprocessed), end_idx + fade_samples)
            
            if fade_start >= start_idx:
                fade_length = fade_end - fade_start
                fade_curve = np.linspace(1, 0, fade_length)
                
                result[fade_start:fade_end] = (processed[fade_start:fade_end] * fade_curve +
                                             unprocessed[fade_start:fade_end] * (1 - fade_curve))
        
        return result


if __name__ == "__main__":
    # Test the DRC engine
    import librosa
    
    # Load test audio
    audio, sr = librosa.load("input.wav", sr=None, mono=True)
    
    # Create and test compressor
    compressor = DynamicRangeCompressor(threshold_db=-20.0, ratio=4.0, 
                                       attack_ms=10.0, release_ms=100.0)
    
    # Process audio
    compressed = compressor.process_audio(audio, sr)
    
    print(f"Original audio RMS: {np.sqrt(np.mean(audio**2)):.6f}")
    print(f"Compressed audio RMS: {np.sqrt(np.mean(compressed**2)):.6f}")
    print(f"Peak reduction: {np.max(np.abs(compressed))/np.max(np.abs(audio)):.3f}")
