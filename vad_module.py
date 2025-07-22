"""
Voice Activity Detection (VAD) Module
Implements robust speech detection using WebRTC VAD with preprocessing and segmentation.
"""

import numpy as np
import librosa
import webrtcvad
from typing import List, Tuple, Generator


class RobustVAD:
    """
    Robust Voice Activity Detection using WebRTC VAD with preprocessing and segmentation.
    """
    
    def __init__(self, aggressiveness: int = 2, frame_duration_ms: int = 20, hangover_ms: int = 200):
        """
        Initialize VAD with specified parameters.
        
        Args:
            aggressiveness: VAD aggressiveness mode (0-3, higher = more aggressive)
            frame_duration_ms: Frame duration in milliseconds (10, 20, or 30)
            hangover_ms: Hangover time in milliseconds to prevent cutting off speech
        """
        self.aggressiveness = aggressiveness
        self.frame_duration_ms = frame_duration_ms
        self.hangover_ms = hangover_ms
        self.vad = webrtcvad.Vad(aggressiveness)
        
        # Supported sample rates by WebRTC VAD
        self.supported_sample_rates = [8000, 16000, 32000, 48000]
        self.target_sample_rate = 16000  # Use 16kHz as default
        
    def preprocess_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load and preprocess audio for WebRTC VAD.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (preprocessed_audio_int16, sample_rate)
        """
        # Load audio with librosa
        audio, original_sr = librosa.load(audio_path, sr=None, mono=True)
        
        # Resample to target sample rate if needed
        if original_sr != self.target_sample_rate:
            audio = librosa.resample(audio, orig_sr=original_sr, target_sr=self.target_sample_rate)
        
        # Convert float audio to int16 format required by WebRTC VAD
        # Ensure audio is in [-1, 1] range first
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)
        
        return audio_int16, self.target_sample_rate
    
    def frame_generator(self, audio: np.ndarray, sample_rate: int) -> Generator[bytes, None, None]:
        """
        Generate frames of specified duration for VAD processing.
        
        Args:
            audio: Audio data as int16 numpy array
            sample_rate: Sample rate of audio
            
        Yields:
            Audio frames as bytes
        """
        frame_length = int(sample_rate * self.frame_duration_ms / 1000)
        
        for i in range(0, len(audio), frame_length):
            frame = audio[i:i + frame_length]
            
            # Pad the last frame if it's shorter than required
            if len(frame) < frame_length:
                frame = np.pad(frame, (0, frame_length - len(frame)), mode='constant')
            
            yield frame.tobytes()
    
    def apply_vad_with_hangover(self, audio: np.ndarray, sample_rate: int) -> List[bool]:
        """
        Apply VAD to audio frames with hangover logic.
        
        Args:
            audio: Preprocessed audio as int16 numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            List of boolean values indicating speech activity for each frame
        """
        vad_results = []
        hangover_frames = int(self.hangover_ms / self.frame_duration_ms)
        hangover_counter = 0
        
        for frame in self.frame_generator(audio, sample_rate):
            is_speech = self.vad.is_speech(frame, sample_rate)
            
            if is_speech:
                vad_results.append(True)
                hangover_counter = hangover_frames  # Reset hangover counter
            else:
                if hangover_counter > 0:
                    vad_results.append(True)  # Continue speech during hangover
                    hangover_counter -= 1
                else:
                    vad_results.append(False)
        
        return vad_results
    
    def segment_speech(self, vad_results: List[bool], sample_rate: int) -> List[Tuple[float, float]]:
        """
        Convert frame-level VAD results to speech segments with timestamps.
        
        Args:
            vad_results: List of boolean VAD results for each frame
            sample_rate: Sample rate of audio
            
        Returns:
            List of (start_time, end_time) tuples in seconds
        """
        segments = []
        frame_duration_sec = self.frame_duration_ms / 1000.0
        
        in_speech = False
        start_time = 0.0
        
        for i, is_speech in enumerate(vad_results):
            current_time = i * frame_duration_sec
            
            if is_speech and not in_speech:
                # Start of speech segment
                start_time = current_time
                in_speech = True
            elif not is_speech and in_speech:
                # End of speech segment
                end_time = current_time
                segments.append((start_time, end_time))
                in_speech = False
        
        # Handle case where speech continues until the end
        if in_speech:
            end_time = len(vad_results) * frame_duration_sec
            segments.append((start_time, end_time))
        
        return segments
    
    def detect_speech_segments(self, audio_path: str) -> List[Tuple[float, float]]:
        """
        Main method to detect speech segments in an audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of (start_time, end_time) tuples in seconds for speech segments
        """
        # Preprocess audio
        audio_int16, sample_rate = self.preprocess_audio(audio_path)
        
        # Apply VAD with hangover
        vad_results = self.apply_vad_with_hangover(audio_int16, sample_rate)
        
        # Convert to speech segments
        segments = self.segment_speech(vad_results, sample_rate)
        
        return segments


if __name__ == "__main__":
    # Test the VAD module
    vad = RobustVAD(aggressiveness=2)
    segments = vad.detect_speech_segments("input.wav")
    
    print(f"Detected {len(segments)} speech segments:")
    for i, (start, end) in enumerate(segments):
        print(f"Segment {i+1}: {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")
