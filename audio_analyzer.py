"""
Audio Analysis Tool
Detailed analysis of input and output audio files to identify differences and issues.
"""

import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal
import os


class AudioAnalyzer:
    """
    Comprehensive audio analysis tool for comparing input and output files.
    """
    
    def __init__(self):
        pass
    
    def load_audio(self, file_path):
        """Load audio file and return audio data and sample rate."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        audio, sr = librosa.load(file_path, sr=None, mono=True)
        return audio, sr
    
    def analyze_audio_properties(self, audio, sr, label="Audio"):
        """Analyze basic audio properties."""
        duration = len(audio) / sr
        rms = np.sqrt(np.mean(audio**2))
        peak = np.max(np.abs(audio))
        
        # Calculate dynamic range
        db_audio = 20 * np.log10(np.abs(audio) + 1e-10)
        dynamic_range = np.max(db_audio) - np.min(db_audio)
        
        # Calculate loudness statistics
        loudness_db = 20 * np.log10(rms + 1e-10)
        peak_db = 20 * np.log10(peak + 1e-10)
        
        # Calculate spectral centroid (brightness)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        avg_spectral_centroid = np.mean(spectral_centroid)
        
        # Calculate zero crossing rate (speech activity indicator)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        avg_zcr = np.mean(zcr)
        
        return {
            'label': label,
            'duration': duration,
            'rms': rms,
            'peak': peak,
            'loudness_db': loudness_db,
            'peak_db': peak_db,
            'dynamic_range_db': dynamic_range,
            'spectral_centroid': avg_spectral_centroid,
            'zero_crossing_rate': avg_zcr,
            'sample_rate': sr,
            'samples': len(audio)
        }
    
    def analyze_volume_distribution(self, audio, label="Audio"):
        """Analyze volume distribution and dynamics."""
        # Convert to dB
        audio_db = 20 * np.log10(np.abs(audio) + 1e-10)
        
        # Calculate percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        volume_percentiles = np.percentile(audio_db, percentiles)
        
        # Calculate RMS in sliding windows
        window_size = int(0.1 * len(audio))  # 100ms windows
        rms_windows = []
        
        for i in range(0, len(audio) - window_size, window_size // 2):
            window = audio[i:i + window_size]
            rms = np.sqrt(np.mean(window**2))
            rms_windows.append(rms)
        
        rms_windows = np.array(rms_windows)
        rms_db_windows = 20 * np.log10(rms_windows + 1e-10)
        
        return {
            'label': label,
            'percentiles': dict(zip(percentiles, volume_percentiles)),
            'rms_windows': rms_windows,
            'rms_db_windows': rms_db_windows,
            'rms_std': np.std(rms_windows),
            'rms_range': np.max(rms_windows) - np.min(rms_windows)
        }
    
    def compare_files(self, input_file, output_file):
        """Compare input and output audio files."""
        print("=== Audio File Comparison ===\n")
        
        # Load both files
        try:
            input_audio, input_sr = self.load_audio(input_file)
            output_audio, output_sr = self.load_audio(output_file)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
        
        # Basic properties analysis
        input_props = self.analyze_audio_properties(input_audio, input_sr, "Input")
        output_props = self.analyze_audio_properties(output_audio, output_sr, "Output")
        
        print("1. BASIC PROPERTIES COMPARISON")
        print("-" * 50)
        properties = ['duration', 'rms', 'peak', 'loudness_db', 'peak_db', 'dynamic_range_db', 
                     'spectral_centroid', 'zero_crossing_rate']
        
        for prop in properties:
            input_val = input_props[prop]
            output_val = output_props[prop]
            change = ((output_val - input_val) / input_val * 100) if input_val != 0 else 0
            
            print(f"{prop:20s}: {input_val:10.6f} -> {output_val:10.6f} ({change:+6.2f}%)")
        
        # Volume distribution analysis
        input_dist = self.analyze_volume_distribution(input_audio, "Input")
        output_dist = self.analyze_volume_distribution(output_audio, "Output")
        
        print("\n2. VOLUME DISTRIBUTION COMPARISON")
        print("-" * 50)
        print("Percentiles (dB):")
        for percentile in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            input_val = input_dist['percentiles'][percentile]
            output_val = output_dist['percentiles'][percentile]
            change = output_val - input_val
            print(f"  {percentile:2d}th percentile: {input_val:8.2f} -> {output_val:8.2f} ({change:+6.2f} dB)")
        
        print(f"\nRMS variability:")
        print(f"  Input RMS std:  {input_dist['rms_std']:.6f}")
        print(f"  Output RMS std: {output_dist['rms_std']:.6f}")
        print(f"  Change:         {output_dist['rms_std'] - input_dist['rms_std']:+.6f}")
        
        print(f"\nRMS range:")
        print(f"  Input RMS range:  {input_dist['rms_range']:.6f}")
        print(f"  Output RMS range: {output_dist['rms_range']:.6f}")
        print(f"  Change:           {output_dist['rms_range'] - input_dist['rms_range']:+.6f}")
        
        # Difference analysis
        print("\n3. DIRECT DIFFERENCE ANALYSIS")
        print("-" * 50)
        
        # Ensure same length for comparison
        min_length = min(len(input_audio), len(output_audio))
        input_trimmed = input_audio[:min_length]
        output_trimmed = output_audio[:min_length]
        
        # Calculate difference
        diff = output_trimmed - input_trimmed
        diff_rms = np.sqrt(np.mean(diff**2))
        diff_max = np.max(np.abs(diff))
        
        # Calculate correlation
        correlation = np.corrcoef(input_trimmed, output_trimmed)[0, 1]
        
        print(f"Difference RMS:     {diff_rms:.8f}")
        print(f"Max difference:     {diff_max:.8f}")
        print(f"Correlation:        {correlation:.8f}")
        print(f"Identical samples:  {np.sum(diff == 0)} / {len(diff)} ({np.sum(diff == 0)/len(diff)*100:.2f}%)")
        
        # Check if files are essentially identical
        if diff_rms < 1e-6:
            print("\n⚠️  WARNING: Files are essentially identical!")
            print("   The processing may not be working as expected.")
        elif diff_rms < 1e-4:
            print("\n⚠️  WARNING: Very small differences detected!")
            print("   The processing effect may be too subtle to notice.")
        else:
            print(f"\n✓ Significant differences detected (RMS: {diff_rms:.6f})")
        
        return {
            'input_props': input_props,
            'output_props': output_props,
            'input_dist': input_dist,
            'output_dist': output_dist,
            'diff_rms': diff_rms,
            'correlation': correlation
        }
    
    def diagnose_processing_issues(self, input_file, output_file):
        """Diagnose potential issues with the processing pipeline."""
        print("\n4. PROCESSING PIPELINE DIAGNOSIS")
        print("-" * 50)
        
        # Load files
        input_audio, input_sr = self.load_audio(input_file)
        output_audio, output_sr = self.load_audio(output_file)
        
        # Check if compression threshold was appropriate
        input_db = 20 * np.log10(np.abs(input_audio) + 1e-10)
        threshold_db = -18.0  # From the processing command
        
        samples_above_threshold = np.sum(input_db > threshold_db)
        total_samples = len(input_audio)
        threshold_ratio = samples_above_threshold / total_samples
        
        print(f"Compression threshold analysis (-18 dB):")
        print(f"  Samples above threshold: {samples_above_threshold:,} / {total_samples:,} ({threshold_ratio:.2%})")
        
        if threshold_ratio < 0.01:
            print("  ⚠️  Very few samples above threshold - compression may not be effective")
        elif threshold_ratio < 0.1:
            print("  ⚠️  Low percentage above threshold - consider lowering threshold")
        else:
            print("  ✓ Reasonable amount of signal above threshold")
        
        # Check dynamic range
        input_dynamic_range = np.max(input_db) - np.min(input_db)
        output_dynamic_range = np.max(20 * np.log10(np.abs(output_audio) + 1e-10)) - np.min(20 * np.log10(np.abs(output_audio) + 1e-10))
        
        print(f"\nDynamic range analysis:")
        print(f"  Input dynamic range:  {input_dynamic_range:.2f} dB")
        print(f"  Output dynamic range: {output_dynamic_range:.2f} dB")
        print(f"  Reduction:            {input_dynamic_range - output_dynamic_range:.2f} dB")
        
        if abs(input_dynamic_range - output_dynamic_range) < 1.0:
            print("  ⚠️  Very little dynamic range compression occurred")
        
        # Suggest improvements
        print(f"\n5. IMPROVEMENT SUGGESTIONS")
        print("-" * 50)
        
        if threshold_ratio < 0.1:
            suggested_threshold = np.percentile(input_db, 75)  # 75th percentile
            print(f"• Consider lowering compression threshold to {suggested_threshold:.1f} dB")
        
        if abs(input_dynamic_range - output_dynamic_range) < 2.0:
            print("• Consider increasing compression ratio (try 6:1 or 8:1)")
            print("• Consider using a lower threshold")
        
        input_rms_db = 20 * np.log10(np.sqrt(np.mean(input_audio**2)) + 1e-10)
        if input_rms_db > -20:
            print("• Input audio is already quite loud - compression effect may be subtle")
        
        print("• Try more aggressive VAD settings (--vad-mode 3)")
        print("• Try shorter attack/release times for more noticeable effect")


if __name__ == "__main__":
    analyzer = AudioAnalyzer()
    
    # Compare input and output files
    results = analyzer.compare_files("input.wav", "output_normalized.wav")
    
    # Diagnose issues
    analyzer.diagnose_processing_issues("input.wav", "output_normalized.wav")
