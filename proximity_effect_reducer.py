import soundfile as sf
import numpy as np
from scipy.signal import stft, istft, butter, lfilter, sosfilt
import argparse
import os

class DynamicEQ:
    """
    A class to apply dynamic equalization to reduce the proximity effect in audio files.
    """

    def __init__(self, sample_rate, frame_size=2048, hop_size=512):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.fft_size = frame_size

    def _calculate_coeffs(self, time_ms):
        """Calculate attack/release coefficients from time in milliseconds."""
        if time_ms <= 0:
            return 1.0
        return np.exp(-1 / (self.sample_rate * (time_ms / 1000.0) / self.hop_size))

    def _design_low_shelf(self, cutoff_freq, gain_db):
        """Designs a Biquad low-shelf filter and returns SOS coefficients."""
        gain_linear = 10 ** (gain_db / 20.0)
        nyquist = 0.5 * self.sample_rate
        norm_cutoff = cutoff_freq / nyquist
        
        # Using a simple Butterworth filter design as a base for the shelf
        # Note: A more direct low-shelf design could be implemented for more precision.
        # For simplicity, we'll use a standard butterworth and apply gain.
        # This is a simplification; a true shelving filter has a different structure.
        # However, for this application, we can simulate the effect by blending.
        b, a = butter(2, norm_cutoff, btype='low', analog=False)
        return b, a

    def process(self, audio_data, **params):
        """
        Processes the audio data to apply dynamic EQ.

        :param audio_data: Numpy array of the audio signal.
        :param params: Dictionary of processing parameters.
        :return: Processed audio data.
        """
        # Extract parameters with defaults
        low_freq_band = params.get('low_freq_band', (100, 300))
        threshold_db = params.get('threshold_db', -20.0)
        ratio = params.get('ratio', 4.0)
        attack_ms = params.get('attack_ms', 5.0)
        release_ms = params.get('release_ms', 100.0)
        filter_cutoff = params.get('filter_cutoff', 250.0)
        max_gain_reduction_db = params.get('max_gain_reduction_db', -12.0)

        # 1. STFT Analysis
        f, t, Zxx = stft(audio_data, fs=self.sample_rate, nperseg=self.frame_size, noverlap=self.frame_size - self.hop_size)

        # Find frequency bins for the target band
        freq_band_indices = np.where((f >= low_freq_band[0]) & (f <= low_freq_band[1]))[0]

        # 2. Calculate energy in the low-frequency band for each frame
        low_band_energy = np.mean(np.abs(Zxx[freq_band_indices, :]), axis=0)
        low_band_energy_db = 20 * np.log10(low_band_energy + 1e-9) # Add epsilon to avoid log(0)

        # 3. Dynamic Control Logic
        attack_coeff = self._calculate_coeffs(attack_ms)
        release_coeff = self._calculate_coeffs(release_ms)
        
        gain_reduction_db = 0.0
        gain_envelope = []

        for energy_db in low_band_energy_db:
            if energy_db > threshold_db:
                target_gain_reduction = (threshold_db - energy_db) * (1 - 1 / ratio)
                gain_reduction_db = (1 - attack_coeff) * target_gain_reduction + attack_coeff * gain_reduction_db
            else:
                target_gain_reduction = 0.0
                gain_reduction_db = (1 - release_coeff) * target_gain_reduction + release_coeff * gain_reduction_db
            
            # Clamp the gain reduction to the maximum allowed
            gain_reduction_db = max(gain_reduction_db, max_gain_reduction_db)
            gain_envelope.append(gain_reduction_db)

        gain_envelope = np.array(gain_envelope)

        # Upsample gain envelope to match audio length
        gain_envelope_upsampled = np.repeat(gain_envelope, self.hop_size)
        # Ensure the upsampled envelope has the same length as the audio data
        if len(gain_envelope_upsampled) > len(audio_data):
            gain_envelope_upsampled = gain_envelope_upsampled[:len(audio_data)]
        else:
            padding = np.full(len(audio_data) - len(gain_envelope_upsampled), gain_envelope_upsampled[-1])
            gain_envelope_upsampled = np.concatenate([gain_envelope_upsampled, padding])

        # 4. Biquad Filter Application
        processed_audio = np.zeros_like(audio_data)
        
        # Process audio in chunks to apply time-varying filter
        # A simpler approach for non-realtime is to filter a low-passed version and blend it
        b, a = self._design_low_shelf(filter_cutoff, -1) # Design a prototype low-pass filter
        low_freq_component = lfilter(b, a, audio_data)

        gain_linear = 10 ** (gain_envelope_upsampled / 20.0)
        
        # The final signal is the original signal minus the scaled low-frequency component,
        # where the scaling depends on the gain reduction.
        # (1.0 - gain_linear) is the amount of low-frequency component to remove.
        processed_audio = audio_data - low_freq_component * (1.0 - gain_linear)

        return processed_audio

def main():
    parser = argparse.ArgumentParser(description="Reduce proximity effect in an audio file.")
    parser.add_argument("input_file", type=str, help="Path to the input WAV file.")
    parser.add_argument("output_file", type=str, help="Path to save the output WAV file.")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found at {args.input_file}")
        return

    # Load audio file
    audio, sr = sf.read(args.input_file)
    print(f"Loaded '{args.input_file}' - Sample Rate: {sr}, Duration: {len(audio)/sr:.2f}s")

    # If stereo, process the first channel (or average them)
    if audio.ndim > 1:
        print("Stereo file detected, processing the left channel.")
        audio_mono = audio[:, 0]
    else:
        audio_mono = audio

    # Initialize the dynamic EQ processor
    dynamic_eq = DynamicEQ(sample_rate=sr)

    # Define processing parameters
    params = {
        'low_freq_band': (80, 250),      # Frequency range to monitor (Hz)
        'threshold_db': -18.0,          # DBFS level to start gain reduction
        'ratio': 3.0,                   # Compression ratio (e.g., 3:1)
        'attack_ms': 10.0,              # Time to react to loud sounds (ms)
        'release_ms': 150.0,            # Time to recover from gain reduction (ms)
        'filter_cutoff': 250.0,         # Cutoff frequency for the low-shelf filter
        'max_gain_reduction_db': -12.0  # Maximum amount of bass cut (dB)
    }

    # Process the audio
    print("Processing audio...")
    processed_audio = dynamic_eq.process(audio_mono, **params)

    # Normalize the output to prevent clipping
    max_val = np.max(np.abs(processed_audio))
    if max_val > 1.0:
        processed_audio /= max_val

    # Save the processed audio
    sf.write(args.output_file, processed_audio, sr)
    print(f"Processed audio saved to '{args.output_file}'")

if __name__ == "__main__":
    main()

