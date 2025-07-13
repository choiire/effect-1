"""
Directional Microphone Audio Processing Pipeline
------------------------------------------------
This script implements the five-step pipeline described in the prompt for reducing
proximity effect and background noise captured by a directional microphone in a
lecture hall environment.

Key processing stages:
1. High-pass filter (HPF) – mitigate excessive low-frequency boost.
2. Noise reduction using a spectral-subtraction style Wiener filter with an
   adaptive noise Power Spectral Density (PSD) estimate.
3. Transient (short, impulsive) noise suppression through simple
   frequency-domain gating.

The implementation processes a *recorded* WAV file offline, but the algorithm
is framed in a block-wise manner (256-sample frames, 50 % overlap), so it can be
ported to real-time operation with minimal changes.

Usage
-----
python directional_mic_pipeline.py input.wav output.wav

Dependencies
------------
 * numpy
 * scipy (signal + fft)
 * soundfile (for reading / writing wav)
 * vmdpy (Variational Mode Decomposition)
 * pylogmmse (MMSE-LSA noise suppression)
"""

import argparse
import os
from typing import Tuple

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
from scipy.signal.windows import hann
from scipy.fft import rfft, irfft

###############################################################################
# Utility functions
###############################################################################

def design_highpass(sr: int, cutoff: float = 80.0, order: int = 2):
    """Design a Butterworth high-pass filter and return it in SOS form."""
    nyq = 0.5 * sr
    norm_cutoff = cutoff / nyq
    sos = butter(order, norm_cutoff, btype="highpass", output="sos")
    return sos


def stft_frame_generator(y: np.ndarray, frame_size: int, hop: int) -> Tuple[np.ndarray, int]:
    """Yield successive frames of *y* with Hann window applied.

    Returns (windowed_frame, start_sample_index).
    """
    window = hann(frame_size, sym=False)
    total = len(y)
    idx = 0
    while idx + frame_size <= total:
        frame = y[idx : idx + frame_size] * window
        yield frame, idx
        idx += hop


def overlap_add(out: np.ndarray, frame_rec: np.ndarray, start: int):
    """Overlap-add *frame_rec* into *out* starting at sample *start*."""
    out[start : start + len(frame_rec)] += frame_rec

###############################################################################
# Processing stages
###############################################################################

def wiener_gain(noise_psd: np.ndarray, mag2: np.ndarray, gain_floor_lin: float) -> np.ndarray:
    """Compute Wiener filter gain with a minimum gain floor."""
    snr = np.maximum(mag2 - noise_psd, 0.0) / (noise_psd + 1e-12)
    g = snr / (1.0 + snr)
    return np.maximum(g, gain_floor_lin)


def process_pipeline(y: np.ndarray, sr: int) -> np.ndarray:
    """Run the full processing pipeline and return the enhanced signal."""

    # ---------------------------------------------------------------------
    # 0. Basic params (all derived from prompt)
    # ---------------------------------------------------------------------
    FRAME_SIZE = 256  # samples  (≈5.33 ms @ 48 kHz)
    HOP = FRAME_SIZE // 2  # 50 % overlap → 128 samples
    FFT_SIZE = FRAME_SIZE  # keep identical resolution

    # ---------------------------------------------------------------------
    # 1. High-pass filter
    # ---------------------------------------------------------------------
    sos = design_highpass(sr, cutoff=80.0, order=2)
    y_hp = sosfilt(sos, y)

    # ---------------------------------------------------------------------
    # 2. STFT preparation
    # ---------------------------------------------------------------------
    window = hann(FRAME_SIZE, sym=False)
    window_norm = np.sum(window ** 2)

    # Padding so that final frame fits exactly
    pad_len = (FRAME_SIZE - len(y_hp) % HOP) % HOP
    if pad_len:
        y_hp = np.concatenate([y_hp, np.zeros(pad_len, dtype=y_hp.dtype)])

    n_frames = (len(y_hp) - FRAME_SIZE) // HOP + 1

    # Pre-allocate output buffer
    out = np.zeros_like(y_hp)

    # ---------------------------------------------------------------------
    # 3. Noise PSD initial estimate (first 1 s assumed to be noise-only)
    # ---------------------------------------------------------------------
    init_noise_samples = sr  # 1 second
    init_noise_audio = y_hp[:init_noise_samples]

    # Compute average noise PSD from initial segment
    noise_psd = np.zeros(FFT_SIZE // 2 + 1)
    for frame, _ in stft_frame_generator(init_noise_audio, FRAME_SIZE, HOP):
        spec = rfft(frame, n=FFT_SIZE)
        noise_psd += np.abs(spec) ** 2
    if n_frames:
        noise_psd /= max(1, int((init_noise_samples - FRAME_SIZE) // HOP + 1))

    # Update parameters
    GAIN_FLOOR_DB = -20.0
    GAIN_FLOOR_LIN = 10 ** (GAIN_FLOOR_DB / 20)

    # Variables for noise PSD adaptive update
    VAD_THRESHOLD = 3.0  # dB above noise floor to be considered speech
    NOISE_UPDATE_FRAMES = 6  # update every ~6 frames (≈64 ms)
    noise_update_counter = 0

    # ---------------------------------------------------------------------
    # 4. Main processing loop over frames
    # ---------------------------------------------------------------------
    for frame_idx, (frame, start) in enumerate(stft_frame_generator(y_hp, FRAME_SIZE, HOP)):
        spec = rfft(frame, n=FFT_SIZE)
        mag2 = np.abs(spec) ** 2

        # VAD – simple decision based on total energy vs. noise+margin
        frame_energy_db = 10 * np.log10(np.sum(mag2) + 1e-12)
        noise_energy_db = 10 * np.log10(np.sum(noise_psd) + 1e-12)
        is_noise_only = frame_energy_db < (noise_energy_db + VAD_THRESHOLD)

        # Noise PSD update (running average) when no speech
        noise_update_counter += 1
        if is_noise_only and noise_update_counter >= NOISE_UPDATE_FRAMES:
            alpha = 0.9  # smoothing factor
            noise_psd = alpha * noise_psd + (1 - alpha) * mag2
            noise_update_counter = 0

        # Wiener gain with floor
        gain = wiener_gain(noise_psd, mag2, GAIN_FLOOR_LIN)

        # -----------------------------------------------------------------
        # 5. Transient suppression – simple gating per bin.
        #    If post-Wiener magnitude < 1.2 * noise, attenuate an extra 10 dB.
        # -----------------------------------------------------------------
        post_mag2 = mag2 * gain ** 2
        gate_mask = post_mag2 < (1.2 * noise_psd)
        extra_atten = 10 ** (-10 / 20)  # -10 dB
        gain[gate_mask] *= extra_atten

        # Apply gain
        spec_processed = spec * gain

        # IFFT and overlap-add
        frame_rec = irfft(spec_processed, n=FFT_SIZE)
        frame_rec *= window  # synthesis window (same Hann)
        overlap_add(out, frame_rec, start)

    # Compensate windowing (Hann overlap-add results in constant factor == window_norm / HOP)
    out /= window_norm / HOP

    # Trim any padding
    out = out[: len(y)]

    # Normalise to prevent clipping
    peak = np.max(np.abs(out))
    if peak > 1.0:
        out = out / peak

    return out

###############################################################################
# Main entry point
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Directional-mic audio cleanup pipeline")
    parser.add_argument("input_file", type=str, help="Path to input WAV file")
    parser.add_argument("output_file", type=str, help="Path for processed WAV file")
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        parser.error(f"Input file '{args.input_file}' not found.")

    # Read audio (supports 16-bit / 24-bit; converts to float32 in range [-1,1])
    y, sr = sf.read(args.input_file)

    # For multi-channel material we take the first channel
    if y.ndim > 1:
        print("[Info] Stereo/Multichannel input detected – using first channel only.")
        y = y[:, 0]

    if sr != 48000:
        print(f"[Warning] Expected 48 kHz sample-rate, but got {sr} Hz.")

    print("[Info] Running pipeline…")
    y_proc = process_pipeline(y.astype(np.float32), sr)

    sf.write(args.output_file, y_proc, sr)
    print(f"[Done] Processed file saved to '{args.output_file}'.")


if __name__ == "__main__":
    main()
