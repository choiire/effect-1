"""
Enhanced Directional Microphone Pipeline (VMD + MMSE-LSA)
=========================================================
This script realises the *new* processing chain requested by the user:

1.   VMD – decompose the signal into K Intrinsic Mode Functions (IMFs).
2.   MMSE-LSA (via `pylogmmse`) applied *only* to the low-frequency IMFs to
     remove proximity-effect / HVAC rumble etc.
3.   STFT gating on the remaining IMFs for high-frequency / transient noise.
4.   Re-sum the processed IMFs to create the final enhanced signal.

The implementation intentionally keeps the algorithmic building blocks
self-contained and relies on the following 3rd-party libraries:

• `vmdpy` – for Variational Mode Decomposition.
• `pylogmmse` – for an MMSE-LSA implementation.
• `numpy`, `scipy`, `soundfile` – core DSP/IO stack.

USAGE
-----
python directional_mic_pipeline_vmd.py input.wav output.wav
"""
from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
import soundfile as sf
from scipy.signal.windows import hann
from scipy.fft import rfft, irfft
from scipy.signal import butter, sosfilt, resample_poly

try:
    from vmdpy import VMD  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError("vmdpy is required – `pip install vmdpy`.") from e

try:
    from logmmse import logmmse as _logmmse  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError("`logmmse` package missing – install via `pip install logmmse`.") from e

###############################################################################
# Helper functions
###############################################################################

def design_highpass(sr: int, cutoff: float = 20.0, order: int = 2):
    """A gentle high-pass filter to remove DC-offset and sub-audible rumble."""
    from scipy.signal import butter, sosfilt

    sos = butter(order, cutoff / (0.5 * sr), btype="highpass", output="sos")
    return lambda x: sosfilt(sos, x)


def stft_frame_generator(y: np.ndarray, frame_size: int, hop: int):
    win = hann(frame_size, sym=False)
    idx = 0
    while idx + frame_size <= len(y):
        yield y[idx : idx + frame_size] * win, idx
        idx += hop


def overlap_add(dest: np.ndarray, frame: np.ndarray, start: int):
    dest[start : start + len(frame)] += frame

###############################################################################
# Core processing steps
###############################################################################

def vmd_decompose(y: np.ndarray, sr: int, K: int = 4) -> Tuple[np.ndarray, List[int]]:
    """Run VMD and return (imfs, idx_low_freq_imfs).

    Low-frequency IMFs are detected via their median frequency < 150 Hz.
    """
    alpha = 1800        # slightly narrower bandwidth constraint
    tau = 0             # noise-tolerance (0 for discrete time)
    DC = 0              # do not force first IMF to be DC
    init = 1            # initialise omegas uniformly
    tol = 1e-7

    vmd_out = VMD(y, alpha, tau, K, DC, init, tol)
    # vmdpy can return (u, u_hat, omega) or (u, omega) depending on version
    if len(vmd_out) == 3:
        u, _, omega = vmd_out
    else:
        u, omega = vmd_out

    # Identify low-frequency IMFs (centre-freq < 150 Hz)
    idx_low = []
    for k in range(K):
        omega_k = omega[k]
        # Some versions return single scalar, others array over iterations
        if np.ndim(omega_k) > 0:
            omega_k = omega_k[-1]
        cf_hz = omega_k * sr / (2 * np.pi)
        if cf_hz < 120:
            idx_low.append(k)
    return u, idx_low


def mmse_lsa_lowfreq(imfs: np.ndarray, sr: int, idx_low: List[int]) -> np.ndarray:
    """Apply MMSE-LSA noise reduction to the *sum* of low-frequency IMFs."""
    if not idx_low:
        return np.zeros_like(imfs[0])

    low_sum = np.sum(imfs[idx_low, :], axis=0)
    # pylogmmse expects 16-bit PCM range (-32768 … 32767) in int16 or float32 ±1?
    # We'll scale to int16 range, run logmmse, then scale back.
    peak = np.max(np.abs(low_sum)) + 1e-9
    scaled = (low_sum / peak * 32767).astype(np.int16)
    denoised = _logmmse(scaled, sr).astype(np.float32) / 32767 * peak
    return denoised


def stft_gating(y: np.ndarray, sr: int) -> np.ndarray:
    """Simple STFT-domain gating / spectral subtraction for higher-frequency IMFs."""
    FRAME = 256
    HOP = FRAME // 2
    FFT = FRAME
    win = hann(FRAME, sym=False)
    win_norm = np.sum(win ** 2)

    # noise estimation from first 1 s
    init_noise_frames = int((sr - FRAME) / HOP)
    noise_psd = np.zeros(FFT // 2 + 1)
    for i, (frame, _) in zip(range(init_noise_frames), stft_frame_generator(y[: sr], FRAME, HOP)):
        spec = rfft(frame, n=FFT)
        noise_psd += np.abs(spec) ** 2
    noise_psd /= max(1, init_noise_frames)

    out = np.zeros_like(y)
    for frame, start in stft_frame_generator(y, FRAME, HOP):
        spec = rfft(frame, n=FFT)
        mag2 = np.abs(spec) ** 2
        snr = np.maximum(mag2 - noise_psd, 0.0) / (noise_psd + 1e-12)
        gain = snr / (1 + snr)
        gain = np.maximum(gain, 10 ** (-12 / 20))  # ‑12 dB floor
        extra_atten = 10 ** (-2 / 20)   # -2 dB floor

        # gating for residual bins below 1.2× noise
        gate_mask = mag2 * gain ** 2 < 0.6 * noise_psd
        gain[gate_mask] *= extra_atten

        frame_rec = irfft(spec * gain, n=FFT) * win
        overlap_add(out, frame_rec, start)

    out /= win_norm / HOP
    return out[: len(y)]

###############################################################################
# Top-level pipeline
###############################################################################

def _prepare_vmd_signal(y: np.ndarray, sr: int, target_len: int = 80_000) -> tuple[np.ndarray, int]:
    """Down-sample *y* to keep length manageable for VMD.

    Returns (y_ds, ds_factor). If no down-sampling applied, ds_factor == 1.
    """
    if len(y) <= target_len:
        return y, 1
    # choose integer down-sample factor
    ds_factor = int(np.ceil(len(y) / target_len))
    y_ds = resample_poly(y, up=1, down=ds_factor)
    return y_ds.astype(np.float32), ds_factor


def process_pipeline(y: np.ndarray, sr: int) -> np.ndarray:
    # Pre-HPF to remove DC / very low rumble (<20 Hz)
    y = design_highpass(sr)(y)

    # 0. Prepare reduced-length signal for VMD to avoid huge memory
    y_vmd, ds = _prepare_vmd_signal(y, sr)

    # 1. VMD decomposition on the (possibly) down-sampled signal
    K = 4
    imfs, idx_low = vmd_decompose(y_vmd, sr // ds if ds > 1 else sr, K=K)

    # 2. Low-frequency noise suppression on selected IMFs
    low_clean = mmse_lsa_lowfreq(imfs, sr, idx_low)

    # 3. Process remaining IMFs through STFT gating
    other_idx = [k for k in range(K) if k not in idx_low]
    other_sum = np.sum(imfs[other_idx, :], axis=0) if other_idx else np.zeros_like(low_clean)
    other_clean = stft_gating(other_sum, sr)

    # 4. Re-sum
    y_out_ds = low_clean + other_clean

    # 5. Upsample back to original length if we down-sampled
    if ds > 1:
        y_out = resample_poly(y_out_ds, up=ds, down=1)
        y_out = y_out[: len(y)]  # trim to exact original length
    else:
        y_out = y_out_ds

    # Normalise to prevent clipping
    peak = np.max(np.abs(y_out)) + 1e-9
    if peak > 1.0:
        y_out /= peak
    return y_out

###############################################################################
# CLI
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Enhanced directional-mic cleanup (VMD + MMSE-LSA)")
    parser.add_argument("input_file", type=str, help="Input WAV path (mono 48 kHz)")
    parser.add_argument("output_file", type=str, help="Output WAV path")
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        parser.error(f"Input file '{args.input_file}' not found.")

    y, sr = sf.read(args.input_file)
    if y.ndim > 1:
        print("[Info] Multi-channel detected – using first channel.")
        y = y[:, 0]
    if sr != 48000:
        print(f"[Warn] Expected 48 kHz, but got {sr}. Continuing anyway…")

    y_proc = process_pipeline(y.astype(np.float32), sr)
    sf.write(args.output_file, y_proc, sr)
    print(f"[Done] Saved to {args.output_file}")


if __name__ == "__main__":
    main()
