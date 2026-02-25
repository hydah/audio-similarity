#!/usr/bin/env python3
"""Audio Quality Comparison Tool

Compares AAC audio files at different bitrates using spectral analysis
and objective quality metrics. Generates a visual report (PNG) and
console summary to demonstrate which version is closer to the original.

Usage:
    python3 audio_quality_compare.py <original> <high_bitrate> <low_bitrate> [-o output.png]
"""

import argparse
import json
import os
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from numpy.linalg import norm
from scipy.signal import stft as scipy_stft

# ─── Constants ───────────────────────────────────────────────────────────────

SR = 48000          # Target sample rate (Hz)
N_FFT = 4096        # FFT window size
HOP = 1024          # Hop length (75% overlap)
MFCC_N = 20         # Number of MFCC coefficients
MEL_N_BANDS = 128   # Mel filter bank bands

# Colors
C_ORIG = "#2196F3"  # Blue
C_HIGH = "#4CAF50"  # Green
C_LOW = "#F44336"   # Red


# ─── Audio Loading ───────────────────────────────────────────────────────────

def load_audio(path: str, sr: int = SR, channels: int = 1) -> np.ndarray:
    """Decode audio file to float32 PCM via ffmpeg.

    Returns:
      - channels=1: shape (samples,)
      - channels=2: shape (2, samples), channel-first
    """
    cmd = [
        "ffmpeg", "-i", path,
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ar", str(sr),
        "-ac", str(channels),
        "-v", "quiet",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {path}: {result.stderr.decode()}")
    y = np.frombuffer(result.stdout, dtype=np.float32)
    if channels == 1:
        return y

    if len(y) == 0:
        return np.zeros((channels, 0), dtype=np.float32)

    rem = len(y) % channels
    if rem:
        y = y[:-rem]
    return y.reshape(-1, channels).T


def get_file_info(path: str) -> dict:
    """Get audio file metadata via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed for {path} (exit {result.returncode}): "
            f"{result.stderr.strip()}"
        )
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"ffprobe returned invalid JSON for {path}: {exc}"
        ) from exc
    fmt = data.get("format", {})
    return {
        "path": os.path.basename(path),
        "bitrate_kbps": int(fmt.get("bit_rate", 0)) // 1000,
        "duration": float(fmt.get("duration", 0)),
        "size_mb": int(fmt.get("size", 0)) / (1024 * 1024),
    }


def trim_to_shortest(*signals: np.ndarray) -> list[np.ndarray]:
    """Trim all 1D/2D signals to the shortest sample length (last axis)."""
    min_len = min(s.shape[-1] if s.ndim > 1 else len(s) for s in signals)
    out = []
    for s in signals:
        out.append(s[..., :min_len] if s.ndim > 1 else s[:min_len])
    return out


def align_signals(*signals: np.ndarray) -> list[np.ndarray]:
    """Backward-compatible alias for 1D signal alignment."""
    return trim_to_shortest(*signals)


def estimate_shift(reference: np.ndarray, test: np.ndarray, max_shift: int = 48000) -> int:
    """Estimate sample shift to align test to reference via cross-correlation."""
    # Use a chunk from the beginning for speed
    chunk = min(max_shift * 4, len(reference), len(test))
    ref_chunk = reference[:chunk]
    test_chunk = test[:chunk]

    corr = np.correlate(ref_chunk, test_chunk[:max_shift * 2], mode="full")
    best = np.argmax(corr)
    return int(best - len(ref_chunk) + 1)


def apply_shift(reference: np.ndarray, test: np.ndarray, shift: int) -> tuple[np.ndarray, np.ndarray]:
    """Apply sample shift and trim to equal length for 1D/2D signals."""
    if shift > 0:
        test = test[..., shift:] if test.ndim > 1 else test[shift:]
    elif shift < 0:
        cut = -shift
        reference = reference[..., cut:] if reference.ndim > 1 else reference[cut:]

    min_len = min(reference.shape[-1] if reference.ndim > 1 else len(reference),
                  test.shape[-1] if test.ndim > 1 else len(test))
    reference = reference[..., :min_len] if reference.ndim > 1 else reference[:min_len]
    test = test[..., :min_len] if test.ndim > 1 else test[:min_len]
    return reference, test


def build_common_aligned_triplet(reference: np.ndarray,
                                 high: np.ndarray,
                                 low: np.ndarray,
                                 shift_high: int,
                                 shift_low: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a common aligned (reference, high, low) window from pairwise shifts."""
    ref_start = max(0, -shift_high, -shift_low)
    high_start = max(0, shift_high)
    low_start = max(0, shift_low)
    r = reference[ref_start:]
    h = high[high_start:]
    l = low[low_start:]
    r, h, l = trim_to_shortest(r, h, l)
    return r, h, l


def cross_correlate_align(reference: np.ndarray, test: np.ndarray,
                          max_shift: int = 48000) -> tuple[np.ndarray, np.ndarray]:
    """Align test signal to reference using cross-correlation (up to max_shift samples).

    This compensates for encoder delay / padding differences between AAC files.
    """
    shift = estimate_shift(reference, test, max_shift=max_shift)
    return apply_shift(reference, test, shift)


# ─── Spectral Utilities ─────────────────────────────────────────────────────

def compute_stft(y: np.ndarray) -> np.ndarray:
    """Compute magnitude STFT in dB."""
    _, _, Zxx = scipy_stft(y, fs=SR, nperseg=N_FFT, noverlap=N_FFT - HOP)
    mag = np.abs(Zxx)
    return 20 * np.log10(mag + 1e-10)


def average_spectrum(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute average magnitude spectrum in dB. Returns (freqs, magnitude_db)."""
    _, _, Zxx = scipy_stft(y, fs=SR, nperseg=N_FFT, noverlap=N_FFT - HOP)
    mag = np.mean(np.abs(Zxx), axis=1)
    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / SR)[:len(mag)]
    return freqs, 20 * np.log10(mag + 1e-10)


# ─── Mel Filter Bank (replaces librosa dependency) ──────────────────────────

def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int = MEL_N_BANDS) -> np.ndarray:
    """Create a Mel filter bank matrix."""
    fmax = sr / 2.0
    mel_low = hz_to_mel(np.array([0.0]))[0]
    mel_high = hz_to_mel(np.array([fmax]))[0]
    mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    n_freqs = n_fft // 2 + 1
    fb = np.zeros((n_mels, n_freqs))

    for i in range(n_mels):
        left, center, right = bin_points[i], bin_points[i + 1], bin_points[i + 2]
        for j in range(left, center):
            if center != left:
                fb[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right != center:
                fb[i, j] = (right - j) / (right - center)

    return fb


def compute_mfcc(y: np.ndarray, n_mfcc: int = MFCC_N) -> np.ndarray:
    """Compute MFCCs from audio signal."""
    _, _, Zxx = scipy_stft(y, fs=SR, nperseg=N_FFT, noverlap=N_FFT - HOP)
    power = np.abs(Zxx) ** 2

    fb = mel_filterbank(SR, N_FFT, MEL_N_BANDS)
    mel_spec = fb @ power[:fb.shape[1], :]
    log_mel = np.log(mel_spec + 1e-10)

    # DCT-II (type 2) to get MFCCs
    from scipy.fft import dct
    mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[:n_mfcc, :]
    return mfcc


# ─── Quality Metrics ────────────────────────────────────────────────────────

def compute_snr(reference: np.ndarray, test: np.ndarray) -> float:
    """SNR treating difference as noise (dB). Higher = better.

    Note: This metric is only meaningful when reference and test are
    well-aligned versions of the same source. Returns negative values
    when signals differ significantly (e.g., different encoding chains).
    """
    noise = reference - test
    sig_power = np.mean(reference ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-20:
        return 100.0
    return 10 * np.log10(sig_power / noise_power)


def spectral_correlation(reference: np.ndarray, test: np.ndarray) -> float:
    """Pearson correlation of average magnitude spectra (0-1). Higher = better.

    More robust than time-domain SNR for comparing re-encoded audio,
    as it is insensitive to phase shifts and small timing differences.
    """
    _, mag_ref = average_spectrum(reference)
    _, mag_test = average_spectrum(test)
    min_len = min(len(mag_ref), len(mag_test))
    mag_ref, mag_test = mag_ref[:min_len], mag_test[:min_len]

    # Pearson correlation
    ref_centered = mag_ref - np.mean(mag_ref)
    test_centered = mag_test - np.mean(mag_test)
    denom = norm(ref_centered) * norm(test_centered)
    if denom < 1e-20:
        return 0.0
    return float(np.dot(ref_centered, test_centered) / denom)


def spectral_centroid_diff(reference: np.ndarray, test: np.ndarray) -> float:
    """Absolute difference in average spectral centroid (Hz)."""
    def _centroid(y: np.ndarray) -> float:
        freqs, mag_db = average_spectrum(y)
        mag_linear = 10 ** (mag_db / 20)
        total = np.sum(mag_linear)
        if total < 1e-20:
            return 0.0
        return float(np.sum(freqs * mag_linear) / total)

    return abs(_centroid(reference) - _centroid(test))


def mfcc_similarity(reference: np.ndarray, test: np.ndarray) -> float:
    """Cosine similarity of MFCC feature vectors (0-1). Higher = better."""
    mfcc_ref = compute_mfcc(reference).flatten()
    mfcc_test = compute_mfcc(test).flatten()
    min_len = min(len(mfcc_ref), len(mfcc_test))
    mfcc_ref, mfcc_test = mfcc_ref[:min_len], mfcc_test[:min_len]
    denom = norm(mfcc_ref) * norm(mfcc_test)
    if denom < 1e-20:
        return 0.0
    return float(np.dot(mfcc_ref, mfcc_test) / denom)


def rms_difference(reference: np.ndarray, test: np.ndarray) -> float:
    """RMS energy difference as percentage."""
    rms_ref = np.sqrt(np.mean(reference ** 2))
    rms_test = np.sqrt(np.mean(test ** 2))
    if rms_ref < 1e-20:
        return 0.0
    return abs(rms_ref - rms_test) / rms_ref * 100


def to_mid_side(stereo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert stereo signal shape (2, N) to (mid, side)."""
    if stereo.ndim != 2 or stereo.shape[0] != 2:
        raise ValueError("to_mid_side expects shape (2, samples)")
    left, right = stereo[0], stereo[1]
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    return mid, side


def side_energy_ratio(stereo: np.ndarray) -> float:
    """Return side-to-mid RMS ratio (0+), larger means wider stereo image."""
    mid, side = to_mid_side(stereo)
    rms_mid = np.sqrt(np.mean(mid ** 2))
    rms_side = np.sqrt(np.mean(side ** 2))
    if rms_mid < 1e-20:
        if rms_side < 1e-20:
            return 0.0
        # Very wide/anti-phase case: keep finite but high ratio.
        return 100.0
    if rms_mid < 1e-8:
        return min(float(rms_side / 1e-8), 100.0)
    if rms_side < 1e-12:
        return 0.0
    return float(rms_side / rms_mid)


def side_energy_ratio_diff(reference_stereo: np.ndarray, test_stereo: np.ndarray) -> float:
    """Absolute side-to-mid ratio drift (%), lower = closer spatial width."""
    ref_ratio = side_energy_ratio(reference_stereo)
    test_ratio = side_energy_ratio(test_stereo)
    # Near-mono reference has negligible side field; ratio drift is not meaningful.
    if ref_ratio < 1e-3:
        return 0.0
    return abs(ref_ratio - test_ratio) / ref_ratio * 100


def side_spectral_correlation(reference_stereo: np.ndarray, test_stereo: np.ndarray) -> float:
    """Spectral correlation on side channel only (higher = better)."""
    _, side_ref = to_mid_side(reference_stereo)
    _, side_test = to_mid_side(test_stereo)
    ref_rms = np.sqrt(np.mean(side_ref ** 2))
    test_rms = np.sqrt(np.mean(side_test ** 2))
    if ref_rms < 1e-5 and test_rms < 1e-5:
        return 1.0
    if ref_rms < 1e-5:
        return 0.0
    return spectral_correlation(side_ref, side_test)


def side_high_freq_loss(reference_stereo: np.ndarray, test_stereo: np.ndarray, cutoff_hz: float = 6000.0) -> float:
    """Absolute high-frequency side difference in dB above cutoff (lower = better)."""
    _, side_ref = to_mid_side(reference_stereo)
    _, side_test = to_mid_side(test_stereo)
    if np.sqrt(np.mean(side_ref ** 2)) < 1e-5:
        return 0.0
    freqs_ref, mag_ref = average_spectrum(side_ref)
    freqs_test, mag_test = average_spectrum(side_test)
    n = min(len(freqs_ref), len(freqs_test), len(mag_ref), len(mag_test))
    freqs_ref = freqs_ref[:n]
    mag_ref = mag_ref[:n]
    mag_test = mag_test[:n]

    mask = freqs_ref >= cutoff_hz
    if not np.any(mask):
        return 0.0
    ref_hf = float(np.mean(mag_ref[mask]))
    test_hf = float(np.mean(mag_test[mask]))
    return abs(ref_hf - test_hf)


def effective_bandwidth(y: np.ndarray, rolloff_db: float = -20.0) -> float:
    """Find effective bandwidth: highest frequency where average spectrum is within
    rolloff_db of the peak in the 1-10kHz range.

    This reliably detects the hard lowpass filter that low-bitrate AAC applies.
    The 99% cumulative energy approach doesn't work well because energy is
    dominated by low frequencies regardless of bitrate.
    """
    freqs, mag_db = average_spectrum(y)

    # Find peak magnitude in the 1-10kHz range (where most audio content lives)
    mask_mid = (freqs >= 1000) & (freqs <= 10000)
    if not np.any(mask_mid):
        return float(SR / 2)
    peak_db = np.max(mag_db[mask_mid])

    # Find the highest frequency where magnitude is above (peak + rolloff)
    threshold = peak_db + rolloff_db
    above = freqs[mag_db >= threshold]
    if len(above) == 0:
        return 0.0
    return float(above[-1])


def log_spectral_distance(reference: np.ndarray, test: np.ndarray) -> float:
    """Average log-spectral distance (dB). Lower = better."""
    _, _, Zxx_ref = scipy_stft(reference, fs=SR, nperseg=N_FFT, noverlap=N_FFT - HOP)
    _, _, Zxx_test = scipy_stft(test, fs=SR, nperseg=N_FFT, noverlap=N_FFT - HOP)

    S_ref = np.maximum(np.abs(Zxx_ref) ** 2, 1e-10)
    S_test = np.maximum(np.abs(Zxx_test) ** 2, 1e-10)

    min_frames = min(S_ref.shape[1], S_test.shape[1])
    S_ref, S_test = S_ref[:, :min_frames], S_test[:, :min_frames]

    diff_db = 10 * np.log10(S_ref) - 10 * np.log10(S_test)
    return float(np.mean(np.sqrt(np.mean(diff_db ** 2, axis=0))))


def composite_score(spec_corr: float, mfcc_sim: float, centroid_diff: float,
                    bandwidth_diff: float, lsd: float) -> float:
    """Weighted composite similarity score (0-100). Higher = closer to original."""
    corr_score = max(spec_corr, 0.0)
    mfcc_score = max(mfcc_sim, 0.0)
    centroid_score = max(1.0 - centroid_diff / 2000.0, 0.0)
    bw_score = max(1.0 - abs(bandwidth_diff) / 10000.0, 0.0)
    lsd_score = max(1.0 - lsd / 20.0, 0.0)

    weights = [0.25, 0.25, 0.15, 0.20, 0.15]
    scores = [corr_score, mfcc_score, centroid_score, bw_score, lsd_score]
    return sum(w * s for w, s in zip(weights, scores)) * 100


def side_spatial_score(side_corr: float, side_ratio_diff: float, side_hf_loss: float) -> float:
    """Spatial side-channel score (0-100). Higher = closer stereo image."""
    corr_score = max(side_corr, 0.0)
    ratio_score = max(1.0 - side_ratio_diff / 100.0, 0.0)
    hf_score = max(1.0 - side_hf_loss / 20.0, 0.0)
    weights = [0.5, 0.3, 0.2]
    scores = [corr_score, ratio_score, hf_score]
    return sum(w * s for w, s in zip(weights, scores)) * 100


def overall_score(core_score: float, spatial_score: float, spatial_weight: float = 0.2) -> float:
    """Blend core quality score with spatial score."""
    spatial_weight = min(max(spatial_weight, 0.0), 1.0)
    return (1.0 - spatial_weight) * core_score + spatial_weight * spatial_score


def adaptive_spatial_weight(reference_stereo: np.ndarray,
                            max_weight: float = 0.2,
                            full_weight_ratio: float = 0.15) -> float:
    """Downweight side metrics when reference is close to mono."""
    ratio = side_energy_ratio(reference_stereo)
    if full_weight_ratio <= 1e-6:
        return max_weight
    scale = min(max(ratio / full_weight_ratio, 0.0), 1.0)
    return max_weight * scale


# ─── Visualization ───────────────────────────────────────────────────────────

def create_comparison_chart(
    original: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    info_orig: dict,
    info_high: dict,
    info_low: dict,
    metrics_high: dict,
    metrics_low: dict,
    output_path: str,
) -> None:
    """Generate the comparison visualization PNG."""
    fig = plt.figure(figsize=(18, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # ── Row 1: Spectrograms ──────────────────────────────────────────────
    stft_orig = compute_stft(original)
    stft_high = compute_stft(high)
    stft_low = compute_stft(low)

    vmin = min(stft_orig.min(), stft_high.min(), stft_low.min())
    vmax = max(stft_orig.max(), stft_high.max(), stft_low.max())
    # Clamp for visibility
    vmin = max(vmin, vmax - 80)

    freq_bins = np.linspace(0, SR / 2, stft_orig.shape[0])
    time_orig = np.linspace(0, len(original) / SR, stft_orig.shape[1])

    spectro_data = [
        (stft_orig, f"Original (~{info_orig['bitrate_kbps']}kbps)", time_orig),
        (stft_high, f"High BR ({info_high['bitrate_kbps']}kbps)", time_orig),
        (stft_low, f"Low BR ({info_low['bitrate_kbps']}kbps)", time_orig),
    ]

    for i, (data, title, t) in enumerate(spectro_data):
        ax = fig.add_subplot(gs[0, i])
        im = ax.pcolormesh(
            t, freq_bins / 1000, data,
            vmin=vmin, vmax=vmax,
            cmap="magma", shading="auto",
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("Frequency (kHz)" if i == 0 else "")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(0, 24)
        if i == 2:
            plt.colorbar(im, ax=ax, label="dB")

    # ── Row 2: Frequency Response Overlay ────────────────────────────────
    ax_freq = fig.add_subplot(gs[1, :])

    freqs_o, mag_o = average_spectrum(original)
    freqs_h, mag_h = average_spectrum(high)
    freqs_l, mag_l = average_spectrum(low)

    ax_freq.plot(freqs_o / 1000, mag_o, color=C_ORIG, alpha=0.8,
                 label=f"Original (~{info_orig['bitrate_kbps']}kbps)", linewidth=1.2)
    ax_freq.plot(freqs_h / 1000, mag_h, color=C_HIGH, alpha=0.8,
                 label=f"High BR ({info_high['bitrate_kbps']}kbps)", linewidth=1.2)
    ax_freq.plot(freqs_l / 1000, mag_l, color=C_LOW, alpha=0.8,
                 label=f"Low BR ({info_low['bitrate_kbps']}kbps)", linewidth=1.2)

    # Effective bandwidth markers
    bw_o = metrics_high.get("bw_orig", effective_bandwidth(original))
    bw_h = metrics_high["effective_bw"]
    bw_l = metrics_low["effective_bw"]
    ax_freq.axvline(bw_o / 1000, color=C_ORIG, linestyle="--", alpha=0.5,
                    label=f"BW orig: {bw_o/1000:.1f}kHz")
    ax_freq.axvline(bw_h / 1000, color=C_HIGH, linestyle="--", alpha=0.5,
                    label=f"BW high: {bw_h/1000:.1f}kHz")
    ax_freq.axvline(bw_l / 1000, color=C_LOW, linestyle="--", alpha=0.5,
                    label=f"BW low: {bw_l/1000:.1f}kHz")

    ax_freq.set_xlabel("Frequency (kHz)", fontsize=11)
    ax_freq.set_ylabel("Magnitude (dB)", fontsize=11)
    ax_freq.set_title("Average Frequency Response Comparison", fontsize=12, fontweight="bold")
    ax_freq.set_xlim(0, 24)
    ax_freq.legend(loc="upper right", fontsize=9)
    ax_freq.grid(True, alpha=0.3)

    # ── Row 3: Metric Charts ─────────────────────────────────────────────

    # 3a: Normalized per-metric scores (0-1 scale, higher = better)
    ax_snr = fig.add_subplot(gs[2, 0])
    metric_names = [
        "Spectral\nCorr", "MFCC\nSim", "Centroid\nΔ", "LSD", "Bandwidth",
        "Side\nCorr", "Side Ratio\nDrift", "Side HF\nLoss",
    ]

    def _normalize_metrics(m: dict, bw_orig: float) -> list:
        """Normalize each metric to 0-1 (higher = closer to original)."""
        return [
            max(m["spec_corr"], 0.0),
            max(m["mfcc_sim"], 0.0),
            max(1.0 - m["centroid_diff"] / 2000.0, 0.0),
            max(1.0 - m["lsd"] / 20.0, 0.0),
            max(1.0 - abs(m["effective_bw"] - bw_orig) / 10000.0, 0.0),
            max(m["side_corr"], 0.0),
            max(1.0 - m["side_ratio_diff"] / 100.0, 0.0),
            max(1.0 - m["side_hf_loss"] / 20.0, 0.0),
        ]

    norm_high = _normalize_metrics(metrics_high, metrics_high["bw_orig"])
    norm_low = _normalize_metrics(metrics_low, metrics_low["bw_orig"])

    x_pos = np.arange(len(metric_names))
    width = 0.35
    ax_snr.bar(x_pos - width / 2, norm_high,
               width, color=C_HIGH, label=f"{info_high['bitrate_kbps']}kbps")
    ax_snr.bar(x_pos + width / 2, norm_low,
               width, color=C_LOW, label=f"{info_low['bitrate_kbps']}kbps")
    ax_snr.set_xticks(x_pos)
    ax_snr.set_xticklabels(metric_names, fontsize=8)
    ax_snr.set_ylabel("Normalized Score (0-1)")
    ax_snr.set_ylim(0, 1.1)
    side_w = metrics_high.get("spatial_weight", 0.2) * 100
    ax_snr.set_title(f"Per-Metric Scores (including Side, weight {side_w:.0f}%)",
                     fontsize=12, fontweight="bold")
    ax_snr.legend(fontsize=9)
    ax_snr.grid(True, axis="y", alpha=0.3)

    # 3b: Effective Bandwidth
    ax_bw = fig.add_subplot(gs[2, 1])
    bars_bw = ax_bw.bar(
        [0, 1, 2],
        [bw_o / 1000, bw_h / 1000, bw_l / 1000],
        color=[C_ORIG, C_HIGH, C_LOW],
    )
    ax_bw.set_xticks([0, 1, 2])
    ax_bw.set_xticklabels(["Original", f"{info_high['bitrate_kbps']}kbps",
                            f"{info_low['bitrate_kbps']}kbps"])
    ax_bw.set_ylabel("kHz")
    ax_bw.set_title("Effective Bandwidth (-20dB rolloff)", fontsize=12, fontweight="bold")
    ax_bw.axhline(y=SR / 2000, color="gray", linestyle=":", alpha=0.5,
                  label=f"Nyquist ({SR/2000:.0f}kHz)")
    ax_bw.legend(fontsize=9)
    ax_bw.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars_bw, [bw_o / 1000, bw_h / 1000, bw_l / 1000]):
        ax_bw.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                   f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 3c: Composite and Overall Scores
    ax_score = fig.add_subplot(gs[2, 2])
    score_high = metrics_high["overall"]
    score_low = metrics_low["overall"]
    bars_s = ax_score.bar(
        [0, 1],
        [score_high, score_low],
        color=[C_HIGH, C_LOW],
    )
    ax_score.set_xticks([0, 1])
    ax_score.set_xticklabels([f"{info_high['bitrate_kbps']}kbps",
                               f"{info_low['bitrate_kbps']}kbps"])
    ax_score.set_ylabel("Score (0-100)")
    ax_score.set_title("Overall Score (Core + Side)", fontsize=12, fontweight="bold")
    ax_score.set_ylim(0, 100)
    ax_score.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars_s, [score_high, score_low]):
        color = "#2E7D32" if val >= 70 else ("#FF8F00" if val >= 40 else "#C62828")
        ax_score.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                      f"{val:.1f}", ha="center", va="bottom", fontsize=14,
                      fontweight="bold", color=color)

    fig.suptitle("Audio Quality Comparison Report", fontsize=16, fontweight="bold", y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


# ─── Console Report ──────────────────────────────────────────────────────────

def _count_metric_votes(metrics_high: dict, metrics_low: dict) -> dict:
    """Count per-metric wins. Returns {"high": N, "low": N, "tie": N}.

    Metric interpretation:
      - spec_corr, mfcc_sim: higher is better (closer to 1.0)
      - centroid_diff, rms_diff, lsd: lower is better (closer to 0)
      - effective_bw: closer to bw_orig is better
      - side_corr: higher is better
      - side_ratio_diff, side_hf_loss: lower is better
    """
    high_wins = 0
    low_wins = 0
    ties = 0
    eps_map = {
        "spec_corr": 1e-3,
        "mfcc_sim": 1e-3,
        "centroid_diff": 1.0,
        "rms_diff": 0.1,
        "lsd": 0.1,
        "side_corr": 1e-3,
        "side_ratio_diff": 0.5,
        "side_hf_loss": 0.2,
    }

    # Higher-is-better metrics
    for key in ("spec_corr", "mfcc_sim"):
        h, l = metrics_high[key], metrics_low[key]
        eps = eps_map[key]
        if abs(h - l) < eps:
            ties += 1
        elif h > l:
            high_wins += 1
        else:
            low_wins += 1

    # Lower-is-better core metrics
    for key in ("centroid_diff", "rms_diff", "lsd"):
        h, l = metrics_high[key], metrics_low[key]
        eps = eps_map[key]
        if abs(h - l) < eps:
            ties += 1
        elif h < l:
            high_wins += 1
        else:
            low_wins += 1

    # Side metrics are counted only when reference side field is meaningful.
    if metrics_high.get("spatial_weight", 0.2) >= 0.05:
        h, l = metrics_high["side_corr"], metrics_low["side_corr"]
        eps = eps_map["side_corr"]
        if abs(h - l) < eps:
            ties += 1
        elif h > l:
            high_wins += 1
        else:
            low_wins += 1

        for key in ("side_ratio_diff", "side_hf_loss"):
            h, l = metrics_high[key], metrics_low[key]
            eps = eps_map[key]
            if abs(h - l) < eps:
                ties += 1
            elif h < l:
                high_wins += 1
            else:
                low_wins += 1

    # Bandwidth: closer to original is better
    bw_orig = metrics_high["bw_orig"]
    eps_bw = 50.0  # Hz
    diff_h = abs(metrics_high["effective_bw"] - bw_orig)
    diff_l = abs(metrics_low["effective_bw"] - bw_orig)
    if abs(diff_h - diff_l) < eps_bw:
        ties += 1
    elif diff_h < diff_l:
        high_wins += 1
    else:
        low_wins += 1

    return {"high": high_wins, "low": low_wins, "tie": ties}


def print_report(
    info_orig: dict,
    info_high: dict,
    info_low: dict,
    metrics_high: dict,
    metrics_low: dict,
    output_path: str,
) -> None:
    sep = "=" * 62
    line = "-" * 62

    print(f"\n{sep}")
    print("         AUDIO QUALITY COMPARISON REPORT")
    print(sep)

    print("\nFile Properties:")
    print(f"  Original : {info_orig['path']}")
    print(f"             {info_orig['bitrate_kbps']}kbps, {info_orig['duration']:.1f}s, {info_orig['size_mb']:.2f}MB")
    print(f"  High BR  : {info_high['path']}")
    print(f"             {info_high['bitrate_kbps']}kbps, {info_high['duration']:.1f}s, {info_high['size_mb']:.2f}MB")
    print(f"  Low BR   : {info_low['path']}")
    print(f"             {info_low['bitrate_kbps']}kbps, {info_low['duration']:.1f}s, {info_low['size_mb']:.2f}MB")

    print(f"\nObjective Quality Metrics (vs Original):")
    print(f"  {'Metric':<28} {'High BR':>10} {'Low BR':>10}")
    print(f"  {line}")
    print(f"  {'Spectral Corr [0-1]':<28} {metrics_high['spec_corr']:>10.4f} {metrics_low['spec_corr']:>10.4f}")
    print(f"  {'  [higher=better]':<28}")
    print(f"  {'MFCC Cosine Sim [0-1]':<28} {metrics_high['mfcc_sim']:>10.4f} {metrics_low['mfcc_sim']:>10.4f}")
    print(f"  {'Spectral Centroid Δ (Hz)':<28} {metrics_high['centroid_diff']:>10.1f} {metrics_low['centroid_diff']:>10.1f}")
    print(f"  {'RMS Energy Δ (%)':<28} {metrics_high['rms_diff']:>10.2f} {metrics_low['rms_diff']:>10.2f}")
    print(f"  {'Log-Spectral Dist (dB)':<28} {metrics_high['lsd']:>10.2f} {metrics_low['lsd']:>10.2f}")
    print(f"  {'  [lower=better]':<28}")
    print(f"  {'Effective Bandwidth (kHz)':<28} {metrics_high['effective_bw']/1000:>10.1f} {metrics_low['effective_bw']/1000:>10.1f}")
    print(f"  {'  (Original: {:.1f} kHz)'.format(metrics_high['bw_orig']/1000):<28}")
    print(f"  {'Side Spectral Corr [0-1]':<28} {metrics_high['side_corr']:>10.4f} {metrics_low['side_corr']:>10.4f}")
    print(f"  {'Side Ratio Drift (%)':<28} {metrics_high['side_ratio_diff']:>10.2f} {metrics_low['side_ratio_diff']:>10.2f}")
    print(f"  {'Side HF Loss >6k (dB)':<28} {metrics_high['side_hf_loss']:>10.2f} {metrics_low['side_hf_loss']:>10.2f}")
    print(f"  {line}")
    print(f"  {'CORE SCORE [0-100]':<28} {metrics_high['composite']:>10.1f} {metrics_low['composite']:>10.1f}")
    print(f"  {'SPATIAL SIDE SCORE [0-100]':<28} {metrics_high['side_score']:>10.1f} {metrics_low['side_score']:>10.1f}")
    print(f"  {'OVERALL SCORE [0-100]':<28} {metrics_high['overall']:>10.1f} {metrics_low['overall']:>10.1f}")
    print(f"  {'  (Side weight in overall)':<28} {metrics_high['spatial_weight']*100:>9.1f}%")
    if metrics_high["spatial_weight"] < 0.05:
        print(f"  {'  [note: source is near-mono; side metrics downweighted]':<28}")

    score_h = metrics_high["overall"]
    score_l = metrics_low["overall"]

    # Per-metric voting: count how many metrics favor each version
    # For each metric, determine which version is closer to original
    metric_votes = _count_metric_votes(metrics_high, metrics_low)
    high_wins = metric_votes["high"]
    low_wins = metric_votes["low"]
    ties = metric_votes["tie"]

    print(f"\n{'CONCLUSION':}")
    print(f"  Per-metric voting: {info_high['bitrate_kbps']}kbps wins {high_wins}, "
          f"{info_low['bitrate_kbps']}kbps wins {low_wins}, ties {ties}")

    diff = abs(score_h - score_l)
    if diff < 3.0 and high_wins == low_wins:
        print(f"  Both versions are very close in quality to the original.")
    elif score_h > score_l:
        degree = "significantly" if diff > 10 else "moderately" if diff > 5 else "slightly"
        print(f"  The {info_high['bitrate_kbps']}kbps version is {degree} closer")
        print(f"  to the original than the {info_low['bitrate_kbps']}kbps version.")
    else:
        degree = "significantly" if diff > 10 else "moderately" if diff > 5 else "slightly"
        print(f"  The {info_low['bitrate_kbps']}kbps version is {degree} closer")
        print(f"  to the original than the {info_high['bitrate_kbps']}kbps version.")

    print(f"    - {info_high['bitrate_kbps']}kbps overall score: {score_h:.1f} / 100")
    print(f"    - {info_low['bitrate_kbps']}kbps overall score:  {score_l:.1f} / 100")
    if score_l > 0 and score_h > 0:
        ratio = max(score_h, score_l) / min(score_h, score_l)
        winner = info_high['bitrate_kbps'] if score_h >= score_l else info_low['bitrate_kbps']
        print(f"    - Quality ratio: {winner}kbps scores {ratio:.2f}x higher")

    print(f"\nVisualization saved to: {output_path}")
    print(sep)


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare AAC audio quality across bitrates"
    )
    parser.add_argument("original", help="Path to original AAC file")
    parser.add_argument("high_bitrate", help="Path to high-bitrate AAC file")
    parser.add_argument("low_bitrate", help="Path to low-bitrate AAC file")
    parser.add_argument(
        "-o", "--output",
        default="audio_quality_comparison.png",
        help="Output PNG path (default: audio_quality_comparison.png)",
    )
    args = parser.parse_args()

    # File info
    print("Loading file metadata...")
    info_orig = get_file_info(args.original)
    info_high = get_file_info(args.high_bitrate)
    info_low = get_file_info(args.low_bitrate)

    # Load audio (stereo for Mid/Side analysis, mono derived for core metrics)
    print("Decoding audio via ffmpeg...")
    y_orig_st = load_audio(args.original, channels=2)
    y_high_st = load_audio(args.high_bitrate, channels=2)
    y_low_st = load_audio(args.low_bitrate, channels=2)
    y_orig_st, y_high_st, y_low_st = trim_to_shortest(y_orig_st, y_high_st, y_low_st)

    y_orig = np.mean(y_orig_st, axis=0)
    y_high = np.mean(y_high_st, axis=0)
    y_low = np.mean(y_low_st, axis=0)
    y_orig_raw = y_orig.copy()
    y_high_raw = y_high.copy()
    y_low_raw = y_low.copy()

    # Cross-correlation alignment to compensate for encoder delay
    print("Aligning signals via cross-correlation...")
    shift_h = estimate_shift(y_orig, y_high)
    shift_l = estimate_shift(y_orig, y_low)
    y_orig_h, y_high = apply_shift(y_orig.copy(), y_high, shift_h)
    y_orig_l, y_low = apply_shift(y_orig.copy(), y_low, shift_l)
    y_orig_st_h, y_high_st = apply_shift(y_orig_st.copy(), y_high_st, shift_h)
    y_orig_st_l, y_low_st = apply_shift(y_orig_st.copy(), y_low_st, shift_l)

    # Build a common aligned triplet for visualization to avoid pairwise offset bias
    y_orig_vis, y_high_vis, y_low_vis = build_common_aligned_triplet(
        y_orig_raw, y_high_raw, y_low_raw, shift_h, shift_l
    )

    # Trim metric and stereo streams to a common length
    min_all = min(
        len(y_orig_h), len(y_orig_l), len(y_high), len(y_low),
        len(y_orig_vis), len(y_high_vis), len(y_low_vis),
        y_orig_st_h.shape[-1], y_orig_st_l.shape[-1], y_high_st.shape[-1], y_low_st.shape[-1],
    )
    y_high = y_high[:min_all]
    y_low = y_low[:min_all]
    y_orig_vis = y_orig_vis[:min_all]
    y_high_vis = y_high_vis[:min_all]
    y_low_vis = y_low_vis[:min_all]
    y_orig_h = y_orig_h[:min_all]
    y_orig_l = y_orig_l[:min_all]
    y_orig_st_h = y_orig_st_h[:, :min_all]
    y_orig_st_l = y_orig_st_l[:, :min_all]
    y_high_st = y_high_st[:, :min_all]
    y_low_st = y_low_st[:, :min_all]
    print(f"Aligned to {min_all} samples ({min_all/SR:.1f}s)")

    # Compute metrics
    print("Computing quality metrics...")
    bw_orig = effective_bandwidth(y_orig_vis)
    spatial_weight = adaptive_spatial_weight(y_orig_st_h)

    metrics_high = {
        "spec_corr": spectral_correlation(y_orig_h, y_high),
        "mfcc_sim": mfcc_similarity(y_orig_h, y_high),
        "centroid_diff": spectral_centroid_diff(y_orig_h, y_high),
        "rms_diff": rms_difference(y_orig_h, y_high),
        "lsd": log_spectral_distance(y_orig_h, y_high),
        "effective_bw": effective_bandwidth(y_high),
        "side_corr": side_spectral_correlation(y_orig_st_h, y_high_st),
        "side_ratio_diff": side_energy_ratio_diff(y_orig_st_h, y_high_st),
        "side_hf_loss": side_high_freq_loss(y_orig_st_h, y_high_st),
        "bw_orig": bw_orig,
    }
    metrics_high["composite"] = composite_score(
        metrics_high["spec_corr"],
        metrics_high["mfcc_sim"],
        metrics_high["centroid_diff"],
        metrics_high["effective_bw"] - bw_orig,
        metrics_high["lsd"],
    )
    metrics_high["side_score"] = side_spatial_score(
        metrics_high["side_corr"],
        metrics_high["side_ratio_diff"],
        metrics_high["side_hf_loss"],
    )
    metrics_high["spatial_weight"] = spatial_weight
    metrics_high["overall"] = overall_score(
        metrics_high["composite"], metrics_high["side_score"], spatial_weight=spatial_weight
    )

    metrics_low = {
        "spec_corr": spectral_correlation(y_orig_l, y_low),
        "mfcc_sim": mfcc_similarity(y_orig_l, y_low),
        "centroid_diff": spectral_centroid_diff(y_orig_l, y_low),
        "rms_diff": rms_difference(y_orig_l, y_low),
        "lsd": log_spectral_distance(y_orig_l, y_low),
        "effective_bw": effective_bandwidth(y_low),
        "side_corr": side_spectral_correlation(y_orig_st_l, y_low_st),
        "side_ratio_diff": side_energy_ratio_diff(y_orig_st_l, y_low_st),
        "side_hf_loss": side_high_freq_loss(y_orig_st_l, y_low_st),
        "bw_orig": bw_orig,
    }
    metrics_low["composite"] = composite_score(
        metrics_low["spec_corr"],
        metrics_low["mfcc_sim"],
        metrics_low["centroid_diff"],
        metrics_low["effective_bw"] - bw_orig,
        metrics_low["lsd"],
    )
    metrics_low["side_score"] = side_spatial_score(
        metrics_low["side_corr"],
        metrics_low["side_ratio_diff"],
        metrics_low["side_hf_loss"],
    )
    metrics_low["spatial_weight"] = spatial_weight
    metrics_low["overall"] = overall_score(
        metrics_low["composite"], metrics_low["side_score"], spatial_weight=spatial_weight
    )

    # Generate visualization
    print("Generating visualization...")
    create_comparison_chart(
        y_orig_vis, y_high_vis, y_low_vis,
        info_orig, info_high, info_low,
        metrics_high, metrics_low,
        args.output,
    )

    # Print report
    print_report(info_orig, info_high, info_low, metrics_high, metrics_low, args.output)


if __name__ == "__main__":
    main()
