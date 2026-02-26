# audio-similarity

Audio quality comparison tool for AAC files at different bitrates.

Compares one original AAC file against a high-bitrate and a low-bitrate version,
producing a visual report (PNG), a console summary, and an optional text report.

## What the script does

`audio_quality_compare.py`:

- Reads metadata via `ffprobe` (bitrate, duration, file size).
- Decodes audio to stereo float PCM (48 kHz) via `ffmpeg`.
- Aligns signals using two-stage cross-correlation (handles multi-second offsets
  from leading silence, encoder pipeline trimming, etc.).
- Computes objective quality metrics across three dimensions:
  **Core** (spectral fidelity), **Detail** (temporal fine-structure), and
  **Spatial** (stereo image preservation via Mid/Side analysis).
- Produces a weighted overall score and per-metric voting.

## Metrics

### Core metrics (spectral fidelity)

| Metric | Range | Direction | What it measures |
|---|---|---|---|
| **Spectral Correlation** | 0–1 | higher = better | Pearson correlation of the average magnitude spectra. Measures how closely the overall frequency shape matches the original. Insensitive to phase or timing offsets. |
| **MFCC Cosine Similarity** | 0–1 | higher = better | Cosine similarity of Mel-Frequency Cepstral Coefficient vectors. Captures timbral / perceptual similarity the way the human auditory system groups frequencies. |
| **Spectral Centroid Δ** | Hz | lower = better | Absolute difference in the spectral "center of mass." A shift indicates that the frequency balance has changed (e.g., lost highs = centroid moves down). |
| **RMS Energy Δ** | % | lower = better | Percentage difference in overall loudness (RMS level). Catches gain changes introduced by the codec. |
| **Log-Spectral Distance (LSD)** | dB | lower = better | Frame-by-frame RMS of log-spectral differences. More granular than spectral correlation; penalizes per-frame deviations rather than averaging them away. |
| **Effective Bandwidth** | kHz | closer to original = better | Highest frequency where the average spectrum is within −20 dB of the 1–10 kHz peak. Detects the hard low-pass filter that low-bitrate AAC applies. |

### Detail metrics (temporal fine-structure)

These metrics capture the "detail" and "liveliness" that listeners notice — transient sharpness, high-frequency texture, and micro-dynamics.

| Metric | Range | Direction | What it measures |
|---|---|---|---|
| **Envelope Correlation** | 0–1 | higher = better | Pearson correlation of frame-level RMS envelopes (2048-sample frames, 512-sample hop). Measures whether the amplitude dynamics — attacks, decays, swells — are preserved. A low value means transients are smeared or dynamics are flattened. |
| **HF Energy Retention** | % | closer to 100 = better | Percentage of the original's energy above 10 kHz that the encoded version retains. Below 100% means high-frequency "air" and texture are lost; the typical complaint of "sounds dull." |
| **Spectral Flux Correlation** | 0–1 | higher = better | Pearson correlation of spectral flux sequences (how rapidly the spectrum changes frame to frame). Captures whether micro-dynamics and "liveliness" are preserved — attacks that snap vs. attacks that are blurred. |

### Spatial metrics (stereo image — Mid/Side analysis)

The signal is decomposed into Mid (L+R) and Side (L−R) components. These metrics evaluate how well the stereo image is preserved.

| Metric | Range | Direction | What it measures |
|---|---|---|---|
| **Side Spectral Correlation** | 0–1 | higher = better | Spectral correlation computed on the Side channel only. Catches stereo detail loss even when the Mid (mono) channel sounds fine. |
| **Side Ratio Drift** | % | lower = better | Change in the Side-to-Mid RMS energy ratio relative to the original. A large drift means the stereo image has widened or (more commonly) collapsed toward mono. |
| **Side HF Loss** | dB | lower = better | Absolute high-frequency magnitude difference on the Side channel above 6 kHz. AAC encoders aggressively compress Side HF content at low bitrates, making the stereo image sound "flat" in the high end. |

### Composite scores

| Score | Formula | What it means |
|---|---|---|
| **Core Score** (0–100) | Weighted blend of spectral correlation, MFCC similarity, centroid Δ, bandwidth, and LSD | Overall spectral fidelity — does it "sound like" the original? |
| **Detail Score** (0–100) | Weighted blend of envelope correlation, HF retention, spectral flux | Temporal fine-structure — does it "feel alive" like the original? |
| **Spatial Side Score** (0–100) | Weighted blend of side correlation, side ratio drift, side HF loss | Stereo image — does the spatial "width" match the original? |
| **Overall Score** (0–100) | 60% Core + 20% Detail + 20% Spatial (spatial weight adapts for near-mono sources) | Single number summary. Higher = closer to original. |

## Requirements

- Python 3.9+
- `ffmpeg` and `ffprobe` available in `PATH`

## Setup (recommended: venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

```bash
python3 audio_quality_compare.py <original.aac> <high_bitrate.aac> <low_bitrate.aac> \
    -o report.png \
    -t result.txt
```

| Flag | Description |
|---|---|
| `-o`, `--output` | Output PNG path (default: `audio_quality_comparison.png`) |
| `-t`, `--text-output` | Save the console report to a text file |

## Output

- **PNG report** with spectrograms, frequency-response overlay, per-metric bars, effective bandwidth chart, and overall score chart.
- **Console report** with file info, all metric values, composite scores, per-metric voting, and a conclusion.
- **Text file** (optional, via `-t`) with the same console report content.
