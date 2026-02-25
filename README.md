# audio-similarity

Audio quality comparison tool for AAC files at different bitrates.

This repository currently provides `audio_quality_compare.py`, which compares:

- one original AAC file,
- one high-bitrate AAC version,
- one low-bitrate AAC version.

It outputs:

- a visual comparison report (`.png`), and
- a console summary with objective quality metrics.

## What the script does

`audio_quality_compare.py` uses:

- `ffprobe` to read bitrate/duration/size metadata,
- `ffmpeg` to decode audio to mono float PCM (48kHz),
- STFT and spectral features (including MFCC),
- multiple objective metrics and a weighted composite score.

It also aligns signals with cross-correlation to reduce encoder delay/padding effects.

## Metrics included

Compared against the original:

- Spectral correlation (higher is better)
- MFCC cosine similarity (higher is better)
- Spectral centroid difference (lower is better)
- RMS energy difference (lower is better)
- Log-spectral distance (lower is better)
- Effective bandwidth difference (closer to original is better)
- Composite similarity score (0-100, higher is better)

## Requirements

- Python 3.9+
- `ffmpeg` and `ffprobe` available in `PATH`

## Setup (recommended: venv)

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows (PowerShell), use:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies inside the virtual environment:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python3 audio_quality_compare.py <original.aac> <high_bitrate.aac> <low_bitrate.aac> -o report.png
```

Example:

```bash
python3 audio_quality_compare.py original.aac aac_128k.aac aac_48k.aac -o audio_quality_comparison.png
```

## Output

- **PNG report** with:
  - spectrograms for all three files,
  - frequency-response overlay,
  - per-metric normalized bars,
  - effective bandwidth chart,
  - composite score chart.
- **Console report** with file info, per-metric values, and final conclusion.

## Files

```text
audio-similarity/
  audio_quality_compare.py
  requirements.txt
  README.md
```
