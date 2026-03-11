# VOX AI - Vocal Acoustic Analysis Tool

Version 2.0 - Note-aware vocal analysis with scientific accuracy

## Overview

VOX AI performs detailed acoustic analysis of vocal recordings, extracting metrics like jitter, shimmer, HNR, and vibrato characteristics. Unlike traditional tools that average across entire recordings, VOX AI uses note-aware processing to analyze sustained note regions separately from ornamented passages.

**Key Features:**
- Note segmentation with pitch tracking
- Sustained region extraction for stability metrics
- Per-note vibrato analysis (rate, extent, regularity)
- Optional demucs source separation
- No "vocal health score" - purely acoustic metrics with appropriate caveats

## Dependencies

### Required Python Packages

```bash
pip install numpy librosa praat-parselmouth scipy demucs
```

Or using uv:
```bash
uv pip install numpy librosa praat-parselmouth scipy demucs
```

**Core Dependencies:**
- **numpy** - Numerical computations
- **librosa** - Audio processing and pitch tracking (librosa.pyin)
- **praat-parselmouth** - Praat integration for jitter/shimmer/HNR
- **scipy** - Signal processing (bandpass filters, median filters)
- **demucs** - Source separation (optional but recommended)

### System Requirements

- **Python 3.8+**
- **ffmpeg** - Required by librosa for audio format support
  ```bash
  # Ubuntu/Debian
  sudo apt-get install ffmpeg

  # macOS
  brew install ffmpeg

  # Windows
  # Download from https://ffmpeg.org/download.html
  ```

### Optional Dependencies

- **demucs CLI** - For automatic vocal separation (installs with pip package)
- **imageio-ffmpeg** - Alternative ffmpeg wrapper (will be used if available)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rustwoodagent-ops/VOX.git
cd VOX/vox-ai
```

2. Create virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install numpy librosa praat-parselmouth scipy demucs
```

4. Verify ffmpeg is installed:
```bash
ffmpeg -version
```

## Usage

### Basic Analysis

```bash
python analyze_vocal.py your_song.mp3
```

### With Source Separation (Recommended)

Demucs will automatically separate vocals if installed:
```bash
python analyze_vocal.py your_song.mp3
# Output: vocals.wav extracted, then analyzed
```

### Output

Results are saved to `vocal_results.json` and printed to console:
- Note segmentation (total notes, sustained vs ornamented)
- Sustained note metrics (jitter, shimmer, HNR)
- Vibrato summary (rate, extent, notes with vibrato)
- Spectral features
- Suppressed metrics (with reasons if insufficient data)

## Architecture

**Version 2.0 Pipeline:**
1. **Source Separation** (optional) - demucs htdemucs model
2. **Pitch Tracking** - librosa.pyin with voiced probability filtering
3. **Note Segmentation** - Boundary detection with max-duration splits
4. **Sustained Region Extraction** - Stable portions of notes for metrics
5. **Per-Note Vibrato Analysis** - Bandpass filtering and autocorrelation
6. **Metric Aggregation** - Weighted averages across valid regions

## Scientific Notes

### Metric Validity
- **Jitter/Shimmer** - Only calculated on sustained note regions (>200ms, <30 cents variation)
- **HNR** - Harmonics-to-Noise Ratio on periodic regions
- **Vibrato** - Detected only on notes meeting criteria (4-8 Hz, 20-300 cents extent)
- Metrics describe **acoustic characteristics**, not vocal health or pathology

### Limitations
- Requires clean vocal input for best results (use demucs for mixed audio)
- Short recordings (<10s) may have insufficient sustained regions
- Pitch tracking accuracy depends on audio quality and separation

## License

[Your License Here]

## Citation

If using this tool in research, please cite:
```
VOX AI - Vocal Acoustic Analysis Tool
https://github.com/rustwoodagent-ops/VOX
```
