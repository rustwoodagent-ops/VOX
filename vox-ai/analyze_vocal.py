#!/usr/bin/env python3
"""
VOX AI - Professional Vocal Pedagogy Analysis Tool
====================================================
Advanced vocal analysis with source separation and professional metrics:
- Source Separation (demucs htdemucs)
- Jitter (Local): Pitch cycle instability
- Shimmer (Local): Amplitude cycle instability
- HNR (Harmonics-to-Noise Ratio): Tone clarity
- Vibrato Analysis: Rate (Hz) and Extent (Cents)
- Spectral Flatness: Noisiness/grit indicator
- Vocal Health Score: Proprietary 1-100 algorithm
"""

import sys
import json
import os
import io
import subprocess
import tempfile
import warnings
import numpy as np
import librosa
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import parselmouth
from parselmouth.praat import call
from scipy import signal
from scipy.ndimage import median_filter

# Filter warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Configure ffmpeg for audio conversion
try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_PATH = None
    FFMPEG_AVAILABLE = False

# Constants
VOCALS_WAV = "vocals.wav"
DEMUCS_MODEL = "htdemucs"


def check_demucs():
    """Check if demucs is available."""
    try:
        result = subprocess.run(['demucs', '--help'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def convert_to_wav(input_path: str, output_path: str) -> str:
    """Convert audio file to WAV format using ffmpeg."""
    if not FFMPEG_AVAILABLE:
        raise RuntimeError("FFmpeg not available via imageio_ffmpeg")

    cmd = [
        FFMPEG_PATH,
        '-y',  # Overwrite output
        '-i', input_path,
        '-ar', '44100',  # 44.1kHz sample rate
        '-ac', '2',      # Stereo
        '-f', 'wav',
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr.decode()[:200]}")

    return output_path


def separate_vocals(input_path: str, output_dir: str = ".") -> Optional[str]:
    """
    Use demucs to separate vocals from the input audio.

    Args:
        input_path: Path to input audio file
        output_dir: Directory to save separated files

    Returns:
        Path to the separated vocals.wav file, or None if separation fails
    """
    print("\n🎵 SOURCE SEPARATION")
    print("=" * 50)

    if not check_demucs():
        print("WARNING: demucs not found in PATH")
        print("Install with: uv pip install demucs")
        return None

    print(f"Running demucs with {DEMUCS_MODEL} model...")
    print("This may take a few minutes depending on file length.")

    # Check if input needs conversion (m4a, aac, etc.)
    ext = Path(input_path).suffix.lower()
    needs_conversion = ext in ['.m4a', '.aac', '.mp4', '.mov']

    # Create temp directory for separation
    with tempfile.TemporaryDirectory() as temp_dir:
        # Convert to wav if necessary
        if needs_conversion:
            print(f"Converting {ext} to WAV for demucs compatibility...")
            wav_path = os.path.join(temp_dir, "input_converted.wav")
            try:
                convert_to_wav(input_path, wav_path)
                demucs_input = wav_path
            except Exception as e:
                print(f"Conversion failed: {e}")
                return None
        else:
            demucs_input = input_path

        demucs_out = os.path.join(temp_dir, "demucs_out")
        os.makedirs(demucs_out, exist_ok=True)

        cmd = [
            'demucs',
            '-n', DEMUCS_MODEL,
            '--two-stems', 'vocals',
            '--out', demucs_out,
            demucs_input
        ]

        print(f"Command: demucs -n {DEMUCS_MODEL} --two-stems vocals --out {demucs_out} <input>")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"\n⚠️  demucs source separation failed")
            print(f"   This usually means ffmpeg shared libraries are missing.")
            print(f"   The analysis will proceed on the original audio.")
            print(f"   For best results, install: sudo apt install ffmpeg libavcodec-dev")
            return None

        # Find the vocals file in the temp directory
        # demucs creates: demucs_out/htdemucs/filename/vocals.wav
        base_name = Path(demucs_input).stem
        vocals_path = os.path.join(demucs_out, DEMUCS_MODEL, base_name, "vocals.wav")

        if not os.path.exists(vocals_path):
            print(f"WARNING: Expected vocals file not found at {vocals_path}")
            return None
            sys.exit(1)

        # Copy to output directory
        output_vocals = os.path.join(output_dir, VOCALS_WAV)
        import shutil
        shutil.copy2(vocals_path, output_vocals)
        print(f"✓ Vocals extracted to: {output_vocals}")

    return output_vocals


def load_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """Load audio file with librosa."""
    print(f"\nLoading audio: {file_path}")
    y, sr = librosa.load(file_path, sr=None, mono=True)
    print(f"✓ Loaded {len(y)/sr:.2f}s at {sr}Hz")
    return y, sr


def calculate_jitter_shimmer_hnr(y: np.ndarray, sr: int) -> Dict:
    """
    Calculate Jitter, Shimmer, and HNR using Praat via parselmouth.

    These are standard professional vocal health metrics used in clinical voice analysis.
    """
    print("\n🎯 CALCULATING JITTER, SHIMMER & HNR")
    print("=" * 50)

    # Convert to parselmouth Sound object
    sound = parselmouth.Sound(y, sampling_frequency=sr)

    # Create PointProcess for pitch analysis
    try:
        pitch = sound.to_pitch_cc()
        point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)",
                                                 75, 600)  # fmin, fmax

        # Calculate Jitter (local)
        # Jitter is the cycle-to-cycle variation of the fundamental frequency
        jitter_local = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)

        # Calculate Shimmer (local)
        # Shimmer is the cycle-to-cycle variation of the amplitude
        shimmer_local = parselmouth.praat.call([sound, point_process], "Get shimmer (local)",
                                                0, 0, 0.0001, 0.02, 1.3, 1.6)

    except Exception as e:
        print(f"  Warning: Could not calculate Jitter/Shimmer: {e}")
        jitter_local = 0.0
        shimmer_local = 0.0

    # Calculate HNR (Harmonics-to-Noise Ratio)
    # HNR measures the periodicity of the signal - higher = clearer, more periodic
    try:
        harmonicity = sound.to_harmonicity_cc()
        hnr_db = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
    except Exception as e:
        print(f"  Warning: Could not calculate HNR: {e}")
        hnr_db = 0.0

    results = {
        'jitter_local_percent': round(float(jitter_local) * 100, 4) if jitter_local else 0.0,
        'jitter_local_description': 'Cycle-to-cycle pitch variation (lower is better, <1.04% normal)',
        'shimmer_local_percent': round(float(shimmer_local) * 100, 4) if shimmer_local else 0.0,
        'shimmer_local_description': 'Cycle-to-cycle amplitude variation (lower is better, <3.81% normal)',
        'hnr_db': round(float(hnr_db), 2),
        'hnr_description': 'Harmonics-to-Noise Ratio (higher is better, >7dB normal)'
    }

    print(f"  Jitter (local): {results['jitter_local_percent']}%")
    print(f"  Shimmer (local): {results['shimmer_local_percent']}%")
    print(f"  HNR: {results['hnr_db']} dB")

    return results


def analyze_vibrato(y: np.ndarray, sr: int) -> Dict:
    """
    Analyze vibrato characteristics.

    Detects vibrato sections and calculates:
    - Rate: How fast the vibrato oscillates (Hz)
    - Extent: How wide the pitch varies (cents)
    """
    print("\n🎼 VIBRATO ANALYSIS")
    print("=" * 50)

    # Extract pitch contour
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr,
        frame_length=2048
    )

    # Remove unvoiced frames
    voiced_f0 = f0[~np.isnan(f0)]

    if len(voiced_f0) < 100:  # Need enough samples for vibrato analysis
        return {
            'vibrato_detected': False,
            'vibrato_rate_hz': 0.0,
            'vibrato_extent_cents': 0.0,
            'vibrato_coverage_percent': 0.0,
            'description': 'Not enough voiced frames for vibrato analysis'
        }

    # Convert to cents relative to mean
    mean_f0 = np.mean(voiced_f0)
    f0_cents = 1200 * np.log2(voiced_f0 / mean_f0)

    # High-pass filter to isolate vibrato (typically 5-8 Hz)
    from scipy import signal
    sos = signal.butter(4, [4, 10], 'bandpass', fs=sr/512, output='sos')  # 512 is hop length default
    f0_cents_filtered = signal.sosfiltfilt(sos, f0_cents)

    # Calculate vibrato characteristics from filtered signal
    # Find zero crossings to estimate rate
    zero_crossings = np.where(np.diff(np.sign(f0_cents_filtered)))[0]
    if len(zero_crossings) > 2:
        # Average period between zero crossings (2 crossings = 1 period)
        periods = np.diff(zero_crossings[::2])  # Take every other crossing
        if len(periods) > 0:
            avg_period_samples = np.mean(periods)
            vibrato_rate = (sr / 512) / (avg_period_samples * 2)  # Convert to Hz
        else:
            vibrato_rate = 0.0
    else:
        vibrato_rate = 0.0

    # Calculate extent (peak-to-peak variation in cents)
    vibrato_extent = np.std(f0_cents_filtered) * 2.8  # Approximate peak-to-peak from std

    # Detect vibrato presence based on rate (typical vibrato is 5-7 Hz) and extent (>30 cents)
    vibrato_detected = 4.5 <= vibrato_rate <= 8.0 and vibrato_extent > 30

    # Estimate vibrato coverage (what % of the signal has clear vibrato)
    # Simple approach: check variation in sliding windows
    window_size = int(sr / 512 * 0.5)  # 0.5 second windows
    if len(f0_cents) > window_size:
        vibrato_windows = 0
        total_windows = len(f0_cents) - window_size
        for i in range(0, total_windows, window_size // 2):
            window = f0_cents[i:i+window_size]
            if np.std(window) > 20:  # Significant variation
                vibrato_windows += 1
        vibrato_coverage = (vibrato_windows / (total_windows / (window_size // 2))) * 100
    else:
        vibrato_coverage = 0.0

    results = {
        'vibrato_detected': bool(vibrato_detected),
        'vibrato_rate_hz': round(float(vibrato_rate), 2) if vibrato_rate > 0 else 0.0,
        'vibrato_extent_cents': round(float(vibrato_extent), 2),
        'vibrato_coverage_percent': round(float(vibrato_coverage), 1),
        'description': 'Vibrato rate (5-7Hz normal), extent (30-150 cents typical)'
    }

    print(f"  Vibrato detected: {'Yes' if vibrato_detected else 'No'}")
    print(f"  Rate: {results['vibrato_rate_hz']} Hz")
    print(f"  Extent: {results['vibrato_extent_cents']} cents")
    print(f"  Coverage: {results['vibrato_coverage_percent']}%")

    return results


def calculate_spectral_flatness(y: np.ndarray, sr: int) -> Dict:
    """
    Calculate spectral flatness (tonality measure).

    Values close to 1.0 indicate noise-like (flat spectrum).
    Values close to 0.0 indicate tonal (pitched).
    """
    print("\n📊 SPECTRAL FLATNESS")
    print("=" * 50)

    # Calculate spectral flatness
    flatness = librosa.feature.spectral_flatness(y=y)[0]

    # Calculate statistics
    flatness_mean = float(np.mean(flatness))
    flatness_std = float(np.std(flatness))

    # Interpret flatness
    # < 0.3: Very tonal (clean voice)
    # 0.3-0.5: Moderately tonal (some grit/distortion)
    # > 0.5: Noisy (whisper, breathy, or distorted)
    if flatness_mean < 0.3:
        interpretation = "Very tonal - clean vocal production"
    elif flatness_mean < 0.5:
        interpretation = "Moderately tonal - possible vocal grit/distortion"
    else:
        interpretation = "Noisy - breathy voice or high distortion"

    results = {
        'spectral_flatness_mean': round(flatness_mean, 4),
        'spectral_flatness_std': round(flatness_std, 4),
        'interpretation': interpretation,
        'description': '0=tonal/pitched, 1=noisy/flat (indicator of grit/distortion)'
    }

    print(f"  Mean flatness: {results['spectral_flatness_mean']}")
    print(f"  Std dev: {results['spectral_flatness_std']}")
    print(f"  Interpretation: {interpretation}")

    return results


NOTE_SEG_PARAMS = {
    'min_note_duration': 0.50,  # Filter out tiny fragments
    'pitch_jump_threshold': 3.0,
    'unvoiced_gap_threshold': 0.35,  # Require longer silence before splitting
    'smoothing_window': 5,
}

SUSTAINED_REGION_PARAMS = {
    'min_duration': 0.20,
    'max_pitch_variation': 30.0,
    'exclude_attack_ms': 50,
    'exclude_release_ms': 50,
}

VIBRATO_PARAMS = {
    'min_note_duration': 0.5,
    'detrend_window_ms': 100,
    'freq_range': (4.0, 8.0),
    'min_extent_cents': 20,
    'max_extent_cents': 300,
    'min_cycles': 2,
    'autocorr_threshold': 0.3,
}


def interpolate_nans(arr: np.ndarray) -> np.ndarray:
    """Linear interpolation of NaN values in array."""
    mask = np.isnan(arr)
    if not mask.any():
        return arr
    arr_filled = arr.copy()
    valid_idx = np.where(~mask)[0]
    if len(valid_idx) == 0:
        return np.zeros_like(arr)
    arr_filled[mask] = np.interp(np.where(mask)[0], valid_idx, arr[valid_idx])
    return arr_filled


def hz_to_cents(hz: np.ndarray, ref_hz: float) -> np.ndarray:
    """Convert Hz to cents relative to reference."""
    return 1200 * np.log2(hz / ref_hz)


def semitones_to_cents(semitones: float) -> float:
    """Convert semitones to cents."""
    return semitones * 100.0


def segment_notes(f0: np.ndarray, voiced_flag: np.ndarray, times: np.ndarray, sr: int):
    """Segment pitch contour into individual sung notes."""
    params = NOTE_SEG_PARAMS

    if len(f0) == 0 or len(times) == 0:
        return []

    min_len = min(len(f0), len(voiced_flag), len(times))
    f0 = f0[:min_len]
    voiced_flag = voiced_flag[:min_len]
    times = times[:min_len]

    f0_filled = interpolate_nans(f0)
    f0_smooth = median_filter(f0_filled, size=params['smoothing_window'])

    valid_f0 = f0[f0 > 0]
    if len(valid_f0) == 0:
        return []

    ref_hz = np.median(valid_f0)
    f0_cents = hz_to_cents(f0_smooth, ref_hz)

    boundaries = [0]
    i = 1
    while i < len(voiced_flag):
        if not voiced_flag[i] and voiced_flag[i-1]:
            gap_start = i
            while i < len(voiced_flag) and not voiced_flag[i]:
                i += 1
            gap_end = i
            if gap_start < len(times) and gap_end < len(times):
                gap_duration = times[gap_end] - times[gap_start]
                if gap_duration > params['unvoiced_gap_threshold']:
                    boundaries.append(gap_start)
                    if gap_end < len(f0):
                        boundaries.append(gap_end)
        elif i > 0 and voiced_flag[i] and voiced_flag[i-1]:
            pitch_diff_cents = abs(f0_cents[i] - f0_cents[i-1])
            if pitch_diff_cents > semitones_to_cents(params['pitch_jump_threshold']):
                boundaries.append(i)
        i += 1

    boundaries.append(len(f0))
    boundaries = sorted(set(boundaries))

    # === PHASE 2: Refine long notes ===
    refined_boundaries = [boundaries[0]]  # Keep start

    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i+1]

        if start_idx >= end_idx:
            continue

        note_duration = times[min(end_idx, len(times)-1)] - times[start_idx]

        # If note is long, check for internal splits
        if note_duration > 3.0:  # Only check notes > 3 seconds
            internal_splits = find_internal_splits(
                f0_cents[start_idx:end_idx],
                voiced_flag[start_idx:end_idx],
                times[start_idx:end_idx],
                start_idx
            )
            refined_boundaries.extend(internal_splits)

        refined_boundaries.append(end_idx)

    # Remove duplicates and sort
    boundaries = sorted(set(refined_boundaries))

    if len(times) >= 2:
        frame_dts = np.diff(times)
        frame_dt = float(np.median(frame_dts))
    else:
        frame_dt = 0.01

    notes = []
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i+1]

        if start_idx >= end_idx or start_idx >= len(times):
            continue

        start_time = float(times[start_idx])
        if end_idx >= len(times):
            end_time = float(times[-1] + frame_dt)
        else:
            end_time = float(times[end_idx])

        note_duration = end_time - start_time
        if note_duration < params['min_note_duration']:
            continue

        note_f0 = f0[start_idx:end_idx].copy()
        note_voiced = voiced_flag[start_idx:end_idx]
        valid_note_f0 = note_f0[note_f0 > 0]
        if len(valid_note_f0) == 0:
            continue

        mean_pitch = np.median(valid_note_f0)

        # Pitch sanity check - reject implausible octaves/noise
        if mean_pitch < 60 or mean_pitch > 900:
            continue  # Skip this "note" - likely noise or harmonic error

        valid_mask = note_f0 > 0
        if np.sum(valid_mask) > 1:
            note_f0_cents = hz_to_cents(note_f0[valid_mask], mean_pitch)
            pitch_std = float(np.std(note_f0_cents))
        else:
            pitch_std = 0.0

        note_type = 'sustained' if pitch_std < 120 else 'ornamented'  # Changed from 50 to 120

        note = {
            'start_time': start_time,
            'end_time': end_time,
            'start_frame': int(start_idx),
            'end_frame': int(end_idx),
            'duration': float(note_duration),
            'f0': note_f0,
            'voiced_mask': note_voiced,
            'mean_pitch_hz': float(mean_pitch),
            'pitch_std_cents': pitch_std,
            'type': note_type,
            'index': len(notes)
        }
        notes.append(note)

    return notes


def find_internal_splits(f0_cents, voiced_flag, times, global_start_idx):
    """
    Find split points inside a long note based on pitch drift and voicing.
    Conservative approach: max duration cap with smart split points.
    """
    splits = []

    if len(f0_cents) < 50:  # Need enough frames
        return splits

    valid_mask = ~np.isnan(f0_cents)

    # === Method 1: Maximum duration hard cap (5 seconds) ===
    frame_rate = len(f0_cents) / (times[-1] - times[0]) if len(times) > 1 else 100
    max_frames = int(5.0 * frame_rate)

    for i in range(max_frames, len(f0_cents), max_frames):
        # Find a good split point near the max duration
        search_start = max(0, i - 20)
        search_end = min(len(f0_cents), i + 20)

        # Look for stable point (closest to median pitch in window)
        window_cents = f0_cents[search_start:search_end]
        window_valid = valid_mask[search_start:search_end]
        if np.sum(window_valid) > 0:
            median_cents = np.median(window_cents[window_valid])
            min_diff_idx = search_start + np.argmin(np.abs(window_cents - median_cents))
            splits.append(global_start_idx + min_diff_idx)

    # === Method 2: Voicing weakness (gaps > 150ms) ===
    min_voiced_run = int(0.15 * frame_rate)  # 150ms in frames
    i = 0
    while i < len(voiced_flag):
        if not voiced_flag[i]:
            gap_start = i
            while i < len(voiced_flag) and not voiced_flag[i]:
                i += 1
            gap_end = i
            gap_duration = gap_end - gap_start

            # Split on significant gaps
            if gap_duration >= min_voiced_run:
                split_point = global_start_idx + (gap_start + gap_end) // 2
                splits.append(split_point)
        else:
            i += 1

    # Remove duplicates and sort
    splits = sorted(set(splits))

    return splits


def extract_sustained_regions(notes, sr=None):
    """Extract stable sub-regions from notes suitable for jitter/shimmer/HNR."""
    params = SUSTAINED_REGION_PARAMS
    all_regions = []
    last_end_frame = -1

    debug_info = {'too_short': 0, 'no_core': 0, 'voiced_fail': 0, 'pitch_fail': 0, 'success': 0}

    for note in notes:
        if note['duration'] < (params['min_duration'] + 0.1):
            debug_info['too_short'] += 1
            continue

        f0 = note['f0']
        voiced = note['voiced_mask']
        frame_rate = len(f0) / note['duration'] if note['duration'] > 0 else 100

        attack_samples = int(params['exclude_attack_ms'] / 1000 * frame_rate)
        release_samples = int(params['exclude_release_ms'] / 1000 * frame_rate)
        start_idx = attack_samples
        end_idx = max(start_idx, len(f0) - release_samples)

        if start_idx >= end_idx:
            debug_info['no_core'] += 1
            continue

        f0_core = f0[start_idx:end_idx]
        voiced_core = voiced[start_idx:end_idx]
        window_duration = params['min_duration']
        window_samples = int(window_duration * frame_rate)

        if window_samples > len(f0_core) or window_samples < 1:
            continue

        hop_samples = window_samples

        note_has_valid_window = False
        for win_start in range(0, len(f0_core) - window_samples + 1, hop_samples):
            window_f0 = f0_core[win_start:win_start + window_samples]
            window_voiced = voiced_core[win_start:win_start + window_samples]

            # Skip voiced_ratio check - rely on pitch stability instead
            valid_f0 = window_f0[window_f0 > 0]
            if len(valid_f0) < window_samples * 0.9:
                debug_info['pitch_fail'] += 1
                continue

            mean_hz = np.median(valid_f0)
            cents = hz_to_cents(valid_f0, mean_hz)
            std_cents = float(np.std(cents))

            if std_cents <= params['max_pitch_variation']:
                note_has_valid_window = True
                region_start_frame = note['start_frame'] + start_idx + win_start
                region_end_frame = region_start_frame + window_samples

                if region_start_frame < last_end_frame:
                    continue

                last_end_frame = region_end_frame
                region_start_time = note['start_time'] + (start_idx + win_start) / frame_rate
                region_end_time = region_start_time + window_duration

                if sr is not None:
                    start_sample = int(region_start_time * sr)
                    end_sample = int(region_end_time * sr)
                else:
                    start_sample = None
                    end_sample = None

                region = {
                    'note_index': note['index'],
                    'start_time': float(region_start_time),
                    'end_time': float(region_end_time),
                    'duration': float(window_duration),
                    'start_frame': int(region_start_frame),
                    'end_frame': int(region_end_frame),
                    'start_sample': start_sample,
                    'end_sample': end_sample,
                    'mean_pitch_hz': float(mean_hz),
                    'pitch_std_cents': std_cents,
                    'frame_rate': float(frame_rate)
                }
                all_regions.append(region)

        if note_has_valid_window:
            debug_info['success'] += 1

    return all_regions


def analyze_single_note_vibrato(note):
    """Analyze vibrato characteristics for a single note."""
    params = VIBRATO_PARAMS

    if note['duration'] < params['min_note_duration']:
        return {
            'valid': False,
            'reason': 'Note too short (%.2fs < %.1fs)' % (note['duration'], params['min_note_duration'])
        }

    f0 = note['f0']
    valid_mask = (f0 > 0) & ~np.isnan(f0)
    valid_proportion = np.mean(valid_mask)
    valid_duration = note['duration'] * valid_proportion

    if valid_proportion < 0.5 or valid_duration < 0.1:
        return {
            'valid': False,
            'reason': 'Insufficient voiced data (%.0f%%, %.3fs)' % (valid_proportion*100, valid_duration)
        }

    f0_valid = f0[valid_mask]
    f0_cents = hz_to_cents(f0_valid, note['mean_pitch_hz'])
    frame_rate = len(f0) / note['duration'] if note['duration'] > 0 else 100

    window_samples = int(params['detrend_window_ms'] / 1000 * frame_rate)
    if window_samples % 2 == 0:
        window_samples += 1
    window_samples = min(window_samples, len(f0_cents) // 2)
    if window_samples < 3:
        return {'valid': False, 'reason': 'Note too short for detrending'}

    trend = median_filter(f0_cents, size=window_samples)
    detrended = f0_cents - trend

    nyquist = frame_rate / 2
    low_freq = params['freq_range'][0]
    high_freq = min(params['freq_range'][1], nyquist * 0.99)

    if low_freq >= high_freq or low_freq <= 0:
        return {'valid': False, 'reason': 'Insufficient sampling rate for vibrato analysis'}

    min_filter_length = 20
    if len(detrended) < min_filter_length:
        return {'valid': False, 'reason': 'Signal too short for filtering (%d < %d samples)' % (len(detrended), min_filter_length)}

    try:
        sos = signal.butter(4, [low_freq, high_freq], btype='band', fs=frame_rate, output='sos')
        vibrato_signal = signal.sosfiltfilt(sos, detrended)
    except ValueError as e:
        return {'valid': False, 'reason': 'Filter error: %s' % str(e)}

    corr = np.correlate(vibrato_signal, vibrato_signal, mode='full')
    corr = corr[len(corr)//2:]

    min_lag = max(1, int(frame_rate / params['freq_range'][1]))
    max_lag = min(int(frame_rate / params['freq_range'][0]), len(corr) - 1)

    if min_lag >= max_lag or max_lag >= len(corr):
        return {'valid': False, 'reason': 'Signal too short for frequency analysis'}

    search_region = corr[min_lag:max_lag]
    if len(search_region) == 0:
        return {'valid': False, 'reason': 'Invalid search region for autocorrelation'}

    peak_idx_in_region = np.argmax(search_region)
    peak_idx = peak_idx_in_region + min_lag
    peak_value = search_region[peak_idx_in_region]
    normalized_peak = peak_value / corr[0] if corr[0] > 0 else 0

    if normalized_peak < params['autocorr_threshold']:
        return {
            'valid': False,
            'reason': 'No periodic modulation detected (autocorr=%.2f)' % normalized_peak,
            'autocorr_peak': float(normalized_peak)
        }

    period_samples = peak_idx
    vibrato_rate = frame_rate / period_samples
    rms = np.sqrt(np.mean(vibrato_signal**2))
    vibrato_extent = 2 * np.sqrt(2) * rms

    peaks, _ = signal.find_peaks(vibrato_signal)
    valleys, _ = signal.find_peaks(-vibrato_signal)
    if len(peaks) >= 2 and len(valleys) >= 2:
        min_len = min(len(peaks), len(valleys))
        peak_heights = vibrato_signal[peaks[:min_len]]
        valley_heights = vibrato_signal[valleys[:min_len]]
        actual_p2p = np.mean(peak_heights) - np.mean(valley_heights)
        vibrato_extent = abs(float(actual_p2p))

    valid_rate = params['freq_range'][0] <= vibrato_rate <= params['freq_range'][1]
    valid_extent = params['min_extent_cents'] <= vibrato_extent <= params['max_extent_cents']
    cycles_observed = note['duration'] * vibrato_rate
    enough_cycles = cycles_observed >= params['min_cycles']

    validation_failures = []
    if not valid_rate:
        validation_failures.append('Rate %.1f Hz outside range' % vibrato_rate)
    if not valid_extent:
        validation_failures.append('Extent %.1f cents outside range' % vibrato_extent)
    if not enough_cycles:
        validation_failures.append('Only %.1f cycles observed' % cycles_observed)

    if validation_failures:
        return {
            'valid': False,
            'reason': '; '.join(validation_failures),
            'rate_hz': float(vibrato_rate),
            'extent_cents': float(vibrato_extent),
            'cycles_observed': float(cycles_observed)
        }

    period_cv = None
    if len(peaks) >= 3:
        peak_times = np.array(peaks) / frame_rate
        periods = np.diff(peak_times)
        period_cv = float(np.std(periods) / np.mean(periods)) if np.mean(periods) > 0 else 0.0

    return {
        'valid': True,
        'rate_hz': float(vibrato_rate),
        'extent_cents': float(vibrato_extent),
        'cycles_observed': int(cycles_observed),
        'regularity_period_cv': period_cv,
        'autocorr_peak': float(normalized_peak),
        'confidence': 'high' if normalized_peak > 0.6 else 'medium'
    }


def calculate_sustained_region_metrics(y, sr, sustained_regions):
    """Calculate jitter, shimmer, and HNR on sustained note regions."""
    print("\n🎯 CALCULATING SUSTAINED-NOTE METRICS")
    print("=" * 50)

    if len(sustained_regions) == 0:
        print("  ⚠️  No sustained regions found")
        return {
            'sustained_regions_found': 0,
            'total_sustained_duration': 0.0,
            'sufficient_data': False,
            'jitter_local_percent': None,
            'shimmer_local_percent': None,
            'hnr_db': None,
            'caveats': ['No sustained stable regions found for stability analysis']
        }

    total_duration = sum(r['duration'] for r in sustained_regions)
    print("  Analyzing %d sustained regions (%.2fs total)" % (len(sustained_regions), total_duration))

    if total_duration < 0.5:
        return {
            'sustained_regions_found': len(sustained_regions),
            'total_sustained_duration': round(total_duration, 2),
            'sufficient_data': False,
            'jitter_local_percent': None,
            'shimmer_local_percent': None,
            'hnr_db': None,
            'caveats': ['Insufficient sustained audio for reliable metrics (< 0.5s)']
        }

    jitter_values = []
    shimmer_values = []
    hnr_values = []
    durations = []

    for region in sustained_regions:
        if region.get('start_sample') is None or region.get('end_sample') is None:
            continue

        start_sample = region['start_sample']
        end_sample = region['end_sample']
        duration = region['duration']

        if start_sample < 0 or end_sample > len(y) or start_sample >= end_sample:
            continue

        try:
            region_audio = y[start_sample:end_sample]
            region_sound = parselmouth.Sound(region_audio, sampling_frequency=sr)

            point_process = parselmouth.praat.call(
                region_sound, "To PointProcess (periodic, cc)", 75, 600
            )

            jitter = parselmouth.praat.call(
                point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
            )

            shimmer = parselmouth.praat.call(
                [region_sound, point_process], "Get shimmer (local)",
                0, 0, 0.0001, 0.02, 1.3, 1.6
            )

            harmonicity = region_sound.to_harmonicity_cc()
            hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)

            # Filter out NaN values
            jitter_f = float(jitter)
            shimmer_f = float(shimmer)
            hnr_f = float(hnr)

            if not (np.isnan(jitter_f) or np.isnan(shimmer_f) or np.isnan(hnr_f)):
                jitter_values.append(jitter_f)
                shimmer_values.append(shimmer_f)
                hnr_values.append(hnr_f)
                durations.append(duration)

        except Exception:
            continue

    if len(durations) == 0 or sum(durations) < 0.3:
        return {
            'sustained_regions_found': len(sustained_regions),
            'total_sustained_duration': round(total_duration, 2),
            'sufficient_data': False,
            'jitter_local_percent': None,
            'shimmer_local_percent': None,
            'hnr_db': None,
            'caveats': ['Analysis failed on all sustained regions']
        }

    total_weight = sum(durations)
    weights = [d / total_weight for d in durations]

    jitter_avg = sum(j * w for j, w in zip(jitter_values, weights))
    shimmer_avg = sum(s * w for s, w in zip(shimmer_values, weights))
    hnr_avg = sum(h * w for h, w in zip(hnr_values, weights))

    return {
        'sustained_regions_found': len(sustained_regions),
        'regions_analyzed': len(durations),
        'total_sustained_duration': round(total_weight, 2),
        'sufficient_data': True,
        'jitter_local_percent': round(jitter_avg * 100, 4),
        'shimmer_local_percent': round(shimmer_avg * 100, 4),
        'hnr_db': round(hnr_avg, 2),
        'caveats': [
            'Based on %d sustained regions totaling %.2fs' % (len(durations), total_weight),
            'Metrics calculated only on stable note portions (>200ms, <30 cents variation)'
        ]
    }


def aggregate_vibrato_results(vibrato_results):
    """Aggregate per-note vibrato results into summary."""
    valid_results = [v for v in vibrato_results if v.get('valid')]

    if len(valid_results) == 0:
        return {
            'vibrato_detected_anywhere': False,
            'notes_analyzed': len(vibrato_results),
            'notes_with_vibrato': 0,
            'rate_mean_hz': None,
            'rate_std_hz': None,
            'extent_mean_cents': None,
            'extent_std_cents': None,
            'caveat': 'No notes met vibrato criteria'
        }

    rates = [v['rate_hz'] for v in valid_results]
    extents = [v['extent_cents'] for v in valid_results]

    return {
        'vibrato_detected_anywhere': True,
        'notes_analyzed': len(vibrato_results),
        'notes_with_vibrato': len(valid_results),
        'rate_mean_hz': round(float(np.mean(rates)), 2),
        'rate_std_hz': round(float(np.std(rates)), 2) if len(rates) > 1 else 0.0,
        'extent_mean_cents': round(float(np.mean(extents)), 2),
        'extent_std_cents': round(float(np.std(extents)), 2) if len(extents) > 1 else 0.0,
        'regularity_notes': [v.get('regularity_period_cv') for v in valid_results if v.get('regularity_period_cv') is not None],
        'caveat': 'Based on notes meeting vibrato criteria only'
    }


def analyze_vocals(vocals_path, original_path, vocals_were_extracted=False):
    """Run complete vocal analysis pipeline with note-aware analysis."""
    y, sr = load_audio(vocals_path)
    duration = len(y) / sr

    print("\n📊 Analyzing %.1fs audio with note-aware processing..." % duration)

    print("\n🎵 Extracting pitch contour...")
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C5'),  # C5 limits harmonic tracking
        sr=sr,
        frame_length=2048
    )

    # Filter by voiced probability - permissive threshold
    prob_threshold = 0.25
    # Only use voiced probability for note splitting, not as a hard pitch mask
    voiced_flag = voiced_flag & (voiced_probs >= prob_threshold)

    # Optional: Detect noisy intro
    hop_length = 512
    intro_frames = int(2.0 * sr / hop_length)
    if len(voiced_flag) > intro_frames:
        intro_voiced_ratio = np.mean(voiced_flag[:intro_frames])
        if intro_voiced_ratio < 0.3:
            print("   ⚠️  Detected noisy intro, first 2 seconds may be unreliable")

    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    print("\n🎼 Segmenting notes...")
    notes = segment_notes(f0, voiced_flag, times, sr)
    print("   Found %d notes" % len(notes))

    print("\n🎯 Extracting sustained regions...")
    sustained_regions = extract_sustained_regions(notes, sr=sr)
    print("   Found %d sustained regions" % len(sustained_regions))

    print("\n🎻 Analyzing vibrato per note...")
    vibrato_results = [analyze_single_note_vibrato(note) for note in notes]
    valid_vibrato_count = sum(1 for v in vibrato_results if v.get('valid'))
    print("   %d/%d notes have valid vibrato" % (valid_vibrato_count, len(notes)))

    sustained_metrics = calculate_sustained_region_metrics(y, sr, sustained_regions)
    vibrato_summary = aggregate_vibrato_results(vibrato_results)
    spectral_flatness = calculate_spectral_flatness(y, sr)

    suppressed_metrics = []

    if not sustained_metrics.get('sufficient_data'):
        if sustained_metrics.get('caveats'):
            reason = sustained_metrics['caveats'][0]
        else:
            reason = 'No sustained regions found'
        suppressed_metrics.append({
            'metric_name': 'jitter_local_percent',
            'reason': reason
        })
        suppressed_metrics.append({
            'metric_name': 'shimmer_local_percent',
            'reason': reason
        })

    if not vibrato_summary.get('vibrato_detected_anywhere'):
        suppressed_metrics.append({
            'metric_name': 'vibrato_per_note_summary',
            'reason': vibrato_summary.get('caveat', 'No notes met vibrato detection criteria')
        })

    note_summary = []
    for note in notes:
        note_summary.append({
            'start_time': note['start_time'],
            'end_time': note['end_time'],
            'duration': note['duration'],
            'mean_pitch_hz': note['mean_pitch_hz'],
            'type': note['type']
        })

    caveats = [
        "This analysis uses note-aware processing.",
        "Jitter and shimmer are calculated only on sustained note regions.",
        "Vibrato is analyzed per-note and aggregated.",
        "These metrics describe acoustic characteristics, not vocal health or pathology.",
        "Results depend on pitch tracking accuracy and note segmentation."
    ]

    results = {
        'analysis_metadata': {
            'original_file': original_path,
            'vocals_file': vocals_path,
            'vocals_extracted': vocals_were_extracted,
            'demucs_model': DEMUCS_MODEL if vocals_were_extracted else None,
            'sample_rate': sr,
            'duration_seconds': round(duration, 2),
            'total_samples': len(y),
            'analysis_version': '2.0-note-aware'
        },
        'note_segmentation': {
            'total_notes_detected': len(notes),
            'sustained_notes': sum(1 for n in notes if n['type'] == 'sustained'),
            'ornamented_notes': sum(1 for n in notes if n['type'] == 'ornamented'),
            'notes': note_summary
        },
        'sustained_note_metrics': sustained_metrics,
        'vibrato_summary': vibrato_summary,
        'spectral_flatness': spectral_flatness,
        'suppressed_metrics': suppressed_metrics,
        'caveats': caveats,
        'raw_data': {
            'audio_waveform_shape': y.shape,
            'audio_waveform_mean': round(float(np.mean(y)), 6),
            'audio_waveform_std': round(float(np.std(y)), 6),
            'audio_waveform_max': round(float(np.max(np.abs(y))), 6)
        }
    }

    return results


def print_final_report(results):
    """Print a comprehensive final report."""
    print("\n" + "=" * 70)
    print("                    VOX AI - VOCAL ANALYSIS REPORT")
    print("=" * 70)

    meta = results['analysis_metadata']
    vocals_status = "Yes (demucs)" if meta.get('vocals_extracted') else "No (original audio)"
    print("\n📁 File: %s" % meta['original_file'])
    print("   Vocals Extracted: %s" % vocals_status)
    print("   Duration: %.1fs | Sample Rate: %dHz" % (meta['duration_seconds'], meta['sample_rate']))
    print("   Version: %s" % meta.get('analysis_version', '1.0'))

    notes = results['note_segmentation']
    print("\n🎼 NOTES DETECTED: %d" % notes['total_notes_detected'])
    print("   Sustained: %d | Ornamented: %d" % (notes['sustained_notes'], notes['ornamented_notes']))

    sustained = results['sustained_note_metrics']
    print("\n🎯 SUSTAINED NOTE STABILITY")
    print("   Regions: %d (%.2fs total)" % (sustained['sustained_regions_found'], sustained['total_sustained_duration']))

    if sustained.get('sufficient_data'):
        print("   Jitter: %.4f%%" % sustained['jitter_local_percent'])
        print("   Shimmer: %.4f%%" % sustained['shimmer_local_percent'])
        print("   HNR: %.2f dB" % sustained['hnr_db'])
    else:
        print("   ⚠️  %s" % sustained['caveats'][0])

    vib = results['vibrato_summary']
    print("\n🎻 VIBRATO ANALYSIS")
    if vib.get('vibrato_detected_anywhere'):
        print("   Detected on %d/%d notes" % (vib['notes_with_vibrato'], vib['notes_analyzed']))
        if vib.get('rate_mean_hz') is not None:
            print("   Rate: %.2f ± %.2f Hz" % (vib['rate_mean_hz'], vib.get('rate_std_hz', 0)))
        if vib.get('extent_mean_cents') is not None:
            print("   Extent: %.2f ± %.2f cents" % (vib['extent_mean_cents'], vib.get('extent_std_cents', 0)))
    else:
        print("   No vibrato detected on qualifying notes")

    flat = results['spectral_flatness']
    print("\n📊 SPECTRAL FLATNESS")
    print("   Mean: %.4f" % flat['spectral_flatness_mean'])
    print("   %s" % flat['interpretation'])

    if results.get('suppressed_metrics'):
        print("\n🚫 SUPPRESSED METRICS (%d)" % len(results['suppressed_metrics']))
        for sm in results['suppressed_metrics']:
            print("   • %s: %s" % (sm['metric_name'], sm['reason']))

    print("\n" + "=" * 70)
    print("⚠️  IMPORTANT CAVEATS")
    print("=" * 70)
    for caveat in results['caveats']:
        print("   • %s" % caveat)
    print("=" * 70)


def save_results(results: Dict, output_file: str = 'vocal_results.json'):
    """Save complete results to JSON."""
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Complete results saved to: {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('--help', '-h'):
        print("""VOX AI - Vocal Acoustic Analysis Tool

Usage: python analyze_vocal.py <audio_file>

This tool performs:
1. Source separation (demucs htdemucs) to isolate vocals
2. Vocal acoustic analysis:
   - HNR: Signal periodicity
   - Spectral Flatness: Timbre characteristics
   - Jitter/Shimmer: Pitch/amplitude stability (on sustained notes)
   - Vibrato: Rate and extent (per-note analysis)
3. Outputs metrics with appropriate caveats and limitations

IMPORTANT: These metrics describe acoustic characteristics of the
performance, not vocal health, pathology, or technique quality.

Output: vocal_results.json with all metrics and caveats
""")
        sys.exit(0)

    input_path = sys.argv[1]

    if not os.path.exists(input_path):
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)

    print("=" * 70)
    print("     VOX AI - VOCAL ANALYSIS")
    print("=" * 70)

    # Step 1: Source Separation (optional)
    vocals_path = separate_vocals(input_path)
    vocals_were_extracted = vocals_path is not None

    # Step 2: Analyze the separated vocals (or original if separation failed)
    temp_files_to_cleanup = []
    if vocals_path is None:
        print("\n⚠️  Source separation unavailable. Analyzing original audio.")
        print("   (Results may be affected by instrumental backing)\n")

        # Check if we need to convert the input file for librosa
        ext = Path(input_path).suffix.lower()
        if ext in ['.m4a', '.aac', '.mp4']:
            print(f"Converting {ext} to WAV for analysis...")
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
                convert_to_wav(input_path, tmp_wav.name)
                vocals_path = tmp_wav.name
                temp_files_to_cleanup.append(tmp_wav.name)
        else:
            vocals_path = input_path

    results = analyze_vocals(vocals_path, input_path, vocals_were_extracted=vocals_were_extracted)

    # Cleanup temp files
    for tmp_file in temp_files_to_cleanup:
        try:
            os.unlink(tmp_file)
        except:
            pass

    # Step 3: Print final report
    print_final_report(results)

    # Step 4: Save to JSON
    save_results(results)

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
