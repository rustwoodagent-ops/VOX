#!/usr/bin/env python3
"""
VOX AI - Audio File Info
Loads an audio file and prints its duration and sample rate using librosa.
"""

import sys
import librosa


def get_audio_info(file_path):
    """Load an audio file and return its duration and sample rate."""
    try:
        # Load the audio file
        y, sr = librosa.load(file_path, sr=None)  # sr=None preserves original sample rate

        # Get duration
        duration = librosa.get_duration(y=y, sr=sr)

        return {
            'file_path': file_path,
            'sample_rate': sr,
            'duration': duration,
            'duration_formatted': f"{int(duration // 60)}:{int(duration % 60):02d}"
        }
    except Exception as e:
        print(f"Error loading audio file: {e}")
        sys.exit(1)


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('--help', '-h'):
        print("Usage: python audio_info.py <audio_file>")
        print("Example: python audio_info.py sample.mp3")
        print("\nPrints audio file duration and sample rate using librosa.")
        sys.exit(0 if sys.argv[1] in ('--help', '-h') else 1)

    file_path = sys.argv[1]
    info = get_audio_info(file_path)

    print(f"\n{'='*50}")
    print(f"Audio File: {info['file_path']}")
    print(f"{'='*50}")
    print(f"Sample Rate: {info['sample_rate']} Hz")
    print(f"Duration: {info['duration']:.2f} seconds ({info['duration_formatted']})")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
