#!/usr/bin/env python3
"""
Speech Volume Normalizer - Main CLI Interface
Command-line tool for robust speech volume normalization using VAD-gated compression.
"""

import argparse
import sys
import os
from pathlib import Path

from speech_normalizer import SpeechVolumeNormalizer


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Robust Speech Volume Normalizer using VAD-gated compression",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("input_file", 
                       help="Input audio file path")
    parser.add_argument("output_file", 
                       help="Output audio file path")
    
    # VAD parameters
    vad_group = parser.add_argument_group("Voice Activity Detection (VAD)")
    vad_group.add_argument("--vad-mode", type=int, default=2, choices=[0, 1, 2, 3],
                          help="VAD aggressiveness mode (0=least aggressive, 3=most aggressive)")
    vad_group.add_argument("--vad-frame-duration", type=int, default=20, 
                          choices=[10, 20, 30],
                          help="VAD frame duration in milliseconds")
    vad_group.add_argument("--vad-hangover", type=int, default=200,
                          help="VAD hangover time in milliseconds")
    
    # DRC parameters
    drc_group = parser.add_argument_group("Dynamic Range Compression (DRC)")
    drc_group.add_argument("--threshold", type=float, default=-25.0,
                          help="Compression threshold in dB (lower = more compression)")
    drc_group.add_argument("--ratio", type=float, default=6.0,
                          help="Compression ratio (e.g., 6.0 for 6:1, higher = more compression)")
    drc_group.add_argument("--attack", type=float, default=5.0,
                          help="Attack time in milliseconds (lower = faster response)")
    drc_group.add_argument("--release", type=float, default=75.0,
                          help="Release time in milliseconds")
    
    # Processing parameters
    proc_group = parser.add_argument_group("Processing")
    proc_group.add_argument("--crossfade", type=float, default=10.0,
                           help="Crossfade duration at segment boundaries in milliseconds")
    proc_group.add_argument("--block-size", type=int, default=4096,
                           help="Audio processing block size")
    
    # Analysis mode
    parser.add_argument("--analyze-only", action="store_true",
                       help="Only analyze the input file without processing")
    parser.add_argument("--no-progress", action="store_true",
                       help="Disable progress bar")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = Path(args.output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize normalizer with parameters
    normalizer = SpeechVolumeNormalizer(
        # VAD parameters
        vad_aggressiveness=args.vad_mode,
        vad_frame_duration_ms=args.vad_frame_duration,
        vad_hangover_ms=args.vad_hangover,
        # DRC parameters
        threshold_db=args.threshold,
        ratio=args.ratio,
        attack_ms=args.attack,
        release_ms=args.release,
        # Processing parameters
        crossfade_ms=args.crossfade,
        block_size=args.block_size
    )
    
    try:
        if args.analyze_only:
            # Analysis mode
            print("=== Audio Analysis Mode ===")
            print(f"Analyzing: {args.input_file}")
            print()
            
            analysis = normalizer.analyze_audio(args.input_file)
            
            print("=== Analysis Results ===")
            print(f"Total duration: {analysis['total_duration']:.2f} seconds")
            print(f"Speech duration: {analysis['speech_duration']:.2f} seconds")
            print(f"Speech ratio: {analysis['speech_ratio']:.1%}")
            print(f"Number of speech segments: {analysis['segments_count']}")
            print()
            print(f"Overall RMS level: {analysis['overall_rms']:.6f}")
            print(f"Overall peak level: {analysis['overall_peak']:.6f}")
            print(f"Speech RMS level: {analysis['speech_rms']:.6f}")
            print(f"Speech peak level: {analysis['speech_peak']:.6f}")
            print()
            
            if analysis['segments']:
                print("Speech segments:")
                for i, (start, end) in enumerate(analysis['segments'][:10]):  # Show first 10
                    duration = end - start
                    print(f"  {i+1:2d}: {start:7.2f}s - {end:7.2f}s (duration: {duration:5.2f}s)")
                
                if len(analysis['segments']) > 10:
                    print(f"  ... and {len(analysis['segments']) - 10} more segments")
            
        else:
            # Processing mode
            print("=== Speech Volume Normalizer ===")
            print(f"Input file: {args.input_file}")
            print(f"Output file: {args.output_file}")
            print()
            print("Parameters:")
            print(f"  VAD mode: {args.vad_mode} (aggressiveness)")
            print(f"  VAD frame duration: {args.vad_frame_duration}ms")
            print(f"  VAD hangover: {args.vad_hangover}ms")
            print(f"  Compression threshold: {args.threshold}dB")
            print(f"  Compression ratio: {args.ratio}:1")
            print(f"  Attack time: {args.attack}ms")
            print(f"  Release time: {args.release}ms")
            print(f"  Crossfade duration: {args.crossfade}ms")
            print()
            
            # Process audio
            stats = normalizer.process_audio_file(
                args.input_file, 
                args.output_file,
                show_progress=not args.no_progress
            )
            
            print()
            print("=== Processing Summary ===")
            print(f"Total duration: {stats['total_duration']:.2f}s")
            print(f"Speech duration: {stats['speech_duration']:.2f}s ({stats['speech_ratio']:.1%})")
            print(f"Speech segments processed: {stats['segments_count']}")
            print(f"RMS level change: {stats['rms_ratio']:.3f}x")
            print(f"Output saved to: {args.output_file}")
            
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
