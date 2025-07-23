"""
Compare different processing outputs to identify the best settings.
"""

import numpy as np
import librosa
import soundfile as sf


def analyze_file(file_path, label):
    """Analyze a single audio file."""
    try:
        audio, sr = librosa.load(file_path, sr=None, mono=True)
    except:
        print(f"Could not load {file_path}")
        return None
    
    duration = len(audio) / sr
    rms = np.sqrt(np.mean(audio**2))
    peak = np.max(np.abs(audio))
    
    # Calculate dB values
    rms_db = 20 * np.log10(rms + 1e-10)
    peak_db = 20 * np.log10(peak + 1e-10)
    
    # Calculate dynamic range
    audio_db = 20 * np.log10(np.abs(audio) + 1e-10)
    dynamic_range = np.max(audio_db) - np.min(audio_db)
    
    # Calculate percentiles for volume distribution
    percentiles = [10, 25, 50, 75, 90, 95]
    volume_percentiles = np.percentile(audio_db, percentiles)
    
    return {
        'label': label,
        'file': file_path,
        'duration': duration,
        'rms': rms,
        'peak': peak,
        'rms_db': rms_db,
        'peak_db': peak_db,
        'dynamic_range': dynamic_range,
        'percentiles': dict(zip(percentiles, volume_percentiles))
    }


def compare_all_outputs():
    """Compare original, mild processing, and aggressive processing."""
    files = [
        ("input.wav", "Original"),
        ("output_normalized.wav", "Mild Processing (-18dB, 3:1)"),
        ("output_aggressive.wav", "Aggressive Processing (-30dB, 8:1)")
    ]
    
    results = []
    for file_path, label in files:
        result = analyze_file(file_path, label)
        if result:
            results.append(result)
    
    if len(results) < 2:
        print("Not enough files to compare")
        return
    
    print("=== AUDIO PROCESSING COMPARISON ===\n")
    
    # Print header
    print(f"{'Metric':<20} {'Original':<15} {'Mild':<15} {'Aggressive':<15} {'Mild Change':<12} {'Aggr Change':<12}")
    print("-" * 100)
    
    original = results[0]
    
    for i, result in enumerate(results):
        if i == 0:
            continue  # Skip original for change calculation
        
        # Calculate changes from original
        rms_change = (result['rms'] / original['rms'] - 1) * 100
        peak_change = (result['peak'] / original['peak'] - 1) * 100
        dr_change = (result['dynamic_range'] / original['dynamic_range'] - 1) * 100
        
        if i == 1:  # Mild processing
            mild_rms = f"{rms_change:+6.1f}%"
            mild_peak = f"{peak_change:+6.1f}%"
            mild_dr = f"{dr_change:+6.1f}%"
            aggr_rms = aggr_peak = aggr_dr = ""
        else:  # Aggressive processing
            aggr_rms = f"{rms_change:+6.1f}%"
            aggr_peak = f"{peak_change:+6.1f}%"
            aggr_dr = f"{dr_change:+6.1f}%"
            mild_rms = mild_peak = mild_dr = ""
    
    # Print comparison table
    metrics = [
        ('RMS Level', 'rms', 6),
        ('Peak Level', 'peak', 6),
        ('RMS (dB)', 'rms_db', 2),
        ('Peak (dB)', 'peak_db', 2),
        ('Dynamic Range (dB)', 'dynamic_range', 1)
    ]
    
    for metric_name, key, decimals in metrics:
        values = [f"{r[key]:.{decimals}f}" for r in results]
        
        # Calculate changes
        if len(results) >= 2:
            mild_change = (results[1][key] / results[0][key] - 1) * 100
            mild_str = f"{mild_change:+6.1f}%"
        else:
            mild_str = ""
        
        if len(results) >= 3:
            aggr_change = (results[2][key] / results[0][key] - 1) * 100
            aggr_str = f"{aggr_change:+6.1f}%"
        else:
            aggr_str = ""
        
        # Pad values to fit columns
        orig_val = values[0] if len(values) > 0 else ""
        mild_val = values[1] if len(values) > 1 else ""
        aggr_val = values[2] if len(values) > 2 else ""
        
        print(f"{metric_name:<20} {orig_val:<15} {mild_val:<15} {aggr_val:<15} {mild_str:<12} {aggr_str:<12}")
    
    print("\n=== VOLUME DISTRIBUTION COMPARISON ===")
    print(f"{'Percentile':<12} {'Original':<12} {'Mild':<12} {'Aggressive':<12}")
    print("-" * 50)
    
    for percentile in [10, 25, 50, 75, 90, 95]:
        values = []
        for result in results:
            if percentile in result['percentiles']:
                values.append(f"{result['percentiles'][percentile]:.1f}")
            else:
                values.append("")
        
        orig_val = values[0] if len(values) > 0 else ""
        mild_val = values[1] if len(values) > 1 else ""
        aggr_val = values[2] if len(values) > 2 else ""
        
        print(f"{percentile}th:{'':<7} {orig_val:<12} {mild_val:<12} {aggr_val:<12}")
    
    print("\n=== RECOMMENDATIONS ===")
    
    if len(results) >= 3:
        orig_dr = results[0]['dynamic_range']
        mild_dr = results[1]['dynamic_range']
        aggr_dr = results[2]['dynamic_range']
        
        mild_dr_reduction = orig_dr - mild_dr
        aggr_dr_reduction = orig_dr - aggr_dr
        
        print(f"Dynamic Range Reduction:")
        print(f"  Mild processing:       {mild_dr_reduction:.1f} dB")
        print(f"  Aggressive processing: {aggr_dr_reduction:.1f} dB")
        
        if mild_dr_reduction < 5:
            print("  -> Mild processing has minimal effect")
        if aggr_dr_reduction > 10:
            print("  -> Aggressive processing shows significant compression")
        
        # RMS level analysis
        orig_rms_db = results[0]['rms_db']
        mild_rms_db = results[1]['rms_db']
        aggr_rms_db = results[2]['rms_db']
        
        print(f"\nLoudness Changes:")
        print(f"  Mild processing:       {mild_rms_db - orig_rms_db:+.1f} dB")
        print(f"  Aggressive processing: {aggr_rms_db - orig_rms_db:+.1f} dB")
        
        if abs(mild_rms_db - orig_rms_db) < 3:
            print("  -> Mild processing: barely audible change")
        if abs(aggr_rms_db - orig_rms_db) > 6:
            print("  -> Aggressive processing: very noticeable change")


if __name__ == "__main__":
    compare_all_outputs()
