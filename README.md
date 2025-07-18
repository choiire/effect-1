# Proximity Effect Reducer

This project provides a Python script to reduce the proximity effect in audio recordings from directional microphones. The script processes a recorded audio file, dynamically applying a low-shelf filter to attenuate excessive bass frequencies caused by the proximity effect.

## How it Works

The script implements the following steps:

1.  **Load Audio**: Reads a `.wav` audio file.
2.  **STFT Analysis**: Analyzes the audio in short frames to determine the energy in the low-frequency band (e.g., 100-300 Hz).
3.  **Dynamic Gain Control**: Calculates a gain reduction value based on the low-frequency energy. When the energy exceeds a threshold, the gain is reduced. This process is smoothed using attack and release times.
4.  **Biquad Filtering**: Applies a low-shelf Biquad filter to the audio, using the dynamically calculated gain to control the amount of bass reduction.
5.  **Save Output**: Saves the processed audio to a new `.wav` file.

## Prerequisites

- Python 3.x

## Installation

1.  Clone or download this repository.
2.  Install the required Python libraries:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Place your input audio file (e.g., `input.wav`) in the project directory.
2.  Run the script from your terminal:

    ```bash
    python proximity_effect_reducer.py input.wav output.wav
    ```

    - `input.wav`: The name of the original audio file.
    - `output.wav`: The name for the processed audio file.

3.  The script will create `output.wav` with the proximity effect reduced.
#   e f f e c t - 1  
 