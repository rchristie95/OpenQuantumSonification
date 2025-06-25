# OpenQuantumSonification

OpenQuantumSonification generates audiovisual representations of model quantum systems. The repository contains two standalone Python scripts used in the "Sound of Decoherence" project. Each script performs a full simulation and then produces an MP4 video with embedded audio.

## Requirements

* Python 3 with the packages `numpy`, `scipy`, `matplotlib` and `soundfile`
* [FFmpeg](https://ffmpeg.org/) must be installed and available on the command line

Install the Python dependencies with pip if they are not already present:

```bash
pip install numpy scipy matplotlib soundfile
```

## Usage

Run one of the scripts directly with Python. The calculations may take several minutes depending on your hardware and the chosen parameters.

### Quartic Oscillator

```
python SonificationEigenBasis.py
```

This script simulates a quartic oscillator using a basis of harmonic-oscillator eigenstates. It writes out frames for a Wigner-function animation, synthesizes audio from the system's evolution and finally calls FFmpeg to combine them into a video (`SonificationEigenBasis.mp4`).

### Spin Helix Chain

```
python SpinHelix.py
```

`SpinHelix.py` models a short XXZ spin chain driven to a spin-helix state via Lindblad operators. It produces a magnetization animation and stereo audio that are merged into `SpinHelix.mp4`.

### Output

Both scripts create temporary WAV files and PNG frames before producing the final MP4 video with lossless ALAC audio. The intermediate files are automatically removed once FFmpeg has produced the final video.

## Customisation

The simulation parameters (e.g. `L`, `Delta`, `Gamma` in *SpinHelix.py* or `bSize`, `Gamma`, `kT` in *SonificationEigenBasis.py*) are defined near the top of each script. Edit these values if you wish to explore different regimes. Larger basis sizes or spin chains will require more memory and CPU time.

## Notes

The code is intentionally written as simple scripts rather than as importable modules. Read through the functions at the top of each file to learn how states are propagated and how the audio is generated.
