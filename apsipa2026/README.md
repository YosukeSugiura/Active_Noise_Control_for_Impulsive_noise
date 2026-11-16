# APSIPA ASC 2026

Code accompanying Yosuke Sugiura, “Coherence-Based Variable Step-Size FD-FxNLMS for Active Impulsive Noise Control,” APSIPA ASC 2026.

- `algorithms_20260815.py` contains the coherence-based FD-FxNLMS class extracted from the August 15 implementation snapshot.
- `make_anc_figure.py` uses `baseline_inputs.npz` to generate the 30 ms illustrative waveform and spectrum comparison. The archived input was obtained from a seeded simulation with symmetric alpha-stable noise (alpha = 1.45). The comparison amplifies one impulse and inserts a second equal-amplitude impulse of the same sign.

From the repository root, run:

```sh
python -m pip install -r apsipa2026/requirements.txt
python apsipa2026/make_anc_figure.py
```

The script saves PDF and PNG files in `apsipa2026/figures/`. The script records the settings for this demonstration. The exact paper experiment configuration, the settings for all compared algorithms, and reproduction of the reported ANR and RTF remain unverified.
