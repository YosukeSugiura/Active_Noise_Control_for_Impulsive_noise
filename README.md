# Feedforward Active Noise Control for Impulsive Noise
This repository provides a **Feedforward Active Noise Control (ANC)** simulator implemented in **pure Python + NumPy**, accelerated with **Numba**. Multiple robust FxLMS-family algorithms can be enabled via `config.json`. Figures and logs are automatically saved under `logs/`.

## Requirements
- Python ≥ 3.10
- See `requirements.txt`:
  - `numpy`
  - `numba`
  - `matplotlib`
  - `tqdm`

Install:
```bash
pip install -r requirements.txt
```

## Directory Structure
```
.
├─ algorithms/
│   └─ algorithms.py
├─ impulse_response/
│   ├─ primary_ir.dat
│   └─ secondary_ir.dat
├─ sounds/
│   ├─ impulsive_noise_alpha1.45.dat
│   ├─ impulsive_noise_alpha1.65.dat
│   └─ ...
├─ anc.py
├─ config.json
├─ logs/
└─ ffanc.png (optional diagram)
```

## Usage
1. Edit `config.json` to enable/disable algorithms.
2. Specify impulse response paths and noise file.
3. Run:
```bash
python anc.py --config config.json
```
4. Output will be stored in `logs/`.

## Config Notes
- Paths in `config.json` can be **relative** to the config location.
- `amp_scales` enables amplitude invariance tests.
- `order_control` sets control filter length.
- `order_secondary` sets filtered‑reference FIR length.

## Output Files
- `anr.csv` — ANR history per algorithm
- `ratio_amp_invariance.csv` — steady‑state ratio stats (if available)
- `anr.png` — ANR over time
- Timing summary printed to console

## Terminal Summary Example
```
=== Summary (last trial) ===
FxNLMS            ANR_mean[dB]= -23.386  update_mean=   5.514 µs
BlockFxNLMS       ANR_mean[dB]= -23.266  update_mean=   6.595 µs
```

## Citation
Please cite if you use this in academic work:

A. Haneda, Y. Sugiura, and T. Shimamura, “FxlogLMS+: Modified FxlogLMS Algorithm for Active Impulsive Noise Control,”
*Lecture Notes in Electrical Engineering*, vol. 1322, Springer, 2025, pp. 342–351.

## Author
Yosuke Sugiura (Saitama University, Japan)
