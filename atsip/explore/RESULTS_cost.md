# Per-block cost: adaptation and delayless filtering

Script: `atsip/explore/cost.py`. Output: `atsip/explore/out/cost_bench.json`.

Setting: M = 512, N = 2M = 1024, B = 128, fs = 16 kHz, so one block period is B/fs = 8000 us.
Environment: Python 3.13.18, NumPy 2.5.3, SciPy 1.18.1 (`scipy.fft`, pocketfft, workers = 1),
Intel Xeon @ 2.10 GHz (2 vCPUs visible), OMP / OPENBLAS / MKL threads = 1, numba not used in the timed loops.
Per (method, mode): 300 warm-up blocks, 4000 timed blocks, 3 rounds. Table values are medians over the rounds,
with the min-max range over the rounds in brackets. RTF = median / 8000 us.

## Measured per-block time (us)

| method | update: median (p90) | update: RTF | update+filter: median (p90) | update+filter: RTF |
|---|---|---|---|---|
| FD0 (plain, a = 0) | 38.3 (49.7) [38.3-40.7] | 0.0048 | 85.4 (107.7) [81.9-86.9] | 0.0107 |
| B0 (binwise, a = 1) | 37.2 (46.4) [36.2-37.9] | 0.0046 | 83.4 (109.0) [82.8-87.0] | 0.0104 |
| W0.5 (geometric a = 0.5) | 40.2 (47.3) [38.3-41.6] | 0.0050 | 84.8 (105.6) [82.5-84.9] | 0.0106 |
| GS (geometric + ramp) | 39.9 (50.0) [38.6-40.2] | 0.0050 | 86.6 (123.4) [84.7-88.4] | 0.0108 |
| SF (factor every block) | 94.6 (122.8) [90.0-95.5] | 0.0118 | 141.0 (177.0) [140.9-141.3] | 0.0176 |
| SF8 (factor every 8 blocks) | 58.9 (94.7) [58.4-61.7] | 0.0074 | 108.3 (151.1) [106.9-109.1] | 0.0135 |
| APSIPA (Freq_FxNLMS_CohVSS) | 116.1 (178.9) [110.4-116.4] | 0.0145 | 161.6 (198.9) [153.3-162.9] | 0.0202 |
| filter only (512 MAC/sample) | 42.6 (46.2) [38.2-43.3] | 0.0053 | - | - |

Ratios to FD0 (medians): update-only B0 0.97, W0.5 1.05, GS 1.04, SF 2.47, SF8 1.54, APSIPA 3.03.
Update+filter: B0 0.98, W0.5 0.99, GS 1.01, SF 1.65, SF8 1.27, APSIPA 1.89.

## Theoretical operation count per block (update only; filter is common to all)

K = N/2 + 1 = 513 bins. "FFT" counts N-point rfft/irfft calls.

| method | N-point FFT/IFFT | complex mul/div (K bins) | real log/exp/sqrt/pow (K-vectors) | extra cepstral FFTs | predicted FFT ratio to FD0 | measured ratio (update) |
|---|---|---|---|---|---|---|
| FD0 | 3 (rfft x, rfft e, irfft g) | 2 | none | 0 | 1.00 | 1.00 |
| B0 | 3 | 2 (+ EMA, divide) | none | 0 | 1.00 | 0.97 |
| W0.5 | 3 | 2 | sqrt x1 | 0 | 1.00 | 1.05 |
| GS | 3 | 2 (+ multiply) | pow x1 (float exponent) | 0 | 1.00 | 1.04 |
| SF | 7 (2 common, 3 SF operator, 2 cepstrum: irfft(log), rfft(c)) | 4 (Hc*X, conj(R)E, divide by conj F, divide by F) | log x1, exp x1 | 2 | 2.33 | 2.47 |
| SF8 | 5.25 average | 4 | log, exp at 1/8 rate | 0.25 average | 1.75 | 1.54 |
| APSIPA | 3 | 3 | pow x3, exp x1 | 0 | 1.00 | 3.03 |
| filter (common) | 0 | 0 | none | 0 | - | 65,536 real MAC per block (512 per sample) |

Consistency check: for SF the measured update ratio (2.47) is close to the FFT ratio (2.33), and the extra log/exp and complex divisions account for the remainder. SF8 measures below its FFT ratio (1.54 against 1.75), consistent with a fixed per-block overhead diluting the ratio. APSIPA has the same FFT count as FD0 but about 3x the time, so its extra cost is in its dozens of elementwise NumPy operations (EMAs, clips, where) and four K-vector transcendental calls, not in FFTs.

## Three-line conclusion

1. The spectral factor every block costs about 2.5 times FD0 in the update (94.6 against 38.3 us) and about 1.65 times with filtering, and this follows the extra FFTs and log/exp calls in the count table.
2. Recomputing the factor every 8 blocks recovers about 60 percent of that gap (58.9 us update-only, against 94.6 and 38.3), while B0, W0.5 and GS are within about 5 percent of FD0, so the sqrt and pow on 513 bins are not measurable here; APSIPA is the most expensive at about 3 times FD0 in the update.
3. Every method stays below RTF 0.021 (update+filter, single thread, 2-vCPU Xeon), so real-time margin is ample on this machine, but these are NumPy-overhead-dominated timings at N = 1024 and not an embedded-target estimate.

## Notes on the benchmark

- Update-only mode feeds a stored error stream with no feedback, so its weights are not meaningful; only its timing is used.
- Update+filter mode runs the closed loop: the primary path (422-tap IR) drives d, the error is e = d + s*y with the repo's sign convention, and y is the 512-tap delayless filter output. The secondary-path plant that forms e is computed outside the timed region, so the timed work is the filter plus the FFT update.
- GS ramp assumed: a(b) = min(1, 0.5 + 0.5 b / 1000), with a power computed each block. APSIPA uses the default parameters of `_FreqFxNLMS_CohVSS` in `anc-fast/fastsim/kernels_fd.py` (the same NumPy operation set as `Freq_FxNLMS_CohVSS.update` in `algorithms/algorithms.py`; its numba routines are not used here). The timing does not depend on parameter values.
- The anc-fast kernel was reproduced in this script rather than imported, to keep one code style and avoid numba compilation side effects.
