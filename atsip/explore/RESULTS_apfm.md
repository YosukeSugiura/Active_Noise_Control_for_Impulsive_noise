# APFM and VSS-NFBFxLMS evaluation

## Adaptation of the reference

- Single reference, actuator and error microphone, with Q = 1. Each path pair uses its original primary path for 30 s of white-noise pretraining (seed 900 + pair number). The secondary plant is the full S and the model is its first 512 taps. Control starts at zero.
- Parallel.m supplies the candidate ordering and minimum-energy selection. Counterfactual errors use the measured error minus the modeled response to the actual output, plus the modeled response with the candidate held fixed over the relevant history. This replaces the MATLAB simulation's direct access to the primary disturbance and candidate-specific output histories. Ties use the first minimum.
- Equations (10)–(15) supply the gradient, normalization, variable step and zero-selection reset. Parallel.m leaves the gradient unimplemented and uses a different step formula and reset multiplier. Here the step uses the measured current-block error power, and the gradient uses the selected reconstructed error. The zero selection uses the decayed previous increment and resets the three averages.
- The constrained overlap-save implementation uses 512 causal taps, a 128-sample hop and a 1024-point FFT. Selection scores only the newest 128 samples. The error is zero-padded on the left, and the inverse-FFT gradient is projected onto the first 512 taps. The original code uses a block length equal to filter length and a 2B FFT.
- The three EMA factors are 0.95, μ_max = 2 and η = 10⁻⁶. Λ starts at one, and the two block-energy EMAs start at zero. A bank run that diverges retains its last finite deployed coefficients, with the divergence logged in the bank artifact.
- References use the saved evaluation and frozen coefficients where available. The short-window empirical power ratios are computed from fresh traces under the same frozen settings. FxNLMS-family frozen coefficient values absent from the saved archive are shown as unavailable.

## Tuning

Selected μ̃ = 4 (2^2). Each setting covers 90 design cases. Smaller L1 means more attenuation.

| μ̃ | Diverged | L1 whole (dB) | Far KM (s) | Censored |
|---:|---:|---:|---:|---:|
| 0.00390625 | 0 | -0.023 | 0.120 | 7 |
| 0.0078125 | 0 | -0.046 | 0.120 | 7 |
| 0.015625 | 0 | -0.090 | 0.120 | 7 |
| 0.03125 | 0 | -0.178 | 0.120 | 7 |
| 0.0625 | 0 | -0.347 | 0.120 | 7 |
| 0.125 | 0 | -0.662 | 0.120 | 7 |
| 0.25 | 0 | -1.216 | 0.440 | 7 |
| 0.5 | 0 | -2.100 | 1.112 | 7 |
| 1 | 0 | -3.286 | 2.256 | 9 |
| 2 | 0 | -4.524 | 2.736 | 12 |
| 4 | 0 | -5.602 | 4.904 | 15 |
| 8 | 1 | -6.405 | 4.536 | 12 |

Selected bank training divergences: 0 / 8.
Recovery counts use all 200 cases per condition. L1 is in dB (lower is better). The power ratio is cumulative from the switch. Frozen values are medians.

## Final evaluation

| Condition | Method | Diverged | KM s (censored) | ≤0.5 s | ≤2 s | L1 whole | Last 10 s | Event +2 s | Power +0.5 s | +1 s | +2 s | Frozen 0.13 s | Frozen 0.51 s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stat | P5-early-cap | 0 | — (—) | — | — | -9.031 | -9.111 | — | — | — | — | — | — |
| stat | nors-cap | 0 | — (—) | — | — | -9.031 | -9.111 | — | — | — | — | — | — |
| stat | Freq_FxNLMS_CohVSS | 0 | — (—) | — | — | -8.978 | -9.184 | — | — | — | — | — | — |
| stat | APFM | 0 | — (—) | — | — | -7.008 | -7.020 | — | — | — | — | — | — |
| stat | VSS-NFBFxLMS | 0 | — (—) | — | — | -3.338 | -4.519 | — | — | — | — | — | — |
| RS-near | P5-early-cap | 0 | 0.360 (28) | 112 | 151 | -8.925 | -9.198 | -7.155 | -2.918 | -4.673 | -6.145 | -2.587 | -7.886 |
| RS-near | nors-cap | 0 | 1.048 (28) | 43 | 141 | -8.770 | -9.184 | -5.079 | -0.258 | -1.934 | -3.810 | 1.139 | -3.258 |
| RS-near | Freq_FxNLMS_CohVSS | 0 | 0.976 (28) | 45 | 145 | -8.686 | -9.173 | -5.473 | -1.025 | -2.742 | -4.514 | -0.032 | -5.008 |
| RS-near | APFM | 0 | 1.200 (37) | 72 | 102 | -4.093 | -1.518 | -0.180 | 0.284 | 0.160 | -0.085 | -0.016 | -0.275 |
| RS-near | VSS-NFBFxLMS | 0 | 5.400 (62) | 50 | 79 | -2.015 | -2.325 | 0.310 | 0.737 | 0.645 | 0.453 | 1.036 | 0.716 |
| RS-far | P5-early-cap | 0 | 0.400 (28) | 107 | 147 | -10.076 | -11.534 | -9.244 | -4.468 | -6.452 | -8.105 | -6.748 | -10.865 |
| RS-far | nors-cap | 0 | 1.128 (29) | 17 | 135 | -9.873 | -11.513 | -6.549 | -1.020 | -2.891 | -5.018 | 0.215 | -4.471 |
| RS-far | Freq_FxNLMS_CohVSS | 0 | 1.256 (28) | 40 | 137 | -9.701 | -11.347 | -6.864 | -2.003 | -3.978 | -5.938 | -0.881 | -6.838 |
| RS-far | APFM | 1 | 2.896 (58) | 61 | 93 | -4.407 | -2.354 | -0.383 | -0.035 | -0.127 | -0.375 | -0.073 | -0.282 |
| RS-far | VSS-NFBFxLMS | 0 | 9.096 (80) | 41 | 69 | -2.159 | -2.616 | 0.005 | 0.347 | 0.245 | 0.059 | 0.477 | 0.259 |
| EN0 | P5-early-cap | 0 | — (—) | — | — | -8.492 | -8.150 | -5.600 | — | — | — | — | — |
| EN0 | nors-cap | 0 | — (—) | — | — | -8.465 | -8.072 | -5.590 | — | — | — | — | — |
| EN0 | Freq_FxNLMS_CohVSS | 0 | — (—) | — | — | -8.544 | -8.396 | -6.222 | — | — | — | — | — |
| EN0 | APFM | 0 | — (—) | — | — | -6.778 | -6.668 | -5.254 | — | — | — | — | — |
| EN0 | VSS-NFBFxLMS | 0 | — (—) | — | — | -3.135 | -4.030 | -2.635 | — | — | — | — | — |
| ENm10 | P5-early-cap | 0 | — (—) | — | — | -8.778 | -8.631 | -7.449 | — | — | — | — | — |
| ENm10 | nors-cap | 0 | — (—) | — | — | -8.778 | -8.631 | -7.449 | — | — | — | — | — |
| ENm10 | Freq_FxNLMS_CohVSS | 0 | — (—) | — | — | -8.773 | -8.812 | -7.648 | — | — | — | — | — |
| ENm10 | APFM | 0 | — (—) | — | — | -6.884 | -6.821 | -6.019 | — | — | — | — | — |
| ENm10 | VSS-NFBFxLMS | 0 | — (—) | — | — | -3.256 | -4.320 | -3.138 | — | — | — | — | — |

## Paired differences

Proposed minus APFM, seed bootstrap 95% intervals, with pair counts. Recovery uses observed pairs.

| Condition | L1 whole (dB) | Disturbance +2 s (dB) | Recovery (s) | Frozen 0.13 s (dB) | Frozen 0.51 s (dB) |
|---|---|---|---|---|---|
| stat | -2.023 [-2.120, -1.928] (200) | — | — | — | — |
| RS-near | -4.832 [-4.957, -4.702] (200) | — | -1.824 [-2.332, -1.320] (159) | -4.453 [-4.992, -3.901] (200) | -7.203 [-7.851, -6.552] (200) |
| RS-far | -5.669 [-5.776, -5.553] (199) | — | -1.622 [-2.049, -1.184] (140) | -6.326 [-6.961, -5.727] (199) | -9.679 [-10.315, -8.984] (199) |
| EN0 | -1.713 [-1.820, -1.609] (200) | -0.346 [-0.547, -0.136] (200) | — | — | — |
| ENm10 | -1.894 [-1.991, -1.797] (200) | -1.430 [-1.545, -1.305] (200) | — | — | — |

## APFM selections

Percentages pool blocks with a recorded selection. η/P_x uses the median of per-run medians over blocks that ran the gradient update.

| Condition | Bank | Current | Zero | Non-current in first 2 s after switch | Selector changes (no-change) | η / median P_x |
|---|---:|---:|---:|---:|---:|---:|
| stat | 35.18% | 64.05% | 0.77% | — | 162695 | 4.12e-09 |
| RS-near | 20.81% | 70.23% | 8.96% | 16984 | — | 4.21e-09 |
| RS-far | 19.44% | 73.49% | 7.07% | 12864 | — | 4.14e-09 |
| EN0 | 35.59% | 63.43% | 0.98% | — | — | 4.12e-09 |
| ENm10 | 35.21% | 63.92% | 0.86% | — | — | 4.12e-09 |
