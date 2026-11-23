# Table 3. Cost of the restart mechanism

## Stationary path and error-side disturbances

| Condition | Method | 30 s L1 | Last 10 s | Post-event 2 s | Transfers | Matches |
| --- | --- | --- | --- | --- | --- | --- |
| No change | Proposed | -9.03 | -9.11 | -- | 0 | 200/200 |
| No change | CohVSS-FxNLMS | -8.98 | -9.18 | -- | -- | -- |
| No change | No restart | -9.03 | -9.11 | -- | 0 | -- |
| Disturbance (0 dB) | Proposed | -8.49 | -8.15 | -5.60 | 13 | 187/200 |
| Disturbance (0 dB) | CohVSS-FxNLMS | -8.54 | -8.40 | -6.22 | -- | -- |
| Disturbance (0 dB) | No restart | -8.46 | -8.07 | -5.59 | 0 | -- |
| Disturbance (-10 dB) | Proposed | -8.78 | -8.63 | -7.45 | 2 | 198/200 |
| Disturbance (-10 dB) | CohVSS-FxNLMS | -8.77 | -8.81 | -7.65 | -- | -- |
| Disturbance (-10 dB) | No restart | -8.78 | -8.63 | -7.45 | 0 | -- |

## Paired L1 differences (Proposed minus baseline)

| Condition | Baseline | Window | Difference [95% CI] |
| --- | --- | --- | --- |
| No change | CohVSS-FxNLMS | 30 s | -0.053 [-0.081, -0.025] (200) |
| No change | CohVSS-FxNLMS | Last 10 s | +0.072 [+0.039, +0.106] (200) |
| Disturbance (0 dB) | CohVSS-FxNLMS | Post-event 2 s | +0.622 [+0.533, +0.715] (200) |
| Disturbance (0 dB) | No restart | Post-event 2 s | -0.011 [-0.031, +0.000] (200) |
| Disturbance (-10 dB) | CohVSS-FxNLMS | Post-event 2 s | +0.198 [+0.144, +0.256] (200) |
| Disturbance (-10 dB) | No restart | Post-event 2 s | -0.001 [-0.002, +0.000] (200) |

## Internal comparisons (200 runs per condition)

| Method | No change | Disturbance (0 dB) | Disturbance (-10 dB) | Near | Far |
| --- | --- | --- | --- | --- | --- |
| Proposed | 0/0/0 | 0/13/13 | 0/2/2 | 0/140/0 | 0/181/1 |
| No restart | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 |
| Proposed without 1/g | 7/85/85 | 7/92/92 | 7/85/85 | 8/218/95 | 9/255/100 |
| Immediate restart | 0/71/71 | 0/85/85 | 0/72/72 | 0/221/72 | 0/259/76 |
| Always-on background trial | 0/96/96 | 0/248/248 | 0/148/148 | 0/324/243 | 0/354/270 |
| Always reconstructed error | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 |
| Fixed-step SF | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 |

## Approximate operations per block

| State | Ref. FFT | Ref. mult. | Fast FFT | Fast mult. | Ratio |
| --- | --- | --- | --- | --- | --- |
| Ordinary block | 6/1 | 284,672 | 8/5 | 118,016 | 0.41 |
| Ordinary block with trial | 17/2 | 429,056 | 19/5 | 241,920 | 0.56 |

L1 values are in dB. Transfers count events. Output matches require exact numerical equality for both error and control output against No restart.
Internal comparisons show divergences / all transfers / unmatched transfers.
Ref. FFT counts real/complex transforms. Fast FFT counts 1024-/256-point transforms. The benchmark counts FIR and FFT multiplications.
