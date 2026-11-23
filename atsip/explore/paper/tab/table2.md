# Table 2. Reconvergence after a source switch

## Median levels

| Switch | Metric | Proposed | CohVSS-FxNLMS | No restart |
| --- | --- | --- | --- | --- |
| Near | Recovery (s) | 0.360 | 0.976 | 1.048 |
| Near | Frozen-filter J at 0.13 s (dB) | -2.59 | -0.03 | 1.14 |
| Near | Frozen-filter J at 0.51 s (dB) | -7.89 | -5.01 | -3.26 |
| Far | Recovery (s) | 0.400 | 1.256 | 1.128 |
| Far | Frozen-filter J at 0.13 s (dB) | -6.75 | -0.88 | 0.22 |
| Far | Frozen-filter J at 0.51 s (dB) | -10.87 | -6.84 | -4.47 |

## Paired differences

| Switch | Metric | Proposed - APSIPA 2026 | Proposed - No restart |
| --- | --- | --- | --- |
| Near | Recovery (s) | -0.397 [-0.523, -0.282] (172) | -0.459 [-0.564, -0.367] (172) |
| Near | Frozen-filter J at 0.13 s (dB) | -3.50 [-4.08, -2.88] (200) | -4.58 [-5.26, -3.90] (200) |
| Near | Frozen-filter J at 0.51 s (dB) | -2.70 [-3.17, -2.15] (200) | -4.54 [-4.98, -4.08] (200) |
| Far | Recovery (s) | -0.500 [-0.600, -0.384] (172) | -0.587 [-0.721, -0.463] (171) |
| Far | Frozen-filter J at 0.13 s (dB) | -4.99 [-5.48, -4.46] (200) | -6.25 [-6.92, -5.62] (200) |
| Far | Frozen-filter J at 0.51 s (dB) | -3.81 [-4.29, -3.22] (200) | -6.07 [-6.56, -5.50] (200) |

Recovery uses each method's post-change reference level +3 dB and a Kaplan-Meier median.
Frozen-filter J is the residual-to-primary power ratio with the first 256 ms excluded.
Differences are paired means with seed-bootstrap 95% intervals and pair counts in parentheses. Recovery differences use pairs with recovery observed in both runs.
