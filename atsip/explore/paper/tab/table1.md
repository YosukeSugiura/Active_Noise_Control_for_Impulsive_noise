# Table 1. Evaluation settings and selected baseline steps

## Proposed method

| Setting | Value |
| --- | --- |
| Sampling rate; input high-pass cutoff | 16 kHz; 100 Hz |
| Control filter M; FFT N; hop B | 512; 1024; 128 samples (8 ms per block) |
| Secondary-path model; error window | 512 taps; 512 samples |
| SF regularization; late-stage design target | 0.01; 0.03 |
| Early / late step before gain normalization | 1.0 / 0.0566 |
| Gain floor, early and background / late | 0.55 / 0.25 |
| Cold-start phase; restart phase budget | 32 blocks each; restart budget includes the trial |
| Detection error / reference windows | 512 / 1024 samples |
| Detector EMA weight; ratio; consecutive blocks | 0.05; 4; 2 |
| Guard after detection; EMA update | 64 blocks; frozen from detection until guard release |
| Trial abort ratio; earliest check | >4 or nonfinite; block 2 |
| Early transfer / final transfer | ratio <= 0.25 from block 4 / <= 0.5 at block 16 |

## Comparison algorithms

| Method | Selected step μ | Selected setting |
| --- | --- | --- |
| FxNLMS | 0.01875 | -- |
| Th-FxNLMS | 0.01875 | Error clipping to [-200, 200] |
| FxlogNLMS | 0.015625 | -- |
| MFxLMM | 0.000125 | -- |
| FxNFLOC | 0.1 | -- |
| EVSS-FxeLMS | 0.003125 | -- |
| FreqFxLMS | 0.0225 | -- |
| CohVSS-FxNLMS | 0.09 | -- |
| C-FXLMS | 0.0375 (fast), 0.009375 (slow) | Normalized component filters |

The comparison algorithms use the common settings selected on design data (TUNE.json, TUNE_cc.json).
