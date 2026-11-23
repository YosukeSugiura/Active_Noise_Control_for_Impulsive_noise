# Table 1. Settings of the proposed algorithm and selected step sizes of the comparison algorithms

## Proposed algorithm

| Parameter | Value |
| --- | --- |
| Sampling rate; high-pass cutoff of the input | 16 kHz; 100 Hz |
| @@Filter length $M$; FFT length $N$; block shift $B$ | 512; 1024; 128 samples (8 ms) |
| @@Length of $\hat{s}(k)$, $L_s$; error window $M_e$ | 512; 512 samples |
| @@Weight of the exponential average $\hat{\mathbf{S}}_{rr,n}$; regularization $\epsilon$ | 0.05; 0.01 |
| @@Step sizes $\sigma_{\mathrm E}$; $\sigma_{\mathrm L}$ | 1; 0.0566 |
| @@Lower bound $g_{\min}$ in the early phase; in the late phase | 0.55; 0.25 |
| @@Length of the early phase $n_{\mathrm E}$ | 32 blocks, counted from the start of the trial after a transfer |
| @@Windows of $e(k)$ and $r(k)$ in $z_n$ | 512; 1024 samples |
| @@Weight of the exponential average $\bar{z}_n$; threshold $\gamma$ | 0.05; 4, for two consecutive blocks |
| @@Interval $n_{\mathrm G}$ before the next trial | @@64 blocks, with $\bar{z}_n$ held constant |
| @@Discard threshold $\rho_{\mathrm{abort}}$ | 4, from block 2 of the trial |
| @@Transfer rules $(n_1, \rho_1)$; $(n_2, \rho_2)$ | (4, 0.25); (16, 0.5) |

## Comparison algorithms

| Algorithm | Step size μ | Other setting |
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

The comparison algorithms use one setting for all noise types and conditions (Section 5.3).
