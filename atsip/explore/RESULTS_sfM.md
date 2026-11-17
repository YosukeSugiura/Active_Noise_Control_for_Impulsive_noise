# SF fixed-point bias vs finite filter length (atsip/explore/sfM.py)

Linearised mean fixed point, fp.py conventions (sampled blocks i >= 16, every 2nd block).
Excess = 10 log10(mean |ex|^2 / mean |e_opt|^2) over the second half, relative to the M-tap floor J_o(M).
More negative is better. L1 = fixed-point ANR (dB). Primary IR kept at full length (422 taps).
tail = ||r_re(tau>=M)||^2 / ||r_re||^2 over tau in [0,2M) at w_opt(M) (unweighted, r_re = (1/L) sum r(n) e_opt(n+tau)).
tail_norm = sum_{tau>=M} r_re^2 / (mean r^2 * mean e^2) (normalised, so not forced to 1 by the Wiener orthogonality).
pred = stationary SF operator on the circular lag kernel of the frame window, scaled by kappa fitted on the full-lag forcing (tail-only uses tau>=M).

## a145

| M | B | J_o (dB) | SF r0 | SF r1e-2 | geo a0 | geo a0.5 | geo a1 | tail ratio | causal ratio | pred tail-only | pred full | actual SF r0 | kappa | cos(act,pred) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 256 | 64 | -8.74 | -15.60 | -16.13 | -31.25 | -21.34 | -14.62 | 1.000 | 0.000 | -31.38 | -31.72 | -15.60 | 33.3 | 0.16 |
| 512 | 128 | -27.62 | -17.18 | -18.43 | -27.59 | -9.87 | -1.13 | 0.959 | 0.041 | -13.55 | -13.37 | -17.18 | 569 | 0.75 |
| 1024 | 128 | -27.72 | -25.22 | -25.80 | -26.01 | -9.80 | -1.09 | 0.946 | 0.054 | -44.37 | -26.60 | -25.22 | 2.13e+03 | 0.77 |
| 1024 | 256 | -27.72 | -25.03 | -25.40 | -25.52 | -9.79 | -0.92 | 0.946 | 0.054 | -41.42 | -24.59 | -25.03 | 2.67e+03 | 0.81 |
| 2048 | 512 | -27.72 | -23.74 | -23.90 | -24.04 | -9.60 | -0.90 | 0.176 | 0.824 | -54.00 | -27.35 | -23.74 | 953 | 0.71 |

## a185

| M | B | J_o (dB) | SF r0 | SF r1e-2 | geo a0 | geo a0.5 | geo a1 | tail ratio | causal ratio | pred tail-only | pred full | actual SF r0 | kappa | cos(act,pred) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 256 | 64 | -25.75 | -15.85 | -16.27 | -29.07 | -21.07 | -14.51 | 0.999 | 0.001 | -24.79 | -25.50 | -15.85 | 83.1 | 0.42 |
| 512 | 128 | -44.61 | -16.64 | -17.48 | -23.15 | -9.51 | -0.93 | 0.653 | 0.347 | -16.70 | -14.84 | -16.64 | 524 | 0.90 |
| 1024 | 256 | -44.69 | -21.25 | -21.36 | -21.44 | -9.34 | -0.86 | 0.248 | 0.752 | -32.07 | -21.40 | -21.25 | 965 | 0.88 |
| 2048 | 512 | -44.66 | -19.46 | -19.51 | -19.61 | -9.18 | -0.84 | 0.246 | 0.754 | -50.94 | -20.51 | -19.46 | 1.73e+03 | 0.94 |

## col145_s10

| M | B | J_o (dB) | SF r0 | SF r1e-2 | geo a0 | geo a0.5 | geo a1 | tail ratio | causal ratio | pred tail-only | pred full | actual SF r0 | kappa | cos(act,pred) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 256 | 64 | -18.83 | -15.91 | -15.98 | -25.24 | -18.81 | -14.34 | 0.999 | 0.001 | -25.30 | -24.91 | -15.91 | 85.9 | 0.46 |
| 512 | 128 | -34.19 | -14.46 | -14.41 | -22.30 | -10.80 | -2.50 | 0.979 | 0.021 | -14.50 | -14.40 | -14.46 | 432 | 0.81 |
| 1024 | 256 | -34.36 | -21.34 | -21.44 | -21.23 | -10.84 | -2.64 | 0.233 | 0.767 | -37.13 | -22.96 | -21.34 | 1.07e+03 | 0.81 |
| 2048 | 512 | -34.34 | -19.81 | -19.83 | -19.85 | -10.61 | -2.47 | 0.332 | 0.668 | -32.20 | -21.69 | -19.81 | 2.07e+03 | 0.87 |

## gauss

| M | B | J_o (dB) | SF r0 | SF r1e-2 | geo a0 | geo a0.5 | geo a1 | tail ratio | causal ratio | pred tail-only | pred full | actual SF r0 | kappa | cos(act,pred) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 256 | 64 | -32.50 | -15.50 | -15.91 | -27.90 | -20.62 | -14.32 | 0.999 | 0.001 | -23.89 | -24.13 | -15.50 | 97.4 | 0.43 |


Gaussian row: same pipeline with Gaussian excitation (control, not impulsive). SF -15.50 vs flat -27.90 at M=256, so the gap is not specific to alpha-stable noise.

## Controls and settings

- Control (M = 512, primary IR cut to 128 taps, out/sfM_a145_M512_B128_hp128.json): the M-tap Wiener energy beyond lag 256 falls from 0.091 to 0.0003, and the normalised tail energy from 4.0e-3 to 6.6e-5. SF gap to flat drops from 10.4 dB to 3.5 dB (-24.64 vs -28.12). J_o changes too (-27.98 dB), since the plant changes.
- B = 128 vs 256 at M = 1024 (a145): SF -25.22 vs -25.03, flat -26.01 vs -25.52. Same conclusion.
- The primary IR (422 taps) is kept at full length for every M, so M = 256 is a 256-tap controller against a 422-tap plant. The old code trimmed the IR to M, which is the same at M = 512 and above.
- "pred" is the stationary SF operator on the circular frame-window lag kernel, scaled by kappa fitted on the full-lag forcing. It is not calibrated: kappa spans 33 to 2700 and cos(actual, predicted) spans 0.16 to 0.94. Tail-only and full predictions differ by 10 to 30 dB at M >= 1024. The formula check therefore fails quantitatively.
- Sampling caveat: at M = 2048 only 461 blocks are used, and the sampled windows overlap. The effective sample is small, so the M = 2048 numbers carry more estimation noise.
- Reproduction: M = 512 gives SF -17.18, SF 1e-2 -18.43, flat -27.59, the same as RESULTS_sfbias.md.

## Verdict (3 lines)

1. Partly supported: the gap between SF and flat shrinks with M (a145 15.6, 10.4, 0.5, 0.3 dB for M = 256, 512, 1024, 2048; a185 13.2, 6.5, 0.2, 0.2; col145_s10 9.3, 7.8, -0.1, 0.0), and the M = 512 short-filter control cuts the gap from 10.4 to 3.5 dB.
2. Not confirmed as the stated mechanism: the normalised tail energy at lags >= M does not fall from 512 to 1024 on a145 (0.0040 vs 0.0042), the SF excess is not monotone in M (a145 -25.0 at 1024, -23.7 at 2048), and the flat excess rises from -31 to -24 dB, so part of the gap closure is a common floor.
3. Not specific to SF: geometric a = 0.5 and a = 1 stay at -9 to -11 dB and -0.8 to -2.6 dB for every M >= 512 with no shrinkage, and the tail-only stationary prediction does not reproduce the actual SF level. L1 ANR differences are at most about 0.3 dB for M >= 512.
