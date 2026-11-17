# Analytic check of Proposition 3 (SF mean fixed point = truncated infinite Wiener)

Script: `atsip/explore/prop3.py`. Output: `atsip/explore/out/prop3_summary.json`.

Setting: white x, expected second-order statistics only (no sample noise, so no measurement floor).
Measured IRs: `impulse_response/primary_ir.dat` (422 taps), `secondary_ir.dat` trimmed to 128 taps.
rho = 0.992751 (reflected zero radius of the secondary path), 20 log10 rho = -0.0632 dB/tap.

Definitions (all dB except where noted):
- `w_M`: M-tap Wiener solution, `J_o(M)` its expected cost.
- `w_SF = w_M + v`: v solves the linear mean fixed point of the frame model (N = 2M, expected circular
  correlations, SF operator), solved directly with no adaptation. Residual below 1e-14.
- `[w_inf]_M`: first M taps of the causal infinite Wiener solution (G = 2^17 grid, 16384 taps used).
- Excess = d' Phi d / J_o(M), with d = w - w_M and Phi the expected autocorrelation Toeplitz.
- Geometric a: frame-model fixed point with operator `irfft(Y * Lam^-a)[:M]`, and closed form
  `J_b(a) = ||f^a [f^-a q_-]_+||^2` (M -> inf), as in `/tmp/chk4/main2.py`.

## Table

| M | J_o(M)/J_d | SF frame, N=2M | prompt table | ideal [w_inf]_M | SF minus ideal | dist ‖w_SF-[w_inf]_M‖²/‖w_inf‖² | rel dev ‖v-d‖²/‖d‖² | geo a=0 | geo a=0.5 | geo a=1 | closed a=0.5 | closed a=1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | -10.49 | -25.56 | -25.6 | -25.17 | -0.39 | -26.3 | -1.5 | -284 | -28.20 | -20.75 | -30.22 | -21.50 |
| 192 | -10.83 | -18.76 | -18.8 | -19.26 | +0.50 | -32.5 | -13.5 | -283 | -24.93 | -17.07 | -29.89 | -21.17 |
| 256 | -12.17 | -16.54 | -16.5 | -16.24 | -0.30 | -37.8 | -12.9 | -279 | -21.83 | -14.54 | -28.54 | -19.82 |
| 384 | -21.22 | -11.48 | -11.5 | -11.38 | -0.10 | -46.9 | -20.9 | -270 | -16.57 | -7.89 | -19.49 | -10.77 |
| 512 | -31.05 | -17.06 | -17.1 | -16.92 | -0.14 | -63.1 | -22.9 | -260 | -9.44 | -0.73 | -9.66 | -0.94 |
| 768 | -31.15 | -44.02 | -44.0 | -43.94 | -0.08 | -86.2 | -21.5 | -260 | -9.48 | -0.78 | -9.56 | -0.84 |
| 1024 | -31.15 | -64.08 | -64.1 | -63.54 | -0.54 | -106.6 | -24.4 | -260 | -9.50 | -0.80 | -9.56 | -0.84 |
| 1536 | -31.15 | -100.18 | n/a | -99.57 | -0.61 | -143.2 | -25.8 | -259 | -9.52 | -0.81 | -9.56 | -0.84 |
| 2048 | -31.15 | -134.92 | n/a | -134.34 | -0.58 | -177.8 | -27.5 | -257 | -9.53 | -0.82 | -9.56 | -0.84 |

All values are excess J(w) - J_o(M) relative to J_o(M), in dB, except the two distance columns. "geo a=0" is zero to numerical precision (about -260 dB).

## Slopes (M >= 512)

- Least-squares fit over M = 512..2048: ideal -0.0747 dB/tap, SF frame -0.0751 dB/tap. This fit is pre-asymptotic (the early points fall faster).
- Adjacent pairs 1024-1536 and 1536-2048: ideal -0.0704 and -0.0679 dB/tap, SF frame -0.0705 and -0.0679 dB/tap.
- Tail energy of w_inf itself beyond M, same pairs: -0.0684 and -0.0638 dB/tap.
- Reference -0.0632 dB/tap (20 log10 rho).

## Geometric a = 0.5 and a = 1

- Frame model at every M >= 512: a = 0.5 gives -9.4 to -9.5 dB and a = 1 gives -0.7 to -0.8 dB, with no decay in M.
- Closed form (M -> inf): a = 0.5 gives -9.56 dB and a = 1 gives -0.84 dB (the same level as the frame model for M >= 768).
- The closed form is M-independent. At M = 128..384 the frame model differs from it by 0.8 to 7 dB (largest at M = 256, a = 0.5), and by 0.2 dB or less for M >= 512.

## Checks and caveats

- The frame model (expected circular correlations, `impl()` in `/tmp/chk4/sf.py`) is the reviewer's construction, reused as-is and not re-derived here. It is valid only at N = 2M: at M = 512 with N = 2048 or 4096 it gives +8.7 and +10.4 dB, so N = 2M is required.
- The frame excess reproduces the prompt's predicted table to within 0.05 dB at every listed M.
- The sample-based sweep (`RESULTS_sfM.md`) found that the tail-energy formula fails quantitatively. The agreement here is between two analytic quantities, not between the model and samples.
- The sample-based SF value at M = 512 (-17.18 dB, a145 source) is close to -17.06 here, but the sources differ (alpha-stable input in sfM, white here), so this is not a validation.

## Three-line conclusion

1. The SF frame fixed point and the truncated infinite Wiener solution have the same excess to within 0.61 dB at every M from 128 to 2048 (within 0.05 dB of the prompt's table at M <= 1024).
2. The vector identity w_SF = [w_inf]_M holds only approximately: the relative Euclidean deviation is -1.5 dB at M = 128 and -13 to -28 dB for M >= 192. The distance normalised by ||w_inf||^2 (-26 to -178 dB) is small mainly because ||w_inf||^2 is large, so it is weak evidence.
3. The excess falls at about -0.068 dB/tap over M = 1536..2048, against -0.063 predicted by rho, so the rho^(2M) rate is only approximate here. The geometric a = 0.5 and a = 1 biases stay at about -9.5 and -0.8 dB for M >= 512, and a = 0 has zero bias, so the flat -27 dB in the sample sweep is measurement floor, not bias.
