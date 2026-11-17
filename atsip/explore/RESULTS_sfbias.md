# Why the SF normaliser keeps a bias on a145 (atsip/explore/sfbias.py)

Source a145, linearised mean fixed point (fp.py conventions: sampled blocks i >= 16, every 2nd block, 1867 blocks).
Excess = 10 log10(mean |ex|^2 / mean |e_opt|^2) over the second half, relative to the irreducible error J_o.
More negative is better. L1 fixed-point ANR; the L1 floor (w_opt) is -29.52 dB. Runtime 701 s, one process.

| variant (operator unchanged unless stated) | excess (dB) | vs Rall base | L1 fixed-point ANR (dB) |
|---|---|---|---|
| geometric a=0 (Lam = Pbar), reference | -27.59 | -10.41 | -29.45 |
| base: Rall, Lam = Srr + eps (= fp.py ER) | -17.18 | 0 | -29.32 |
| 1. F from 4N grid, sampled to N | -17.22 | -0.04 | -29.33 |
| 1. F from 8N grid, sampled to N | -17.22 | -0.04 | -29.33 |
| 2. causal split on 2N grid (zero-padded) | -17.64 | -0.46 | -29.35 |
| 2. causal split on 4N grid (zero-padded) | -17.67 | -0.49 | -29.35 |
| 2. 2N grid + causalM tail | -17.65 | -0.47 | -29.35 |
| 3. frozen F (long-term mean of abs(R)^2) | -17.05 | +0.13 | -29.38 |
| 4. eps_reg 1e-3 Pbar | -17.22 | -0.04 | -29.33 |
| 4. eps_reg 1e-2 Pbar | -18.43 | -1.25 | -29.38 |
| 5. causalM tail (1/F as M-tap causal filter), N grid | -17.21 | -0.03 | -29.32 |
| control: N grid, zero-padded r-frame R (no padding) | -17.37 | -0.19 | -29.34 |
| diag: flat F (Lam = Pbar), same causal operator | -27.59 | -10.41 | -29.45 |
| diag: frozen a=0.5 (Lam = Lfz^0.5 Pbar^0.5) | -25.50 | -8.32 | -29.51 |

Adaptive (no ramp): no variant met the criterion (at least half of the 10.4 dB gap, i.e. <= -22.4 dB), so no
variant was run adaptively. Current SF (specfac.sim_sf, eps_reg 0), whole / first5 / last10 in dB:

| source | mu 0.1 | mu 0.05 |
|---|---|---|
| a145 | -29.00 / -28.08 / -29.24 | -28.88 / -26.24 / -29.45 |
| a185 | -30.11 / -28.71 / -30.56 | -29.75 / -25.76 / -30.69 |
| a120_s0 | -28.14 / -26.68 / -28.40 | -27.97 / -24.56 / -28.63 |

Conclusion:
1. None of candidates 1 to 5 removes the bias. The best implementation change is eps_reg 1e-2 (-1.25 dB) and the
   largest grid effect is 0.5 dB; the -27.6 dB level appears only with a flat F.
2. The bias follows the spectral shape of F (Lam = Srr, exponent a = 1). The flat F (a = 0) gives -27.59 dB and a = 0.5 gives -25.50 dB with the same causal operator.
3. Time variation (candidate 3) is not the cause: a frozen Srr keeps -17.05 dB, so the shape is deterministic.
   I did not derive from first principles why the causal part of the mean update is nonzero for a shaped F, which conflicts with the K_+- = 0 claim.

Files: out/sfbias_fp_a145.json (fixed-point rows and operator check), out/sfbias_adapt.csv and out/sfbias_adapt.json.
Library: sfbias_lib.py. The fixed-point variants run on a145 only; the adaptive table uses the current SF only.
