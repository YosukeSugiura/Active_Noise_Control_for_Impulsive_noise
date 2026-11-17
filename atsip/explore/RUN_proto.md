# RUN guide: prototype experiments A and B (THEORY_vss2.md Sec. 9)

Code: `atsip/explore/protoA.py` (experiment A = P1 + P2, branched segments) and `atsip/explore/protoB.py`
(experiment B = P3, concentration law). Both reuse `vssT_lib.py`. Purpose: locate which approximation of
THEORY_vss2.md Sec. 9 breaks. Prototype, not confirmatory.

All commands are run from `atsip/explore/`:

```bash
cd atsip/explore
```

Rules for the runner
- At most 2 worker processes at any time. `protoA.py run --procs 2` uses 2; `protoB.py` uses 1. Do not run
  protoA and protoB at the same time unless protoA uses `--procs 1`.
- `protoA.py run` skips states whose output already exists; `--force` recomputes.
- Smoke runs (`--smoke`) write to `out/protoA_smoke/` and `out/protoB_smoke/` and never touch the full outputs.
  Do not commit the smoke directories.

## 0. Self-checks (about 10 s in total, < 300 MB)

```bash
python protoA.py check
python protoB.py check
```

Each must end with `ALL CHECKS PASSED` (exit code 0).

protoA checks: paths reproduce the data builder; the frozen-normaliser SF direction equals `SFm.sf_dir`; the branch
loop equals `vssT_lib.simulate` with the same frozen normaliser (rel. diff 4e-15); at mu = 0, dir(d) + A W0 equals
`vssT_lib.frozen_dirs` (mu = 0 branches reproduce the frozen probes); the re-generated frame equals the actual
frame when w is constant; Delta - Delta^re equals the exact block-by-block split sum_k A^(k)(w_b-k - w_b); the
masked occupancy split sums to A_b v; the offline EMA computed from the Gram matrix equals `DirectEst` online
(L = 4, beta_d = 0.1 and L = 8, beta_d = 0.3); C_0 == Q; the branched prediction reduces to the Sec. 9.1 formula
for long segments; the Sec. 9.1 solver inverts its forward model.

protoB checks: pi of a single pulse = 1, of the uniform frame = 1/N (1/M), of two equal pulses = 0.5; same frame
energy in every condition; a pulse 1, 2, 4, 8 blocks before the reference frame leaves both directions bitwise
unchanged (fixed normaliser; see the note below); a pulse 16 samples before does change them; bilinear scaling;
the probe's direction equals protoA's for the same signal.

## 1. Smoke runs (measured: about 20 s in total)

```bash
python protoA.py run --smoke --R 8 --T 12 --pre 24 --procs 2 --force   # 7 s, 2 processes, ~235 MB each
python protoA.py report --smoke                                         # -> out/protoA_smoke/RESULTS_protoA_smoke.md (2 s)
python protoB.py run --smoke --R 20                                     # 3 s, ~260 MB
python protoB.py report --smoke                                         # -> out/protoB_smoke/RESULTS_protoB_smoke.md
```

## 2. Full runs

```bash
python protoA.py run --procs 2      # states i, ii, iii; R = 64, T = 40, pre-roll 48, mu in {0, .0125, .025, .05, .1}
python protoA.py report             # -> RESULTS_protoA.md
python protoB.py run                # 32 conditions, R = 200
python protoB.py report             # -> RESULTS_protoB.md
```

Measured on this machine (one full-size state run alone, `--procs 1`): protoA R = 64: 15 s and 650 MB per state;
R = 128: 27 s and 1.07 GB per state. Expected full protoA with 2 processes: about 35 s wall time, peak about
1.3 GB in total. protoB R = 200: 6 s, 265 MB. Each report: about 2 s.

Optional, if the tables of A are too noisy (finite-difference slopes, P2 at state iii): rerun with more
realisations, one process (about 1 min and 2 GB per state):

```bash
python protoA.py run --R 256 --procs 1 --force
python protoA.py report
```

Outputs to commit: `RESULTS_protoA.md`, `RESULTS_protoB.md`, `out/protoA/protoA_{i,ii,iii}.json`,
`out/protoA/protoA_{i,ii,iii}.npz` (about 1 MB each; the report is computed from these),
`out/protoB/protoB.json`.

## 3. What is computed

### A (protoA.py)

- Input white Gaussian (std sqrt 2, as G_s0); SF direction; normaliser (Srr, Pbar, eps, and so the SF factor)
  frozen at the time average of the G_s0 run statistics over blocks >= 500; g treated as 1 (u = Delta; the actual g
  is recorded). Delta = KD * direction, so E Delta = h v with h = 0.25.
- States W0: i: w = 0 (v = -wM); ii / iii: w of a fixed-mu 0.1 SF run on G_s0 at blocks 30 / 2000. References:
  the SF model fixed point (`fixed_point(("SF",), 0)`, "fp") and wM.
- Per state, R realisations; local signal of 8 + pre + T blocks; w frozen at W0 for the first 8 + pre blocks
  (frame before the branch point consistent with W0), then T adapted blocks for each mu (same realisations for all mu).
- Saved: Gram matrices E<Delta_t, Delta_s>, E<v_t, Delta_s>, E<v_t, v_s> over all recorded blocks (Euclidean and
  Phi, both references); the frozen (mu = 0) correlation sequences of eta, zeta, H~V and their cross; P2 lag terms;
  the masked occupancy c_k; frame-delay ratios; E<v_n, H~_n v_n> per block for the finite-difference slope.
  Everything else (C_l, Q, the EMA versions for any (L, beta_d), outer EMA, predictions) is computed in `report`
  from these, so the formulas can be changed without rerunning.
- Report tables per state: .0 frozen decomposition (X, |E Delta|^2/X, h_eff, nonscalar, N, S, f, tau_a, tau_m,
  Gamma_0, K_eta); .1 C_l and Q vs mu and l with predictions (Sec. 9.1 steady form; the same first-order theory with
  the finite sums of the branched segment, with Gamma exogenous ("br") or with the H~v part scaled with X_t ("brS");
  Sec. 2), the four-way split h^2<v,v> + h<v,eta> + h<eta,v> + <eta,eta> (measured vs predicted), the EMA versions
  for (L, beta_d) in {4, 8} x {0.1, 0.3, 1}; .2 slopes d/dmu at mu = 0 (finite difference, common random numbers);
  .3 one-step oracle [E<v,Delta>/E|Delta|^2]_+ vs mu/2 and the estimators (a) Sec. 2, (b) Sec. 9.1; .4 frame delay
  ratio vs the mean-operator prediction, masked c_k, effective delay d_eff; .5 P2 terms T1, T2, ratio, implied
  tau_m vs frozen tau_m and 2 tau_m - 1, and the finite-difference slope of E<v_n, H~_n v_n>.

### B (protoB.py)

- x = sqrt(2) sigma_t xi_t; same xi in all conditions; target block with reference frame x[F0, F0 + N) and error
  window its last M samples. Normaliser frozen (same as A). zeta = Delta at w = w_fp; A v for v = -w_fp.
- Conditions (frame energy sum sigma^2 = N in all): uniform (stationary / frame only); one pulse at frame offsets
  0, 256, 511, 512, 640, 768, 1023 (>= 512: inside the error window); two pulses (spacing 1, 16, 128, 512); spacing
  16 at five positions; tail test: stationary base or a single pulse at 768, with and without an earlier pulse 1, 2,
  4, 8 blocks (and 127, 64, 16 samples) before the reference frame.
- Report: mean / g, fluctuation / g^2, noise / g^2 (centred and raw), |E zeta|^2, all relative to the stationary
  uniform condition, next to pi_x, pi_xM (from sigma) and the predictions (noise (1 - pi); fluctuation 1 (Sec. 9.5)
  or (1 - pi) (old Sec. 3)).
- Note on the tail test: with the normaliser fixed, a pulse 1 block (128 samples) or more before the reference frame
  cannot reach the error window (the longest response, residual + secondary path, is 639 samples < 128 + 512), and the
  reference frame does not contain it. So these rows are identical to their base by construction (this is a
  self-check). A tail effect seen in the closed-loop runs must come from the adaptive normaliser (Srr, Pbar EMA, and so
  g) or from the window used for pi, not from the direction itself. The 127 / 64 / 16-sample rows show the size of the
  direct effect at the boundary.
