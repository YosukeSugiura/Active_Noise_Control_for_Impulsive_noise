# RUN guide: protoC (THEORY_vss2.md Sec. 10.6 method changes, SF direction, prototype)

Code: `atsip/explore/protoC.py` (reuses `vssT_lib.simulate`, `vssT.rules_all`, `protoA.solve91`/`FixedSF`;
`specfac.sf_dir` has a new `ret_y=True` option that also returns the intermediate vector and the factor F;
its default output is unchanged, checked bitwise). All commands are run from `atsip/explore/`:

```bash
cd atsip/explore
```

Rules for the runner
- At most 2 worker processes at any time: one command with `--procs 2`, or two commands with `--procs 1`. Never more.
- Every C1 job writes `out/protoC/protoC_C1_<tag>.json` (+ `.npz`) and rewrites it after every rule, so a killed
  job keeps its finished rules. Re-running the same command skips finished tags and resumes unfinished ones at
  the first missing rule. `--force` recomputes. C2 jobs write `out/protoC/protoC_C2_<tag>.json`.
- Smoke runs (`--nb <blocks>`) write to `out/protoC_smoke/` and never touch the full outputs. Do not commit
  `out/protoC_smoke/`.
- Divergence: the run stops, curves and scores become NaN, `div` = true (non-finite w or Phi-excess > 1e8).
  Amplification (`amp`: L1 whole > 0 dB without divergence) is counted separately.

## 0. Self-check (4 s, ~200 MB)

```bash
python protoC.py check
```

Must end with `ALL CHECKS PASSED` (exit code 0). Checks:
- `sf_dir(..., ret_y=True)` returns the same direction bitwise as before, and equals `protoA.FixedSF`;
- y -> Delta: 1/F applied to y (zero-padded) reproduces KD * direction (rel. diff 0), x is zero beyond N/2;
- gamma = 0 with no correction == fixed mu 0.1 (bitwise w and e), for all three targets and both coordinates;
- a* for c_k = (0.238, 0.247, 0.247, 0.247, 0.010) is 0.161 (rho 0.671; rho(0.5) = 0.844 as in the Sec. 10.1 table);
- identity factor (Lam = 1, F = 1): y == Delta, and the y statistics == the Euclidean ones == `vssT_lib.DirectEst('E')`;
- the (X, N) solve == `protoA.solve91` with tau_m -> 2 tau_m - 1, h -> h_eff; the forward forms reproduce (C, Q);
- delay-opt sanity (X = 0 -> target 0; N = 0 -> target near a*/h_eff);
- the C2 continuation with the dp+g policy == the simulator (gamma = 0), rel. diff 0.

## 1. Smoke runs (about 50 s in total)

```bash
python protoC.py c1 --tags G_s0,II300_s0 --nb 300 --procs 2          # all 53 rules, 9 s
python protoC.py c1 --tags pc_s0,PS_s0 --nb 2600 --procs 2 \
  --rules "SF-Y delaycap dp gam.03;SF-E delaycap dp gam.03;SF-Y delayopt g gam.1;SF-Y onestep dp+g gam.03;SF-D g gam0.03 a0 1;SF mu.5"   # 8 s
python protoC.py c2 --tags G_s0,pc_s0 --nb 2000 --procs 2            # 26 s
python protoC.py report --smoke        # -> out/protoC_smoke/RESULTS_protoC_smoke.md
```

(The pc L1/L2 windows need full-length runs; in smoke they are NaN by design.)

## 2. Full runs

Run one after the other (each uses 2 processes):

```bash
python protoC.py c1 --procs 2      # G, BU, SM, PS, II300, a145, a120, pc x seeds 0,1,2 = 24 jobs, 53 rules each
python protoC.py c2 --procs 2      # G_s0, a145_s0, pc_s0; 4 rules (delayopt / delaycap x dp / g, gam.03)
python protoC.py report            # -> atsip/explore/RESULTS_protoC.md (tables only)
```

Then commit `RESULTS_protoC.md` (and nothing under `out/protoC_smoke/`).

Constants (CLI; defaults in brackets): `--f` [0.85], `--tau-a` [3.8], `--tau-m` [4.0], `--heff` [0.95 h = 0.2375],
`--ck` [0.238,0.247,0.247,0.247,0.010], `--gz --gm` [0, 0] (Gbar of Sec. 9.1), `--dopt-T` [8],
`--dopt-k excl1|full` [excl1], `--mu-fmax` [0.75/(h_eff (tau_a-1)) = 1.128], `--rec-db` [-20].
The job files store these constants; a job whose stored constants differ is recomputed from scratch, so to
keep a run with other constants move `out/protoC/` aside first.

## 3. Runtime and memory

Measured in the smoke runs: the simulator costs ~0.33 ms per block per rule, i.e. ~1.3 s per full 3750-block
run; data preparation ~2-4 s per tag.

| job | smoke measured | full, per job (extrapolated) | full, whole command with 2 procs | peak RSS per process |
|---|---|---|---|---|
| check | 3.3 s | - | - | 195 MB |
| C1 (per tag, 53 rules) | 300 blocks, 53 rules: 4 s; 2600 blocks, 6 rules: 6 s | ~75 s | ~15-20 min (24 jobs) | ~200 MB at 1200 blocks, < 350 MB full |
| C2 (per tag, 4 rules) | 2000 blocks: G 17 s, pc 25 s | ~30-45 s | ~1-2 min (3 jobs) | < 350 MB |

Total under 25 min; memory with 2 processes < 1 GB.

## 4. Outputs

| file | content |
|---|---|
| `out/protoC/protoC_C1_<tag>.json` | constants (`par`); per rule: L1 / L2 (whole, first 5 s, last 10 %; pc windows), div, amp, excess summaries (first 1 s, first 5 s, last 10 %; pc 15-17 s, 25-30 s), PS: max excess in the 2 s after each step and recovery time to -20 dB (`ps_rec_*`) and to the 1 s pre-step mean + 3 dB (`ps_recrel_*`), BU: excess folded over the 1 s burst period (`bu_fold`, peak, 0-0.25 s, 0.5-1 s), mean applied mu per phase, zero dwell, clip fractions |
| `out/protoC/protoC_C1_<tag>.npz` | per rule (`<rule>|<key>`): `ex` (Phi-excess every block), `mu_app`; for SF-Y / SF-E also `mu_state`, `target`, `X`, `N`, `C`, `Q`, `ypow` (= |y_n|^2), `dpclip`, `mufclamp`, `xclip`, `stclip`; `g` |
| `out/protoC/protoC_C2_<tag>.json` | per rule, per sampled block: g, mu_state, target, mu_app, factor, X_hat, N_hat, Q, ex0, argmin and min of the T-step mean excess for T = 4, 8, 16 |
| `RESULTS_protoC.md` | `python protoC.py report` (tables only; more negative dB is better) |

## 5. Rule names

| name | meaning |
|---|---|
| `SF-Y <target> <corr> gam<g>` | statistics C, Q on u = factor x y (y coordinates, Sec. 10.4) |
| `SF-E <target> <corr> gam<g>` | same rule, statistics on u = factor x Delta (control for change 1; dp still uses |y|^2) |
| target `onestep` | [X/h_eff - mu ((tau_a-1) N + (2tau_m-2) f X)/2]_+ / Q |
| target `delaycap` | min(onestep, a*/h_eff), a* from the c_k (0.161 -> 0.678) |
| target `delayopt` | grid argmin (40 values in [0, 1.5]) of the predicted 8-block-ahead second moment (model in the protoC.py docstring) |
| corr `none` / `g` / `dp` / `dp+g` | applied mu = mu_state x {1, 1/max(g, 0.25), clip(Qbar_y/|y_n|^2, 0.1, 10), product} |
| `gam.03`, `gam.1` | mu_state <- clip((1-gamma) mu_state + gamma target, 0, 1.5), start 0.3 |
| `SF-D g gam0.03 a0 1`, `SF+ramp`, `SF-V2`, `SF mu.1` | controls from vssT (pilot best, baselines) |
| `SF mu.5` | fixed mu 0.5 (delayed-LMS optimum region; shows the second-order limit) |
