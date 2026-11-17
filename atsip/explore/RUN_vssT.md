# RUN guide: vssT verification experiments (THEORY_vss2.md, Sec. 7)

Code: `atsip/explore/vssT.py` (CLI, experiments, report) and `atsip/explore/vssT_lib.py` (simulator loop,
rules, estimators, probes). All commands below are run from `atsip/explore/`:

```bash
cd atsip/explore
```

Rules for the runner
- At most 2 worker processes at any time. Either run one command with `--procs 2`, or two commands with
  `--procs 1` each, never more.
- Every job writes `out/vssT_<E>_<tag>.json` (+ `.npz`) and rewrites it after every rule, so a killed job
  keeps its finished rules. Re-running the same command skips finished jobs and resumes unfinished E4-E6 jobs
  at the first missing rule. `--force` recomputes.
- Smoke runs (`--nb <blocks>`) write to `out/vssT_smoke/` and never touch the full outputs. Do not commit
  `out/vssT_smoke/`.
- Divergence: the run stops, the excess curve and scores become NaN, `div` = true (non-finite w, or
  Phi-excess above 1e8).

## 0. Self-check (8 s, ~300 MB)

```bash
python vssT.py check
```

Must end with `ALL CHECKS PASSED` (exit code 0). What it checks:
- the vssT loop reproduces `simx.run` (SF mu 0.1, GS ramp, B0, SF-V2, GS-V2): e, w, excess, relative diff < 1e-9
  (measured: 0 for e and w, ~3e-15 for the excess);
- direct rule with gamma = 0 and no correction == fixed mu (bitwise); Switch with omega frozen at 0 == direct
  rule on B0 (bitwise);
- Sec. 2 numerical check: C/Q = -0.048 at mu = 0.1 gives psi = 0.042, mu_hat = 0.046;
- spectral inner product with w = None or w = 1 equals the Euclidean product (and `vssI2.ipw`);
  the spectral estimator with no weights equals the Euclidean estimator;
- E1: on a synthetic quadratic the brute-force grid argmin equals the analytic optimum;
- frozen-coefficient probe == simulator with mu = 0 and w0 = w; constant-mu continuation == simulator;
- E2 correlation-sum code on a vector AR(1) sequence gives (1+a)/(1-a);
- participation ratio: flat frame 1/N, single impulse 1;
- Wiener reference and model fixed points (SF, B0 -0.81 dB, geo -9.55 dB) and that model.py's cache is restored;
- full-length L1 metrics == `simx.scores`.

## 1. Smoke runs (all together under 2 min)

```bash
python vssT.py e1 --tags G_s0,pc_s0 --nb 2100 --procs 2
python vssT.py e2 --tags G_s0,a145_s0 --nb 400 --procs 2
python vssT.py e3 --tags II_s0,BU_s0 --nb 400 --procs 2
python vssT.py e4 --tags G_s0,II_s0 --nb 300 --procs 2
python vssT.py e5 --tags a145_s0 --nb 300 --procs 1
python vssT.py e6 --tags pc_s0,a120_s0 --nb 2100 --procs 2
python vssT.py report --smoke        # -> out/vssT_smoke/RESULTS_vssT_smoke.md
```

## 2. Full runs

Run the commands one after another (each uses 2 processes). The order does not matter; they are independent.

```bash
python vssT.py e1 --procs 2      # tags G_s0, a145_s0, a120_s0, pc_s0; rules: SF mu.1, SF+ramp, SF-D g gam0.03 a0 1, GS
python vssT.py e2 --procs 2      # G_s0, G_s1, a145_s0, a145_s1; kinds SF, B0, geo; probes init / rand / late
python vssT.py e3 --procs 2      # G, PS, BU, SM, II, a145, a120 (seed 0); kind SF
python vssT.py e4 --procs 2      # 8 classes x seeds 0,1,2 = 24 jobs, 23 rules each
python vssT.py e5 --procs 2      # G, a145, a120, pc x seeds 0,1,2 = 12 jobs, 11 rules
python vssT.py e6 --procs 2      # G, a145, a120, pc x seeds 0,1,2 = 12 jobs, 6 rules
python vssT.py report            # -> atsip/explore/RESULTS_vssT.md
```

Optional extensions (run only if asked; they write separate tags or need `--force`):

```bash
python vssT.py e3 --classes II300 --seeds 0 --procs 1           # impulses 300x std (pi -> 1 regime)
python vssT.py e3 --kinds SF,B0,geo --force --procs 2           # E3 for all three directions
python vssT.py e4 --seeds 3,4 --procs 2                         # more seeds (adds tags *_s3, *_s4)
python vssT.py e1 --rules "SF mu.1;SF-D g gam0.03 a0 3.8;GS-D g gam0.03 a0 1" --force --procs 2
```

`--pidef x|xM|r` selects the pi window for the 1/(g rho) rules (default `x`), `--gmin` the floor of g
(default 0.25). The file names do not record these values and finished jobs are skipped, so to rerun with other
values move the previous `out/vssT_E4_*` files elsewhere first, then run with `--force`.

## 3. Expected runtime and memory

Measured from the smoke runs (one process; the simulator costs ~0.4 ms per block per rule, i.e. ~1.5 s for a
full 3750-block run, measured in `check`). Full-length numbers are linear extrapolations in the number of blocks.

| job | smoke measured | full, per job (extrapolated) | full, whole command with 2 procs | peak RSS per process |
|---|---|---|---|---|
| E1 (per tag, 4 rules) | 2100 blocks, 3 rules: 10 s | ~25 s (pc tag: ~30 s) | ~1 min | ~320 MB |
| E2 (per tag, 3 kinds) | 1200 blocks: 3 s | ~10-15 s | ~1 min | ~350 MB (231 MB at 1200 blocks) |
| E3 (per tag, SF) | 400 blocks: 1 s; 1200 blocks, 2 kinds: 2 s | ~5 s | < 1 min | ~300 MB |
| E4 (per tag, 23 rules) | 300 blocks: 3 s | ~40 s | ~8-10 min (24 jobs) | ~200-300 MB |
| E5 (per tag, 11 rules) | 300 blocks: 2 s | ~20 s | ~2-3 min (12 jobs) | ~200-300 MB |
| E6 (per tag, 6 rules) | 2100 blocks: 6 s | ~12 s | ~2 min (12 jobs) | ~280 MB |

Total for all full runs: well under 30 min with 2 processes; memory with 2 processes < 1 GB.

## 4. Outputs

| file | content |
|---|---|
| `out/vssT_E1_<tag>.json` | per rule, per sampled block: g, pi, mu applied, mu in the formula, one-step oracle (wM and fixed-point reference, grid and analytic, Phi and Euclidean), frozen-ensemble oracle over K = 32 blocks (Delta and u units), T = 8 continuation oracle, C, Q (Euclidean and Phi), psi and mu_hat for a0 in {1, 3.8}, f in {0.9, 1.0}, tau_m in {2.5, 3.8} |
| `out/vssT_E2_<tag>.json/.npz` | tau_a, tau_m, cross sums (Euclid / Phi), N, f, h_eff; npz: lag profiles |
| `out/vssT_E3_<tag>.json/.npz` | binned tables (g x pi for the three pi windows), regressions; npz: per-frame g, pi_x, pi_xM, pi_r, noise, signal, fluctuation |
| `out/vssT_E4_<tag>.json/.npz` | per rule: L1 / L2 (whole, first 5 s, last 10 %; pc windows), excess summaries, zero-dwell, divergence; `loss` = predicted E[g^2]/E[g], E[g^2 rho]/E[g] and measured ratios; npz: Phi-excess every block, mu_app, mu_state, psi, rho per rule |
| `out/vssT_E5_<tag>.*` | as E4 for the GS weight variants |
| `out/vssT_E6_<tag>.*` | as E4; npz also psi_B0, psi_geo, omega, excess relative to the B0 and geo fixed points |
| `RESULTS_vssT.md` | `python vssT.py report` (tables only; more negative dB is better) |

## 5. Tags

`<class>_s<seed>`: `G` Gaussian, `PS` power steps (+20 dB 10-15 s, -20 dB 20-25 s), `BU` bursts (+20 dB, 50 ms,
once per second), `SM` sinusoidal log-power modulation (period 1 s, +-10 dB), `II` isolated impulses on a
Gaussian background (one per 0.5 s +- 0.1 s, amplitude 30x background std; `II300_s0` sets 300x),
`a145` / `a120` SaS alpha 1.45 / 1.2, `pc` path change at 15 s (alpha 1.45 input, as `mfblms.build_pc`).
For `pc` the excess is measured against the Wiener solution of the path active in that block.

## 6. Rule names (E4-E6)

| name | meaning |
|---|---|
| `SF mu.05`, `SF mu.1` | SF direction, fixed mu |
| `SF+ramp`, `GS` | baselines of vss2 (`GS`: mu 0.05 x ramp, omega ramp 0 -> 1 over 4 s) |
| `SF-V2`, `GS-V2 k2 .05/.2` | vss2.Ctrl2 |
| `SF-I` | vssI `SF-I g.026 m.3` |
| `GS-T2 fixed-om` | vssI2 integral rule with weights, fixed ramp omega |
| `SF 1/g mu0` | mu0 / max(g, g_min) |
| `SF 1/(g rho) mu0` | mu0 / (max(g, g_min) rho), rho = max(1 - pi (1 - psi), 0.1), psi from a passive direct estimator |
| `SF-D <corr> gam<g> a0 <a0>` | direct form: C, Q EMA(0.05) on u = Delta/max(g, g_min), L = 4, beta_d = 0.1; mu_state <- (1-gamma) mu_state + gamma clip(mu_hat, 0, 1); start 0.3; f = 0.9, tau_m = 2.5, tau_a = 3.8; applied = mu_state x {1 (`none`), 1/g (`g`), 1/(g rho) (`grho`)} |
| `GS-D g gam0.03 a0 1` | direct form on the GS direction with the fixed ramp omega |
| `GS-D w:<none/C/Q/both> a0 <a0>` | E5: same, with bin weights Lam/Pbar on C and/or Srr/Pbar on Q |
| `B0 mu.1 +track`, `B0-D g +track` | E6: B0 direction (omega = 0), fixed mu or direct step; psi_B0 and psi_geo tracked (geo not applied) |
| `Switch` | E6 closed loop: omega <- EMA_0.05(psi_geo / (psi_geo + psi_B0)), direct step on the applied W(0.5, omega) direction |
