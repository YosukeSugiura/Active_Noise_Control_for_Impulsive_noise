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

---

# protoC2: Sec. 11 estimator variants (same script, new subcommands c3 / c2x / report2)

Spec: THEORY_vss2.md Sec. 11 (with Sec. 10.1-10.3). Only the SF-Y rules with the 1/g correction are varied;
the protoC rules and outputs are unchanged (check 9 compares all 48 protoC SF-Y/SF-E rules bitwise against the
protoC.py of commit 1b0f256, loaded with `git show`; run the check inside the git checkout).

## 6. Self-check (11 s, ~200 MB)

```bash
python protoC.py check      # checks 1-8 (protoC) + 9-14 (protoC2); must end with ALL CHECKS PASSED
```

New checks: (9) all protoC SF-Y/SF-E rules bitwise == 1b0f256 (w, e, all traces, 200 blocks);
(10) `SF-Y est91 <t> g <gam>` == `SF-Y <t> g <gam>` bitwise; (11) estL4 forms (lbar = 4) and estX forms reproduce
(C, Q), a = 0 limits, geometric sum at lam = 1 equals 4, rho grid vs roots < 1e-4; (12) estX det-keep logic;
(13) X_true scale: frozen w = 0 on G_s0, |E u|^2 / X_true = kappa_y (0.93, cos 0.997); (14) the shadow estimators
of the c2x hook reproduce each rule's own X_hat, N_hat bitwise.

## 7. Smoke runs (about 3 min in total; max 2 processes)

```bash
python protoC.py c3 --tags G_s0,BU_s0 --nb 600 --procs 2          # 16 rules each, 5 s
python protoC.py c3 --tags pc_s0 --nb 2600 --procs 1 --rules "SF-Y estX delayopt g gam.03;SF-Y est91 delayopt g gam.03"
python protoC.py c2x --tags G_s0 --nb 2000 --procs 1              # 13 drivers, oracle on: 59 s
python protoC.py c2x --tags pc_s0 --nb 2600 --procs 1             # 97 s (run beside the G_s0 one, not more)
python protoC.py report2 --smoke     # -> out/protoC_smoke/RESULTS_protoC2_smoke.md
```

Measured: c3 pc_s0 2600 blocks, 16 rules: 17.6 s, peak RSS 220 MB.

## 8. Full runs

Run one after the other (each uses 2 processes):

```bash
python protoC.py c3 --procs 2      # G, BU, a145, pc x seeds 0,1,2 = 12 jobs, 16 rules each
python protoC.py c2x --procs 2     # G_s0, pc_s0; 12 estimator rules + SF+ramp as drivers
python protoC.py report2           # -> atsip/explore/RESULTS_protoC2.md
```

Seeds / inputs are CLI-settable: `--classes G,BU,a145,pc --seeds 0,1,2` (c3 default) and
`--classes G,pc --seeds 0` (c2x default), or `--tags`. Then commit `RESULTS_protoC2.md` only.

Estimated full runtime: c3 ~25 s per job -> ~3 min for 12 jobs with 2 procs; c2x ~2 min (G_s0) and
~2.5 min (pc_s0) in parallel; report2 < 5 s. Memory < 300 MB per process. Total under 10 min.

Resume: c3 behaves as c1 (per-rule rewrite of `out/protoC/protoC_C3_<tag>.json/.npz`, resume at the first missing
rule, `--force` recomputes). c2x writes `out/protoC/protoC_C2X_<tag>.json/.npz` after every driver but restarts a
tag from the first driver if killed (no resume within a tag); finished tags are skipped.

Constants added (stored as `x2_*` in `par` of the C3 / C2X files): `--kappa` [1] (kappa of Sec. 11.2),
`--estx-a applied|state` [applied] (a from the applied mu, as written in Sec. 11.2, or from mu_state),
`--det-rel` [1e-3], `--no-aq-cap` (the a of the estX Q form is otherwise capped at h_eff mu_fmax), `--no-oracle`
(c2x without the continuation argmin).

## 9. Outputs (protoC2)

| file | content |
|---|---|
| `out/protoC/protoC_C3_<tag>.json/.npz` | as C1; estX rules also have traces `a`, `rho`, `detkeep` and `frac_detkeep` |
| `out/protoC/protoC_C2X_<tag>.json` | per driver: `xstats` (per phase: median X_hat/X_true, fraction X_hat = 0, median N_hat/N_true at probes, median X_true, N_true; for `own`, est91, estL4, estX), `probe` (i, Ntrue, Qfrz), oracle records as C2 (SF-Y drivers) |
| `out/protoC/protoC_C2X_<tag>.npz` | per driver, every block: `i`, `Xtrue`, `ex0`, `<est>_X`, `<est>_N` (shadows), `own_X`, `own_N` |
| `RESULTS_protoC2.md` | `python protoC.py report2`: C3 tables in the C1 format, estX det-keep fractions, C2X X_hat/X_true and N_hat/N_true tables per phase (8-63, 64-255, 256-1023, >=1024 blocks; pc: >=1024 ends at 15 s, plus after 15 s), tracking summary |

Rule names: `SF-Y <est91|estL4|estX> <delayopt|onestep> g gam<.03|.1>`; controls `SF-Y delayopt g gam.03`
(= protoC, identical to est91 delayopt g gam.03), `SF-V2`, `SF-D g gam0.03 a0 1`, `SF+ramp`.

---

# protoC3: early phase from theory (A) and adaptive smoothing with restarts (B) (subcommands c4 / c4x / report3)

Spec: the protoC3 task (THEORY_vss2.md Sec. 10-11; RESULTS_protoC2.md C2X: X_hat, N_hat far too large in blocks
8-63 and after the path change). Only the SF-Y rules with the 1/g correction and the delayopt target are varied.
All existing rules stay bitwise unchanged (check 15 compares every protoC / protoC2 rule with the protoC.py of
commit ac8eb5b, loaded with `git show`; run the check inside the git checkout).

Variants (suffix of `SF-Y <est91|estX> delayopt g gam<.03|.1>`; no suffix = the protoC2 rule):
- `early`: internal step (mu_state) = mu_early for the first n_e blocks after the start and after every restart,
  then the delayopt EMA from mu_state = mu_early. mu_early = argmin of the delayopt model with N = 0 and the
  injection mu^2 [1 + (2tau_m-1) f] h_eff^2 x (`--early-k full`, the spec; `excl1` = without the '1', as the
  delayopt target's default), T = 8, 1e-3 grid: **0.678** with the default constants (a = 0.161 = a*, i.e. the
  delayed-LMS optimum; `excl1` gives 0.694). Printed by `check`, `c4`, `c4x`.
  n_e: `--n-early N` fixes it; default 0 = trust rule: the first block with the estimator ok, >= 2L blocks since the
  (re)start, Q <= `--trust-q` [2] x mean |u|^2 of the last L blocks and that mean >= `--trust-dec` [0.5] x the mean of
  the L blocks before; at most `--ne-max` [64].
- `rs`: C and Q EMAs with beta_k = max(`--beta-min` [0.05], 1/(k+1)), k = updates since the last (re)start.
  Restart: |u_n|^2 > `--r-thr` [4] x Q_{n-1} for `--rs-d` [2] consecutive blocks (`--rs-kmin` [0]: min. Q updates
  since the last restart before detection). On a restart: C = Q = None, ubar and the lag-L buffer rebuilt from the
  d - 1 previous trigger blocks (C resumes L - d + 1 blocks later, Q at once), estX's kept (X, N) cleared,
  mu_state = mu_early (with early) or kept (rs alone; frozen until C is available again).
- New input `pcb` (path change and back): alpha 1.45 as pc (same x per seed), primary path -> the pc path at 10 s,
  back to the original at 20 s. Implemented in protoC (`data_pcb`) with a 6-line extension of `vssT_lib.path_of`
  (optional `data['pseg']`; unchanged for every other input, checked).

## 10. Self-check (22 s, ~200 MB)

```bash
python protoC.py check      # checks 1-14 + 15-20; must end with ALL CHECKS PASSED
```

New checks: (15) all 64 protoC / protoC2 rules bitwise == ac8eb5b, path_of unchanged without `pseg`;
(16) SFY3 with early = rs = off == the protoC2 rule (bitwise); (17) YEstRS: running mean then EMA, no restart on
stationary data, a single spike ignored, a step restarts after d blocks, C resumes L - d + 1 blocks later;
(18) mu_early X-invariant (printed with a* / h_eff and the excl1 value), early phase with fixed n_e, trust-rule
handover on G_s0; (19) pcb data segments == pc_s0 / original, path_of 0/1/0.

## 11. Smoke runs (about 1.5 min; max 2 processes)

```bash
python protoC.py c4 --tags G_s0,BU_s0 --nb 600 --procs 2                       # 13 rules each, 5 s
python protoC.py c4 --tags pc_s0,pcb_s0 --nb 3000 --procs 2 \
  --rules "SF-Y estX delayopt g gam.03;SF-Y estX delayopt g gam.03 rs;SF-Y estX delayopt g gam.03 early+rs;SF-Y est91 delayopt g gam.03 early+rs"   # 8 s
python protoC.py c4x --tags G_s0 --nb 2000 --procs 1                            # 10 drivers, 17 s
python protoC.py c4x --tags pc_s0 --nb 2600 --procs 1                           # 22 s (may run beside the G_s0 one)
python protoC.py report3 --smoke     # -> out/protoC_smoke/RESULTS_protoC3_smoke.md
```

(pc / pcb L1 windows need full-length runs; NaN in smoke by design.)

## 12. Full runs

Run one after the other (each uses 2 processes):

```bash
python protoC.py c4 --procs 2      # G, BU, PS, a145, a120, pc, pcb x seeds 0,1,2 = 21 jobs, 13 rules each
python protoC.py c4x --procs 2     # G_s0, pc_s0; the 10 SF-Y drivers (no oracle)
python protoC.py report3           # -> atsip/explore/RESULTS_protoC3.md
```

Inputs / seeds: `--classes G,BU,PS,a145,a120,pc,pcb --seeds 0,1,2` (c4 default), `--classes G,pc --seeds 0` (c4x
default; c4x refuses pcb), or `--tags`. Then commit `RESULTS_protoC3.md` only (nothing under `out/`).

Estimated full runtime: c4 ~20-25 s per job (measured: pcb_s0 full length, 2 rules, 5.6 s incl. data preparation)
-> ~4-5 min for 21 jobs with 2 procs; c4x ~35 s per tag (both in parallel); report3 < 5 s. Peak RSS ~230 MB per
process (measured, c4 full length). Total under 10 min.

Resume: c4 as c1 / c3 (`out/protoC/protoC_C4_<tag>.json/.npz`, rewritten after every rule, resumes at the first
missing rule; a stored `par` that differs -> recomputed from scratch; `--force` recomputes). c4x writes
`out/protoC/protoC_C4X_<tag>.json/.npz` after every driver, restarts a tag from the first driver if killed.

## 13. Outputs (protoC3)

| file | content |
|---|---|
| `out/protoC/protoC_C4_<tag>.json/.npz` | as C1 / C3; `par` with `x3_*` (incl. `x3_mu_early`), `changes` (true change blocks); per SFY3 rule `handover` (blocks where an early phase ended), `early_frac`, `rs` (restart blocks, detection delays per change, false restarts, per 30 s); pcb: `pcb` (L1 and excess over 10-12, 15-20, 20-22, 25-30 s); npz traces also `early`, `restart` |
| `out/protoC/protoC_C4X_<tag>.json/.npz` | as C2X without the oracle; `xstats` with pc split into +0-63 / +64-255 / +256.. blocks; `handover`, `rs` per driver; npz also `early`, `restart`, `mu_state` |
| `RESULTS_protoC3.md` | `python protoC.py report3`: C4 tables in the C1 format, pcb change-window table, early / restart table, C4X own X_hat / X_true and N_hat / N_true per phase, restart times vs the true change |

Rule names: `SF-Y <est91|estX> delayopt g gam.03 [early|rs|early+rs]`, `SF-Y <est91|estX> delayopt g gam.1 early+rs`;
controls `SF-Y estX delayopt g gam.03` (protoC2 best, = the estX base rule above), `SF-V2`, `SF+ramp`,
`SF-D g gam0.03 a0 1`.

---

# protoC4: restart statistics that do not fire on bursts / SaS impulses (subcommands c5 / report4)

Spec: the protoC4 task (RESULTS_protoC3.md: the old `rs` statistic fires 22-28 times per 30 s on BU and a120, ~10 on
a145; THEORY_vss2.md 10.5). Only new rule names are added; every existing rule stays bitwise unchanged (check 21
compares all 77 protoC / protoC2 / protoC3 rules with the protoC.py of commit 0073387, loaded with `git show`, on
G_s0 and, for the SFY3 rules incl. the old rs restarts, on a120_s0; run the check inside the git checkout).

New variants (suffix of `SF-Y <est91|estX> delayopt g gam.03`; class SFY4 / YEstRSX; same restart actions as `rs`:
C, Q, k, ubar / delayed buffer reset, estX's kept (X, N) cleared, mu_state = mu_early and a new early phase):
- `early+rsN` (the spec): z_n = sum e^2 / sum r^2 over the M = 512-sample error window of block n (the errors in the
  direction's error frame, via Parseval of Ctx.E; r = `data['r']` = hS * x, the filtered reference; the method's
  R_n = rfft(hS, N) x rfft(x frame) uses the same hS, so r is the method's filtered reference, as a linear
  convolution). zbar = EMA with beta_z = `--z-beta` [0.05], never reset, started with z_0; restart when
  z_n > `--z-thr` [4] x zbar_{n-1} for `--z-d` [2] consecutive blocks.
- `early+rsN2` (added): rsN with sum r^2 over the 2M samples ending at the block end. The errors in the window depend
  on r over 2M - 1 samples (M-tap filter); with the M window, z_n jumps for ~M/B = 4 blocks right after an impulse
  leaves the r window while its filtered tail is still in e (seen in smoke: rsN 15-36 false restarts per 30 s on
  a120 / a145 / II300 / pc, 0 on BU).
- `early+rsG` (cheap alternative): |y_n|^2 / g_n^2 (no floor on g) against its own EMA (adaptive beta, started and
  reset together with Q), `--r-thr`, `--rs-d` as rs. Note: the old statistic |u_n|^2 = |y_n|^2 / max(g_n, 0.25)^2
  already divides the power by g^2 above the floor, so rsG differs from rs only for g < 0.25, by its own EMA and by
  the guard.
- rsN / rsN2 / rsG: no restart within `--rs-guard` [64] blocks of the previous restart (trigger counting held at 0).
- Also available (not in the default list): `... rsN`, `rsN2`, `rsG` without early, and the gam.1 forms.

Default c5 rule list (12): `SF-Y <est91|estX> delayopt g gam.03 <early+rsN|early+rsN2|early+rsG>` and the controls
`SF-Y <est91|estX> delayopt g gam.03 <early|early+rs>`, `SF-V2`, `SF+ramp`.

## 14. Self-check (45 s, ~250 MB)

```bash
python protoC.py check      # checks 1-20 + 21-24; must end with ALL CHECKS PASSED
```

New checks: (21) all 77 existing rules bitwise == 0073387 on G_s0 (200 blocks); all protoC3 SFY3 rules bitwise and
same restart blocks on a120_s0 (600 blocks, old rs restarts 4-6 times); (22) YEstRSX 'G' with g = 1 and guard 0 ==
YEstRS bitwise (C, Q, restarts); 'N' on a synthetic z sequence: zbar = EMA used before its update, a single spike
ignored, a step restarts after d blocks, a step inside the guard ignored, a step after it restarts; (23) the error
window energy (Parseval of Ctx.E) and the r window energy equal direct sums; (24) no restart on G_s0 (300 blocks) for
the three variants; on pc_s0 (up to the change + 150 blocks) every restart sets mu_state = mu_early and the early
flag, the statistic exceeds threshold x reference at the restart block, restarts are >= guard apart, and rsN2
detects the change (rsN / rsG restart counts printed).

## 15. Smoke runs (about 30 s; max 2 processes)

```bash
python protoC.py c5 --tags BU_s0,a120_s0,a145_s0,II300_s0 --nb 1250 --procs 2 --n-early 64 --force   # 12 rules, 15 s
python protoC.py c5 --tags pc_s0,pcb_s0 --nb 3000 --procs 2 --n-early 64 --force \
  --rules "SF-Y est91 delayopt g gam.03 early+rsN;SF-Y est91 delayopt g gam.03 early+rsN2;SF-Y est91 delayopt g gam.03 early+rsG;SF-Y est91 delayopt g gam.03 early+rs;SF-Y estX delayopt g gam.03 early+rsN2"   # 10 s
python protoC.py report4 --smoke     # -> out/protoC_smoke/RESULTS_protoC4_smoke.md
```

`--n-early 64`: the protoC3 full run (RESULTS_protoC3.md, `x3_n_early = 64`) used a fixed early length of 64, so the
early / early+rs controls of c5 reproduce the protoC3 numbers. Note: a finished tag (`done`) is skipped even when the
constants differ; use `--force` when changing constants.

## 16. Full runs

```bash
python protoC.py c5 --procs 2 --n-early 64   # G, BU, PS, a145, a120, pc, pcb, II300 x seeds 0,1,2 = 24 jobs, 12 rules each
python protoC.py report4                     # -> atsip/explore/RESULTS_protoC4.md
```

Inputs / seeds: `--classes G,BU,PS,a145,a120,pc,pcb,II300 --seeds 0,1,2` (default) or `--tags`. Then commit
`RESULTS_protoC4.md` only (nothing under `out/`).

Estimated full runtime: ~20 s per job (smoke: 4 jobs x 12 rules x 1250 blocks in 15 s with 2 procs) -> ~5 min for 24
jobs with 2 procs; report4 < 5 s. Peak RSS < 300 MB per process. Resume as c4 (`out/protoC/protoC_C5_<tag>.json/.npz`,
rewritten after every rule, resumes at the first missing rule; a stored `par` that differs -> recomputed from
scratch for unfinished tags; `--force` recomputes).

## 17. Outputs (protoC4)

| file | content |
|---|---|
| `out/protoC/protoC_C5_<tag>.json/.npz` | as C4; `par` also with `x4_*` (z_beta, z_thr, z_d, guard); per SFY3 / SFY4 rule `handover`, `early_frac`, `rs` (restart blocks, delays per change, false restarts, per 30 s); npz traces of SFY4 also `rsstat` (z_n or |y|^2/g^2) and `rsref` (zbar_{n-1} or its EMA) |
| `RESULTS_protoC4.md` | `python protoC.py report4`: C5 tables in the C1 format, pcb change windows, early / restart table (as report3), restart table (false restarts per 30 s with min / max over seeds, total restarts per 30 s, delay and missed count per change) |
