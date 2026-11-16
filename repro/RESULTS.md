# Table I reproduction attempt (atsip/repro)

Run: `MPLBACKEND=Agg python3 anc.py --config repro/config_<tag>.json`, one config per noise file.
Configs: `repro/config_a145.json` (alpha 1.45), `config_a165.json` (1.65), `config_a185.json` (1.85), `config_mix.json` (1.65_1.45_1.85).
Column order is assumed to match the paper's Table I as 1.45, 1.65, 1.85, mixed.

## Metric definitions (from anc.py)

- ANR per frame: `anr[k,:] = 20*log10(e_m/(d_m+1e-12) + 1e-12)`, where `e_m` and `d_m` are EMAs (factor 0.999) of |e| and |yP| (anc.py:244-249).
- Reported "mean ANR" (`ANR_mean`): `np.nanmean(anr[:, i])` over the whole curve, including the convergence transient (anc.py:321). `steady_pct` is used only for the ratio statistics, not for ANR (anc.py:253-254).
- Timing: `update_mean` = accumulated `perf_counter()` time around each `algos[i].update(...)` call divided by call count (anc.py:227-229, 322). No RTF is computed anywhere in the code.
- RTF in the table below is my own derivation, not a code output: `update_mean * fs` (total update time / 30 s of audio). It covers only the update call, so it is not comparable to the paper's RTF.

## Results (whole-curve mean ANR, dB)

| Algorithm | 1.45 | 1.65 | 1.85 | mixed | avg | RTF (derived) |
|---|---|---|---|---|---|---|
| Prop (Freq_FxNLMS_CohVSS) | -27.03 | -28.19 | -28.77 | -28.28 | -28.06 | 0.056 |
| Paper Prop | -27.19 | -28.11 | -28.63 | -28.18 | -28.03 | 0.330 |
| LSN_FxlogLMS_plus (placeholder params) | -6.93 | -2.75 | -0.84 | -3.13 | -3.41 | 0.055 |
| Paper LSN_FxlogLMS+ | -26.24 | -27.26 | -27.81 | -27.26 | -27.14 | 0.246 |
| MFxLCH | -26.16 | -26.85 | -27.87 | -27.23 | -27.03 | 0.060 |
| Paper MFxLCH | -25.26 | -26.34 | -27.44 | -26.81 | -26.46 | - |
| CSS_FxAPCMAF | -24.95 | -26.21 | -27.15 | -26.77 | -26.27 | 0.167 |
| Paper CSS_FxAPCMAF | -24.95 | -26.21 | -27.15 | -26.76 | -26.27 | - |
| FxNLMS | -24.46 | -26.32 | -27.75 | -26.54 | -26.27 | 0.068 |
| Paper FxNLMS | -24.32 | -25.76 | -26.88 | -25.94 | -25.73 | - |
| FreqFxLMS (no paper row) | -25.67 | -26.36 | -27.01 | -26.45 | -26.37 | 0.037 |

## Match status (|ours - paper| <= 0.1 dB)

- Prop: 1.65 (-0.08), mixed (-0.10), avg (-0.03) match. 1.45 (+0.16) and 1.85 (-0.14) do not.
- CSS_FxAPCMAF: all five values match (within 0.01 dB). This also supports the column order and the whole-curve metric.
- FxNLMS: no match. Differences are -0.14, -0.56, -0.87, -0.60 dB, avg -0.53 dB.
- MFxLCH: no match. Differences are -0.90, -0.51, -0.43, -0.42 dB, avg -0.57 dB.
- LSN_FxlogLMS_plus: no match, and the run is not meaningful. The config has no LSN entry. I used mu 0.1 and G_ml 5.0 (copied from FxlogNLMS_plus in config.json, placeholder). It is unstable or badly tuned in this setup, with ANR around -1 to -7 dB. The paper's LSN parameters are unknown.
- Paper RTF (Prop 0.330, LSN 0.246) cannot be checked. The code does not compute RTF, and my derived number measures only the update call.

## Notes

- The secondary IR (406 taps) is trimmed to 128 taps by `pad_or_trim` (anc.py:175). The primary IR (422 taps) is zero-padded to 512 (anc.py:174).
- Freq_FxNLMS_CohVSS: the class was copied verbatim (checked with diff) into `algorithms/algorithms.py`. Only the `adv` and `AlgoBase` definitions were not copied, because the file already has identical ones. Params are as specified, with `mu_scale_max` 1.3 (class default is 1.6).
- The class's update uses `mu_ = 2.0 * mu * BlockSize/M`, while FreqFxLMS uses `mu * BlockSize/M`. The factor 2 is not documented.
- CSS_FxAPCMAF has no `mu` in config.json. anc.py defaults to 0.1, but the update ignores `mu` and uses mu1/mu2.
- FxNLMS: mu 0.3, params {} (from config.json). MFxLCH: mu 0.025, rho 10, lmd_m 0.999, dlt2 0.01 (from config.json).
- Wall clock: each run took about 20 s (two runs in parallel). The full set took about 1 to 2 min.
