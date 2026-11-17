# RESULTS vssI2: weighted-inner-product step rules (delayless FD-FxNLMS)

L1 = ANR mean in dB, lower (more negative) is better. Mean over non-diverged runs; n = runs used, div = diverged runs (NaN in the json). Path change columns: 15-17 s / 25-30 s / whole.

## Observations

- Divergences: 0 of 10 jobs x 8 rules (no NaN runs). Baselines reproduce the vssI run (GS -30.24, SF-I -30.74 on the Gaussian pair), so the table is like-for-like.
- Gaussian and alpha 1.45: GS-T2 (weighted c and Q) is worse than its Euclidean control GS-T2 E (-30.03 vs -30.30; -29.12 vs -29.28). This is the opposite of the Sec. 12 prediction for this rule.
- GS-T2 does not reach SF-I on the stationary groups (SF-I -30.74 Gaussian, -29.41 alpha 1.45). On alpha 1.2 the GS-T2 variants are at or slightly better than SF-I (-28.91 and -28.95 vs -28.77), on 2 seeds.
- Path change: GS-T2 whole -29.38 vs SF-I -29.77 and GS-T2 E -29.55. The 15-17 s window: GS-T2 -18.88, GS-T2 E -19.59, SF-I -20.74. The fixed-om variant gives -14.24 in that window, so its omega ramp is slow on the path change.
- GS-V2T (L=4, chi=0.5) is within 0.01 dB of GS-V2 k2 on the Gaussian pair; on the path change its 15-17 s window is -19.64 vs -18.55 for L=8.
- Omega under the q rules reaches 1.0 by about block 500 on G0 (q is near 0 there). Applied mu of GS-T2 is about 0 by block 500.
- Update-only time per block: GS-V2 0.051 ms, GS-T2 0.087 ms, SF-IE 0.113 ms. The weighted products use FFT spectra and bin weights, so there is no MxM product.
- Caveat: 2-3 seeds per group, so differences below about 0.05 dB are not resolved.

## Gaussian (2)

| rule | whole | first5 | last10 | n | div |
|---|---|---|---|---|---|
| GS | -30.24 | -28.66 | -30.55 | 2 | 0 |
| GS-V2 k2 .05/.2 | -30.23 | -28.97 | -30.51 | 2 | 0 |
| SF-I g.026 m.3 | -30.74 | -29.55 | -30.93 | 2 | 0 |
| SF-IE g.026 m.3 | -30.71 | -29.54 | -30.92 | 2 | 0 |
| GS-T2 | -30.03 | -28.62 | -30.33 | 2 | 0 |
| GS-T2 fixed-om | -30.00 | -28.76 | -30.28 | 2 | 0 |
| GS-T2 E | -30.30 | -29.04 | -30.54 | 2 | 0 |
| GS-V2T | -30.23 | -28.95 | -30.52 | 2 | 0 |

## alpha 1.45 (3)

| rule | whole | first5 | last10 | n | div |
|---|---|---|---|---|---|
| GS | -29.10 | -26.88 | -29.58 | 3 | 0 |
| GS-V2 k2 .05/.2 | -29.24 | -27.84 | -29.60 | 3 | 0 |
| SF-I g.026 m.3 | -29.41 | -28.21 | -29.69 | 3 | 0 |
| SF-IE g.026 m.3 | -29.35 | -28.17 | -29.60 | 3 | 0 |
| GS-T2 | -29.12 | -27.47 | -29.51 | 3 | 0 |
| GS-T2 fixed-om | -29.18 | -27.81 | -29.52 | 3 | 0 |
| GS-T2 E | -29.28 | -27.83 | -29.62 | 3 | 0 |
| GS-V2T | -29.25 | -27.87 | -29.60 | 3 | 0 |

## alpha 1.2 (2)

| rule | whole | first5 | last10 | n | div |
|---|---|---|---|---|---|
| GS | -28.83 | -26.96 | -29.19 | 2 | 0 |
| GS-V2 k2 .05/.2 | -28.90 | -27.43 | -29.24 | 2 | 0 |
| SF-I g.026 m.3 | -28.77 | -27.64 | -28.95 | 2 | 0 |
| SF-IE g.026 m.3 | -28.73 | -27.62 | -28.89 | 2 | 0 |
| GS-T2 | -28.79 | -26.91 | -29.18 | 2 | 0 |
| GS-T2 fixed-om | -28.91 | -27.49 | -29.22 | 2 | 0 |
| GS-T2 E | -28.95 | -27.49 | -29.26 | 2 | 0 |
| GS-V2T | -28.92 | -27.57 | -29.25 | 2 | 0 |

## path change (3)

| rule | 15-17 s | 25-30 s | whole | n | div |
|---|---|---|---|---|---|
| GS | -4.77 | -30.56 | -25.92 | 3 | 0 |
| GS-V2 k2 .05/.2 | -18.55 | -31.71 | -29.39 | 3 | 0 |
| SF-I g.026 m.3 | -20.74 | -31.93 | -29.77 | 3 | 0 |
| SF-IE g.026 m.3 | -21.25 | -31.92 | -29.78 | 3 | 0 |
| GS-T2 | -18.88 | -31.78 | -29.38 | 3 | 0 |
| GS-T2 fixed-om | -14.24 | -31.83 | -29.12 | 3 | 0 |
| GS-T2 E | -19.59 | -31.84 | -29.55 | 3 | 0 |
| GS-V2T | -19.64 | -31.70 | -29.46 | 3 | 0 |

## Step traces (applied mu per direction, and omega for GS rules)

Block indices: [8, 31, 125, 500, 1500, 3700] (non-path), [1870, 1880, 1900, 1950, 2100, 3000] (path change). Values from G0 and pc0.

### G0 (idx [8, 31, 125, 500, 1500, 3700])

| rule | direction | mu at idx | om at idx |
|---|---|---|---|
| GS | mu | 0.1484 0.1438 0.1250 0.0500 0.0500 0.0500 | 0.02 0.06 0.25 1.00 1.00 1.00 |
| GS-V2 k2 .05/.2 | mu | 0.1686 0.3000 0.1375 0.0100 0.0100 0.0100 | 0.33 0.00 0.87 1.00 1.00 1.00 |
| SF-I g.026 m.3 | SF | 0.1686 0.2547 0.2392 0.0245 0.0262 0.0030 |  |
| SF-IE g.026 m.3 | SF | 0.1686 0.3308 0.2666 0.0266 0.0777 0.0076 |  |
| GS-T2 | GS | 0.1686 0.2791 0.2723 0.0000 0.0012 0.0000 | 0.33 0.80 1.00 1.00 1.00 1.00 |
| GS-T2 fixed-om | GS | 0.1686 0.2549 0.2253 0.0027 0.0028 0.0000 | 0.02 0.06 0.25 1.00 1.00 1.00 |
| GS-T2 E | GS | 0.1686 0.3773 0.3687 0.0004 0.0067 0.0000 | 0.33 0.00 0.93 1.00 1.00 1.00 |
| GS-V2T | mu | 0.1686 0.3000 0.1432 0.0100 0.0100 0.0100 | 0.33 0.00 0.86 1.00 1.00 1.00 |

### pc0 (idx [1870, 1880, 1900, 1950, 2100, 3000])

| rule | direction | mu at idx | om at idx |
|---|---|---|---|
| GS | mu | 0.0500 0.0500 0.0500 0.0500 0.0500 0.0500 | 1.00 1.00 1.00 1.00 1.00 1.00 |
| GS-V2 k2 .05/.2 | mu | 0.0106 0.0100 0.0569 0.1616 0.2223 0.0100 | 1.00 1.00 1.00 0.79 0.59 1.00 |
| SF-I g.026 m.3 | SF | 0.0080 0.0103 0.0536 0.1952 0.3408 0.0334 |  |
| SF-IE g.026 m.3 | SF | 0.0677 0.0691 0.1081 0.2190 0.3408 0.0675 |  |
| GS-T2 | GS | 0.0004 0.0017 0.0373 0.1578 0.3408 0.0012 | 1.00 1.00 1.00 0.80 0.20 1.00 |
| GS-T2 fixed-om | GS | 0.0002 0.0013 0.0369 0.1527 0.3408 0.0000 | 1.00 1.00 1.00 1.00 1.00 1.00 |
| GS-T2 E | GS | 0.0009 0.0025 0.0460 0.1877 0.3408 0.0004 | 1.00 1.00 1.00 0.72 0.56 1.00 |
| GS-V2T | mu | 0.0100 0.0100 0.0550 0.1936 0.2239 0.0100 | 1.00 1.00 1.00 0.69 0.59 1.00 |

## Per-block update time (one process, 500 blocks of G0 inputs)

Update only: direction, step, statistics. Weighted products use FFT spectra and bin weights (no MxM product).

| method | ms per block |
|---|---|
| GS-V2 k2 .05/.2 | 0.051 |
| GS-T2 | 0.087 |
| SF-IE g.026 m.3 | 0.113 |
