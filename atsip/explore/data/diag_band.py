"""S1b band diagnostics for 16 kHz paths, inputs, and pre-divergence error."""
import numpy as np
from scipy.signal import periodogram

FS = 16000
EDGES = (0, 60, 100, 250, 500, 1000, 2000, 4000, 8000)
BANDS = tuple(f"{a}–{b}" for a, b in zip(EDGES, EDGES[1:]))


def masks(freq):
    for i, (lo, hi) in enumerate(zip(EDGES, EDGES[1:])):
        yield (freq >= lo) & ((freq <= hi) if i == len(BANDS) - 1 else (freq < hi))


def error_peak(error, end):
    """Diagnose the last finite second before the failed block (or available prefix)."""
    values = error[max(0, end - FS):end]
    if not values.size or not np.isfinite(values).all() or not np.any(values):
        return {"band": None, "hz": None, "samples": int(values.size)}
    freq, power = periodogram(values, fs=FS, window="hann", detrend=False, nfft=FS)
    peak = int(np.argmax(power))
    return {"band": next(name for name, mask in zip(BANDS, masks(freq)) if mask[peak]),
            "hz": float(freq[peak]), "samples": int(values.size)}


def path_rows(primary, secondary, length):
    nfft = 16000
    freq = np.fft.rfftfreq(nfft, 1 / FS)
    p, s, model = [np.fft.rfft(x, nfft) for x in (primary, secondary, secondary[:length])]
    phase = np.abs(np.rad2deg(np.angle(model * np.conj(s))))
    for name, mask in zip(BANDS, masks(freq)):
        yield (name, 20 * np.log10(np.mean(np.abs(s[mask]))), 20 * np.log10(np.mean(np.abs(p[mask]))),
               float(np.max(phase[mask])), 100 * np.mean(phase[mask] > 90))


def input_fractions(samples):
    freq, power = periodogram(samples, fs=FS, window="hann", detrend=False)
    return [100 * np.sum(power[mask]) / np.sum(power) for mask in masks(freq)]


def report_tables(final):
    """Rows for the S1b report, using the runner's path scale and excitation."""
    paths = [(pair, length, band, f"{s:.2f}", f"{p:.2f}", f"{phase:.2f}", f"{bad:.2f}")
             for pair in range(1, 9) for length in final.PILOT_LS
             for band, s, p, phase, bad in path_rows_from_pair(final, pair, length)]
    inputs = []
    for cls in ("G", *final.REAL):
        for seed in final.PILOT_SEEDS:
            for stage, signal in (("前", final._excitation(cls, seed)), ("後", final.excitation(cls, seed))):
                inputs.append((cls, seed, stage, *(f"{v:.4f}" for v in input_fractions(signal))))
    return paths, inputs


def path_rows_from_pair(final, pair, length):
    responses = final.load_pair(pair)
    return path_rows(responses["P"], responses["S"], length)


def main():
    import csv
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import final

    paths, inputs = report_tables(final)
    out = csv.writer(sys.stdout, lineterminator="\n")
    out.writerow(("pair", "Ls", "band_hz", "S_db", "P_db", "phase_max_deg", "phase_gt_90_pct"))
    out.writerows(paths)
    out.writerow(("input", "seed", "highpass", *BANDS))
    out.writerows(inputs)


if __name__ == "__main__":
    main()
