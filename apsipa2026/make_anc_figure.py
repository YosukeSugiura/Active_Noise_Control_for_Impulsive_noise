"""Reproduce a 30-ms ANC overview comparison from archived numerical signals.

Usage (run from any directory):
    python make_anc_figure.py
    python make_anc_figure.py --spectrum-khz 1
    python make_anc_figure.py --spacing-ms 2.5 --nfft 4096

All inputs are resolved relative to THIS script (not the current working directory).
Requires numpy, scipy, matplotlib. Creates figures/ automatically.

Provenance: baseline_inputs.npz is a trimmed copy of an independent ANC
simulation based on a seeded S-alpha-S signal (alpha=1.45, seed=20260815).
The right-hand signal is a CONTROLLED illustrative experiment: one existing
impulse is amplified, and one equal-magnitude, same-sign impulse is inserted.
No rescaling or clipping of the resulting waveform is performed.
Not a reproduction of the published APSIPA numeric results.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.signal import lfilter

from algorithms_20260815 import Freq_FxNLMS_CohVSS

BASE = Path(__file__).resolve().parent


def run_anc(x: np.ndarray, h_primary: np.ndarray, h_secondary: np.ndarray, mu: float = 0.005):
    """Causal feedforward ANC with recovered frequency-domain VSS algorithm."""
    d = lfilter(h_primary, [1.], x)
    r = lfilter(h_secondary, [1.], x)
    params = dict(
        M=512, BlockSize=128, c_impulse=h_secondary, eps=1e-6, rg_alpha=.05,
        coh_pivot=.35, mu_scale_min=.6, mu_scale_max=1.3,
        boost_power=.6, suppress_power=1.6, coh_gain=1.5,
        coh_weight_mix=.6, coh_weight_power=1.6,
        coh_weight_min=.55, grad_norm_mix=.45,
    )
    alg = Freq_FxNLMS_CohVSS(params)
    w = np.zeros(512, dtype=np.float64)
    x_ring = np.zeros(512)
    u_ring = np.zeros(len(h_secondary))
    ix = iu = 0
    e = np.zeros(len(x), dtype=np.float64)
    for k, sample in enumerate(x):
        x_ring[ix] = sample
        x_history = np.concatenate((x_ring[:ix+1][::-1], x_ring[ix+1:][::-1]))
        u = float(w @ x_history)
        u_ring[iu] = u
        u_history = np.concatenate((u_ring[:iu+1][::-1], u_ring[iu+1:][::-1]))
        e[k] = d[k] + float(h_secondary @ u_history)
        alg.update(w, e[k], None, np.array([sample]), mu)
        ix = (ix + 1) % 512
        iu = (iu + 1) % len(h_secondary)
    if not np.isfinite(e).all():
        raise ArithmeticError('ANC produced a non-finite error signal')
    return r, e


def main():
    parser = argparse.ArgumentParser(description='Two same-sign impulses, 30-ms ANC overview')
    parser.add_argument('--spacing-ms', type=float, default=12.5,
                        help='Delay of the second impulse after the first (default: 12.5 ms)')
    parser.add_argument('--gain', type=float, default=50.,
                        help='Strength multiplier of the first existing impulse (default: 50)')
    parser.add_argument('--spectrum-khz', type=float, default=8.,
                        help='Displayed maximum frequency in kHz (e.g. 1 for peak/notch visibility)')
    parser.add_argument('--nfft', type=int, default=480,
                        help='FFT size; >480 uses zero-padding only, not extra data')
    args = parser.parse_args()

    with np.load(BASE / 'baseline_inputs.npz') as src:
        fs = int(src['fs'])
        frame = int(src['frame'])
        right_start = int(src['right_start'])
        p1 = int(src['first_impulse_index'])
        x = np.array(src['x_prefix'], dtype=np.float64, copy=True)
        hP = np.asarray(src['primary'], dtype=np.float64)
        hS = np.asarray(src['secondary'], dtype=np.float64)
        Lr = np.array(src['left_r'], copy=True)
        Le = np.array(src['left_e'], copy=True)
        alpha = float(src['alpha'])
        seed = int(src['seed'])

    if frame != int(round(.030 * fs)) or len(Lr) != frame or len(Le) != frame:
        raise ValueError('Baseline must contain 30-ms / 480-sample windows')
    if not (0 < args.spacing_ms < 1000 * (right_start+frame-1-p1) / fs):
        raise ValueError('spacing-ms would put the second impulse outside the 30-ms frame')
    if not np.isfinite(args.gain) or args.gain <= 0:
        raise ValueError('gain must be positive and finite')
    if not (0 < args.spectrum_khz <= fs / 2000):
        raise ValueError('spectrum-khz must lie within the Nyquist frequency')
    if args.nfft < frame:
        raise ValueError('nfft must be at least 480; the waveform window is always 480 samples')

    p2 = p1 + round(args.spacing_ms * fs / 1000)
    if p2 == p1 or p2 >= right_start + frame:
        raise ValueError('Second pulse must be distinct and inside the window')
    orig_first = float(x[p1])
    x[p1] *= args.gain
    x[p2] = x[p1]  # identical sign AND identical amplitude
    Rr_all, Re_all = run_anc(x, hP, hS)
    Rr = Rr_all[right_start:right_start+frame]
    Re = Re_all[right_start:right_start+frame]

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Nimbus Roman', 'Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'axes.unicode_minus': False,
        'font.size': 16, 'axes.labelsize': 19,
        'xtick.labelsize': 15, 'ytick.labelsize': 15, 'pdf.fonttype': 42,
    })
    blue, orange = '#1467D7', '#E97024'
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.5),
                            gridspec_kw={'height_ratios': [1.12, 1.1]})
    fig.subplots_adjust(left=.085, right=.982, top=.915, bottom=.19, wspace=.25, hspace=.39)
    time_ms = np.arange(frame) / fs * 1000
    freq_khz = np.fft.rfftfreq(args.nfft, 1/fs) / 1000
    hann = np.hanning(frame)

    for col, (r, e, title) in enumerate([
        (Lr, Le, 'Impulsive interval'),
        (Rr, Re, 'Two same-sign impulses'),
    ]):
        ax_time, ax_freq = axes[:, col]
        ax_time.plot(time_ms, r, color=blue, lw=1.35, alpha=.95)
        ax_time.plot(time_ms, e, color=orange, lw=1.55, alpha=.9)
        ax_time.set(xlim=(0, 30), xlabel='Time [ms]', ylabel='Amplitude')
        ax_time.set_title(title, fontsize=23, pad=11)
        ax_time.grid(alpha=.21)
        if col == 0:
            ax_time.set_ylim(-830, 830)
            ax_time.set_yticks([-800, -400, 0, 400, 800])
        else:
            lim = np.ceil(max(np.max(np.abs(r)), np.max(np.abs(e))) / 500) * 500 + 250
            ax_time.set_ylim(-lim, lim)
            for impulse in (p1, p2):
                ax_time.axvline((impulse-right_start)/fs*1000, ls=':', lw=.9,
                                color='0.6', zorder=1)

        mag_r = 20*np.log10(np.abs(np.fft.rfft(r * hann, n=args.nfft))+1e-8)
        mag_e = 20*np.log10(np.abs(np.fft.rfft(e * hann, n=args.nfft))+1e-8)
        ax_freq.plot(freq_khz, mag_r, color=blue, lw=1.6)
        ax_freq.plot(freq_khz, mag_e, color=orange, lw=1.65, alpha=.94)
        ax_freq.set(xlim=(0, args.spectrum_khz), ylim=(-36, 112),
                    xlabel='Frequency [kHz]', ylabel='Magnitude [dB]')
        ax_freq.grid(alpha=.22)

    fig.legend(handles=[
        Line2D([0], [0], color=blue, lw=2.4, label='Filtered reference'),
        Line2D([0], [0], color=orange, lw=2.4, label='Error signal'),
    ], loc='lower center', ncol=2, frameon=False, fontsize=19,
        bbox_to_anchor=(.52, .045), columnspacing=2.8, handlelength=3)

    out = BASE / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    name = (f'anc_same_sign_{args.spacing_ms:g}ms_gain{args.gain:g}_'
            f'{args.spectrum_khz:g}khz_nfft{args.nfft}').replace('.', 'p')
    png, pdf = out / f'{name}.png', out / f'{name}.pdf'
    fig.savefig(png, dpi=300, facecolor='white')
    fig.savefig(pdf, facecolor='white')
    plt.close(fig)

    print(f'Input source: baseline_inputs.npz (SαS alpha={alpha}, seed={seed})')
    print(f'30-ms frame: {frame} samples, fs={fs} Hz')
    print(f'Impulses at {(p1-right_start)/fs*1000:.3f}, {(p2-right_start)/fs*1000:.3f} ms')
    print(f'Input impulse amplitude: {orig_first:g} -> {x[p1]:g}; second={x[p2]:g}')
    print(f'Expected comb spacing: {1000/(args.spacing_ms):g} Hz')
    print(f'FFT-bin spacing: {fs/args.nfft:g} Hz; true window-limited resolution ~{fs/frame:g} Hz')
    print('Saved:', png)
    print('Saved:', pdf)


if __name__ == '__main__':
    main()
