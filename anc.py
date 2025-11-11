# -*- coding: utf-8 -*-
"""Feedforward ANC simulator (FxLMS variants). Minimal header; see README for details."""
from __future__ import annotations

import os
import json
import argparse
from typing import Dict, Any, List

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import perf_counter

# ----------------------------
# Args & profiling
# ----------------------------
ap = argparse.ArgumentParser()
_here = os.path.dirname(os.path.abspath(__file__))
ap.add_argument("--config", default=os.path.join(_here, "config.json"))
args, _ = ap.parse_known_args()

# ----------------------------
# Path resolve helper based on config location
# ----------------------------
CFG_DIR = os.path.dirname(os.path.abspath(args.config))

def resolve(path: str | None) -> str | None:
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.join(CFG_DIR, path)

# ----------------------------
# I/O helpers
# ----------------------------
def _rel(p: str) -> str:
    try:
        return os.path.relpath(p)
    except Exception:
        return p

def save_csv(path: str, arr: np.ndarray, header: str, fmt: str = "%.10f") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, arr, delimiter=",", header=header, comments="", fmt=fmt)
    print(f"[save] CSV: {_rel(path)}")

def save_png(fig: plt.Figure, path: str, label: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[plot] {label}: {_rel(path)}")

# ----------------------------
# Algo registry
# ----------------------------
from algorithms.algorithms import make_algo  # noqa: E402

try:
    from numba import njit
except Exception as exc:
    raise RuntimeError("Numba is required. pip install numba") from exc

# ----------------------------
# Utils
# ----------------------------
def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def pad_or_trim(h: np.ndarray, L: int) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64)
    if h.size == L:
        return h
    if h.size < L:
        out = np.empty(L, dtype=np.float64)
        out[:h.size] = h
        out[h.size:] = 0.0
        return out
    return h[:L].copy()

# ----------------------------
# Single-sample step (TD path + filtered ref)
# ----------------------------
@njit(cache=True, fastmath=True)
def step_td(x_in: float,
            x_vec: np.ndarray,
            W_mat: np.ndarray,
            yb_mat: np.ndarray,
            hP: np.ndarray,
            hS: np.ndarray,
            C: int,
            RB_mat: np.ndarray,
            yC_all: np.ndarray,
            yS_all: np.ndarray) -> float:
    W = x_vec.shape[0]
    Ls = hS.shape[0]
    A = W_mat.shape[0]

    # x shift
    for i in range(W - 1, 0, -1):
        x_vec[i] = x_vec[i - 1]
    x_vec[0] = x_in

    # yP = hP · x
    yP = 0.0
    for i in range(W):
        yP += hP[i] * x_vec[i]

    # yC_all = W_mat * x_vec
    for a in range(A):
        s = 0.0
        row = W_mat[a]
        for i in range(W):
            s += row[i] * x_vec[i]
        yC_all[a] = s

    # yb push
    for a in range(A):
        for j in range(Ls - 1, 0, -1):
            yb_mat[a, j] = yb_mat[a, j - 1]
        yb_mat[a, 0] = yC_all[a]

    # yS_all = hS · yb
    for a in range(A):
        s = 0.0
        for j in range(Ls):
            s += yb_mat[a, j] * hS[j]
        yS_all[a] = s

    # filtered ref r
    up = C if C < Ls else Ls
    r = 0.0
    for i in range(up):
        r += hS[i] * x_vec[i]

    # RB push
    for a in range(A):
        for j in range(W - 1, 0, -1):
            RB_mat[a, j] = RB_mat[a, j - 1]
        RB_mat[a, 0] = r

    return yP

# ----------------------------
# Main
# ----------------------------
def main() -> None:
    cfg = load_cfg(args.config)
    g = cfg["global"]

    fs = int(g["fs"])
    W = int(g["order_control"])          # control length
    C = int(g["order_secondary"])        # secondary length for r
    step = int(g["frame_step"])          # ANR update stride

    do_plot = bool(g.get("plot", g.get("plot_show", True)))

    # input
    noise_path = resolve(g["noise_file"])
    x0 = np.loadtxt(noise_path, ndmin=1).astype(np.float64)
    if x0.ndim > 1:
        x0 = x0[:, 0]
    N = x0.size

    amp_scales = list(g.get("amp_scales", [1.0]))
    trials = len(amp_scales)

    # IR
    hP_file = resolve(g.get("primary_ir_file", "impulse_response/primary_ir.dat"))
    hS_file = resolve(g.get("secondary_ir_file", "impulse_response/secondary_ir.dat"))
    hP = pad_or_trim(np.loadtxt(hP_file, ndmin=1), W)
    hS = pad_or_trim(np.loadtxt(hS_file, ndmin=1), C)
    Ls = hS.size

    # algorithms
    alg_cfg = cfg["algorithms"]
    items = [(k, v) for k, v in alg_cfg.items() if v.get("enabled", True)]
    if not items:
        raise ValueError("No enabled algorithms in config")
    names: List[str] = [k for k, _ in items]
    mus = np.array([float(v.get("mu", 0.1)) for _, v in items], dtype=np.float64)
    params0 = [dict(v.get("params", {})) for _, v in items]
    for p in params0:
        p.setdefault("c_impulse", hS)
        p.setdefault("order_control", W)

    A = len(names)

    # buffers
    ratio_logs = np.full((N, trials), np.nan, dtype=np.float64)
    ratio_ema_logs = np.full((N, trials), np.nan, dtype=np.float64)

    frames = N // step
    anr = np.full((frames, A), np.nan, dtype=np.float64)
    ref = np.zeros(N, dtype=np.float64)
    errs = np.zeros((N, A), dtype=np.float64)

    logs_dir = resolve(g.get("logs_dir", "logs")) or os.path.join(CFG_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # timing accumulators
    algo_time_total = np.zeros(A, dtype=np.float64)
    algo_calls = np.zeros(A, dtype=np.int64)

    # trials
    for t, sc in enumerate(amp_scales):
        x_in = sc * x0
        algos = [make_algo(nm, p) for nm, p in zip(names, params0)]

        W_mat = np.zeros((A, W), dtype=np.float64)
        RB_mat = np.zeros((A, W), dtype=np.float64)
        yb_mat = np.zeros((A, Ls), dtype=np.float64)
        x_vec = np.zeros(W, dtype=np.float64)
        yC_all = np.zeros(A, dtype=np.float64)
        yS_all = np.zeros(A, dtype=np.float64)
        e_m = np.zeros(A, dtype=np.float64)
        d_m = 0.0

        for n in tqdm(range(N), desc=f"trial {t+1}/{trials}", unit="samp", leave=False):
            yP = step_td(float(x_in[n]), x_vec, W_mat, yb_mat, hP, hS, C, RB_mat, yC_all, yS_all)
            e_vec = yP + yS_all

            for i in range(A):
                t0 = perf_counter()
                W_mat[i], _ = algos[i].update(W_mat[i], float(e_vec[i]), RB_mat[i], x_vec, float(mus[i]))
                algo_time_total[i] += (perf_counter() - t0)
                algo_calls[i] += 1

            for idx, algo in enumerate(algos):
                r = getattr(algo, "ratio", np.nan)
                re = getattr(algo, "ratio_ema", np.nan)
                if r == r:
                    ratio_logs[n, t] = r
                    ratio_ema_logs[n, t] = re
                    break

            if t == trials - 1:
                ref[n] = yP
                errs[n, :] = e_vec
                for i in range(A):
                    e_m[i] = 0.999 * e_m[i] + 0.001 * abs(e_vec[i])
                d_m = 0.999 * d_m + 0.001 * abs(yP)
                if (n + 1) % step == 0:
                    k = (n + 1) // step - 1
                    denom = d_m + 1e-12
                    anr[k, :] = 20.0 * np.log10((e_m / denom) + 1e-12)

    # steady stats for ratio
    if np.isfinite(ratio_ema_logs).any():
        steady_pct = float(g.get("steady_pct", 0.20))
        i0 = int((1.0 - steady_pct) * N)
        rows = []
        for t, sc in enumerate(amp_scales):
            r_ss = ratio_logs[i0:, t]
            re_ss = ratio_ema_logs[i0:, t]
            rows.append([sc, np.nanmean(r_ss), np.nanstd(r_ss), np.nanmean(re_ss), np.nanstd(re_ss)])
        rows = np.asarray(rows, dtype=np.float64)
        save_csv(
            os.path.join(logs_dir, "ratio_amp_invariance.csv"),
            rows,
            header="amp_scale,ratio_mean,ratio_std,ratio_ema_mean,ratio_ema_std",
            fmt="%.10f",
        )

        if do_plot:
            t_samp = np.arange(N) / fs
            cmap = mpl.colormaps.get_cmap("tab20")
            cols = [cmap(i / max(trials - 1, 1)) for i in range(trials)]

            fig, ax = plt.subplots(figsize=(9, 4))
            for t, sc in enumerate(amp_scales):
                ax.plot(t_samp, ratio_ema_logs[:, t], label=f"x×{sc}", color=cols[t])
            ax.axvspan(i0 / fs, N / fs, color='k', alpha=0.05, label="steady window")
            ax.set_xlabel("Time [s]"); ax.set_ylabel("ratio_ema"); ax.set_ylim(0, 1.1)
            ax.legend(ncol=3, fontsize=8); ax.grid(True, ls=":")
            save_png(fig, os.path.join(logs_dir, "ratio_amp_timeseries.png"), "ratio-EMA timeseries")

            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.plot(rows[:, 0], rows[:, 3], "o-")
            ax2.set_xscale("log"); ax2.grid(True, ls=":", which="both")
            ax2.set_xlabel("Input amplitude scale (log)"); ax2.set_ylabel("steady ratio_ema mean")
            ax2.set_ylim(0, 1.1)
            save_png(fig2, os.path.join(logs_dir, "ratio_amp_steady.png"), "ratio-EMA steady vs amplitude")

    # ANR outputs
    t_frame = (np.arange(anr.shape[0]) * step) / fs
    header = "time," + ",".join(names)
    save_csv(
        os.path.join(logs_dir, "anr.csv"),
        np.column_stack([t_frame, anr]),
        header=header,
        fmt="%.10f",
    )

    if do_plot:
        if len(names) <= 20:
            cmap2 = mpl.colormaps.get_cmap("tab20")
            alg_cols = [cmap2(i / max(len(names) - 1, 1)) for i in range(len(names))]
        else:
            import colorsys
            A_ = len(names)
            alg_cols = [colorsys.hsv_to_rgb(i / A_, 0.75, 0.9) for i in range(A_)]
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, nm in enumerate(names):
            ax.plot(t_frame, anr[:, i], color=alg_cols[i], label=nm.replace("_", "-"))
        ax.set_xlabel("Time [s]"); ax.set_ylabel("ANR (dB)"); ax.grid(True, ls=":")
        ax.legend()
        ylo, yhi = ax.get_ylim()
        ax.set_ylim(ylo, 0.0)
        ax.set_xlim(t_frame[0], t_frame[-1])
        save_png(fig, os.path.join(logs_dir, "anr.png"), "ANR over time")

    # Terminal summary: mean ANR and mean update time per algorithm
    anr_mean = np.array([np.nanmean(anr[:, i]) for i in range(A)], dtype=np.float64)
    time_mean = algo_time_total / np.maximum(algo_calls, 1)
    print("\n=== Summary (last trial) ===")
    for i, nm in enumerate(names):
        print(f"{nm:16s}  ANR_mean[dB]={anr_mean[i]:8.3f}  "
              f"update_mean={time_mean[i]*1e6:8.3f} µs")

if __name__ == "__main__":
    main()
