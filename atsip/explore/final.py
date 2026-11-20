"""Final evaluation on measured paths (EVAL_PLAN.md).  Stage S1: scenario builder, simulators, measures, pilot.

Paths: MeshRIR pairs in data/paths16 (P, S, P_near, P_far, P_<d>cm, S_<d>cm; 16 kHz, common delay removed).
The plant uses the full responses; the controller uses the model S-hat = first Ls taps of S.  P and S of each pair
are scaled so that ||P|| and ||S|| equal those of the default paths (keeps unnormalised baselines in their range).

Scenarios (change at 15 s = block 1875):
  stat                 no change
  RS-near, RS-far      the noise source switches: P changes, S and S-hat stay
  RM-<d>               the error microphone moves d cm: P and S change, S-hat is replaced by the new one
  RM-<d>s              the same with the old S-hat kept (stale)
  EN0, ENm10           bursts at the error microphone at 10 s and 20 s (0 / -10 dB re d), no path change
  post:<scenario>      the paths (and model) after the change, used from the start (level reference for recovery)

Methods: the frozen rule and its variants run in a block simulator with the same arithmetic as vssT_lib.simulate
(without the Phi-excess, which needs the default paths); the external methods of algorithms/algorithms.py run in a
per-sample loop.  Seeds 100-119 are refused until EVAL_FROZEN.json exists and --final is given.

Commands (resumable):
  python final.py check
  python final.py pilot [--procs K] [--smoke]
  python final.py report
"""
import argparse
import copy
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import ablate as ab
import prop2 as q
import prop4 as q4
import prop5 as q5
import protoC as p
from numpy.lib.stride_tricks import sliding_window_view
from scipy.io import wavfile
from algorithms.algorithms import make_algo
from data.diag_band import error_peak

np, T = p.np, p.T
B, M, N, FS, BPS = p.B, p.M, p.N, p.FS, p.BPS
L_SIG = p.O.N_NOISE
CB = 15 * BPS                                  # block of the change
REAL = ("machinegun", "factory1", "destroyerengine", "volvo", "valve", "jackhammer")
SYNTH = ("G", "PS", "BU", "II300", "a145", "a120")
BURSTS = ((10.0, 10.5), (20.0, 20.5))
DIV_GAIN = 1e4                                 # block error power above this x the full-run primary power = divergence
VERSION = 4
HP_HZ = 100.0                                  # operating band: 4th-order Butterworth high-pass on every excitation
RMODES = ("fd",)                               # S1: "td" gave no gain and broke the identity with no-restart; dropped

BASE = {  # name: (default step, parameters) from config.json / repro/config_a145.json
    "FxNLMS": (0.3, {}),
    "Th_FxNLMS": (0.3, {"c1": -200.0, "c2": 200.0}),
    "FxlogNLMS": (0.5, {"G_l": 10000.0}),
    "MFxLMM": (0.001, {"eps_l": 0.001, "gzai_l": 0.05, "lmd_l": 0.99}),
    "FxNFLOC": (0.1, {"Ne": 512, "Nx": 512, "eps": 1e-12, "omega": 0.01}),
    "EVSS_FxeLMS": (0.05, {"alpha": 1.0, "gamma": 0.0001, "k": 0.2, "lmd_e": 0.99}),
    "FreqFxLMS": (0.36, {"M": 512, "BlockSize": 128, "eps": 1e-06}),
    "Freq_FxNLMS_CohVSS": (0.36, {"M": 512, "BlockSize": 128, "eps": 1e-06, "rg_alpha": 0.05, "coh_pivot": 0.35,
                                  "mu_scale_min": 0.6, "mu_scale_max": 1.3, "boost_power": 0.6, "suppress_power": 1.6,
                                  "coh_gain": 1.5, "coh_weight_mix": 0.6, "coh_weight_power": 1.6,
                                  "coh_weight_min": 0.55, "grad_norm_mix": 0.45}),
}
RULES = ("nors", "P5-early", "P5-early-noG", "P4", "TP-always", "ND-always", "SF-fix")


# ----------------------------------------------------------------------------------------------
# Inputs and scenarios
# ----------------------------------------------------------------------------------------------
def guard_seed(seed, final=False):
    if 100 <= int(seed) <= 119 and not (final and (HERE / "EVAL_FROZEN.json").exists()):
        raise SystemExit("seeds 100-119 are reserved for the final run (EVAL_FROZEN.json and --final are required)")


def excitation(cls, seed):
    """Filter each excitation before matching real-audio power to filtered G."""
    x = highpass(_excitation(cls, seed))
    return x * np.sqrt(g_power() / np.mean(x ** 2)) if cls in REAL else x


def highpass(x):
    return p.signal.sosfilt(p.signal.butter(4, HP_HZ, "highpass", fs=FS, output="sos"), x)


def _excitation(cls, seed):
    if cls in REAL:
        fs, wav = wavfile.read(HERE / "data" / "prepared" / f"{cls}.wav")
        assert fs == FS and wav.ndim == 1, (cls, fs, wav.shape)
        off = (int(seed) % 5) * 2 * FS
        x = np.asarray(wav[off:off + L_SIG], float)
        assert len(x) == L_SIG, (cls, len(wav))
        return x
    if cls == "G":
        return p.IC.make_input("A", seed)
    if cls == "PS":
        return p.IC.make_input("B", seed)
    if cls == "BU":
        return p.IC.make_input("C", seed)
    if cls == "II300":
        return T.impulses(seed, 300.0)
    if cls == "a145":
        return p.O.cms_symmetric_stable(1.45, L_SIG, seed)
    if cls == "a120":
        return p.O.cms_symmetric_stable(1.2, L_SIG, seed)
    raise ValueError(cls)


_GP = []


def g_power():
    if not _GP:
        _GP.append(float(np.mean(highpass(_excitation("G", 10)) ** 2)))
    return _GP[0]


def load_pair(k):
    z = np.load(HERE / "data" / "paths16" / f"pair_{k:02d}.npz")
    gp = float(np.linalg.norm(T.BS.hP) / np.linalg.norm(z["P"]))
    gs = float(np.linalg.norm(T.BS.hS) / np.linalg.norm(z["S"]))
    return {key: np.asarray(z[key], float) * (gp if key.startswith("P") else gs) for key in z.files if key != "fs"}


def paths_of(scen, pair):
    """(P0, S0, model source 0), (P1, S1, model source 1) or None."""
    P, S = pair["P"], pair["S"]
    post = scen.startswith("post:")
    s = scen[5:] if post else scen
    if s in ("stat", "EN0", "ENm10"):
        assert not post
        return (P, S, S), None
    if s in ("RS-near", "RS-far"):
        new = (pair["P_" + s[3:]], S, S)
    elif s.startswith("RM-"):
        stale = s.endswith("s"); d = s[3:-1] if stale else s[3:]
        new = (pair[f"P_{d}cm"], pair[f"S_{d}cm"], S if stale else pair[f"S_{d}cm"])
    else:
        raise ValueError(scen)
    return (new, None) if post else ((P, S, S), new)


def build(cls, scen, seed, Ls, rmode="fd", final=False):
    """Data dictionary for the simulators.  rmode: 'fd' = filtered reference by the N-point product (as before);
    'td' = spectrum of the time-domain filtered reference."""
    guard_seed(seed, final)
    if rmode not in ("fd", "td") or not 0 < Ls <= N:
        raise ValueError((rmode, Ls))
    x = excitation(cls, seed)
    pair = load_pair(int(seed) % 8 + 1)
    a, b = paths_of(scen, pair)
    lf = p.signal.lfilter
    n0 = CB * B
    d = lf(a[0], [1.0], x); m0 = a[2][:Ls].copy(); r = lf(m0, [1.0], x)
    plant, model, changes = [(0, a[1])], [(0, m0)], []
    filtered = [(0, r)]
    if b is not None:
        d = np.concatenate([d[:n0], lf(b[0], [1.0], x)[n0:]])
        m1 = b[2][:Ls].copy()
        if b[1] is not a[1]:
            plant.append((CB, b[1]))
        if not np.array_equal(m1, m0):
            model.append((CB, m1))
            r1 = lf(m1, [1.0], x)
            filtered.append((CB, r1))
            r = np.concatenate([r[:n0], r1[n0:]])
        changes = [CB]
    Sf = np.fft.rfft(m0, N); S2 = np.abs(Sf) ** 2
    dd = dict(x=x, d=d, r=r, xq=np.concatenate([np.zeros(N), x]), hS=a[1], hS_seq=plant, hSm_seq=model, Sf=Sf,
              S2=S2 / S2.mean(), filtered=filtered, primary_power=float(np.mean(d ** 2)), pc=False, changes=changes, tag=f"{cls}|{scen}|s{seed}|L{Ls}|{rmode}", events=[])
    if scen in ("EN0", "ENm10"):
        q.error_noise(dd, int(seed), 0.0 if scen == "EN0" else -10.0, spans=BURSTS)
        dd["events"] = [int(t * BPS) for t, _ in BURSTS]
    nb = len(d) // B
    if rmode == "fd":
        sf = [(b0, np.fft.rfft(m, N)) for b0, m in model]
        Rall = np.array([seq_at(sf, i) * np.fft.rfft(dd["xq"][i * B + B:i * B + B + N]) for i in range(nb)])
    else:
        rq = np.concatenate([np.zeros(N), r])
        Rall = np.array([np.fft.rfft(rq[i * B + B:i * B + B + N]) for i in range(nb)])
    return dd, Rall, T.Stats(dd, Rall)


def seq_at(seq, i):
    out = seq[0][1]
    for b0, v in seq:
        if i >= b0:
            out = v
    return out


# ----------------------------------------------------------------------------------------------
# Block simulator for the rules (arithmetic of vssT_lib.simulate; time-varying plant and model)
# ----------------------------------------------------------------------------------------------
def fir_block(hist, yb, h):
    """Output of the FIR h for the block yb, given hist = the len(hist) samples before it (len(hist) >= len(h) - 1)."""
    seg = np.concatenate([hist[len(hist) - (len(h) - 1):], yb])
    return sliding_window_view(seg, len(h)) @ h[::-1]


def filtered_block(states, i, hist, yb, seq):
    """Initialize lfilter history against the new impulse response at a switch."""
    h = seq_at(seq, i)
    old_h, zi = states
    if h is not old_h:
        zi = np.zeros(len(h) - 1)
        if len(zi):
            _, zi = p.signal.lfilter(h, [1.0], hist[-len(zi):], zi=zi)
    out, zi = p.signal.lfilter(h, [1.0], yb, zi=zi)
    return out, (h, zi)


def diverged(eb, primary_power):
    power = float(np.mean(eb ** 2))
    return not np.isfinite(power) or power > DIV_GAIN * primary_power


def simulate(dd, Rall, st, rule, nb=None, error_frame=None, hook=None):
    d, xq = dd["d"], dd["xq"]
    nbmax = min(len(d) // B, len(Rall)); nb = nbmax if nb is None else min(nb, nbmax)
    swv = sliding_window_view(xq, M)
    w = np.zeros(M); e_all = np.zeros(nb * B); y_all = np.zeros_like(e_all)
    Lh = max(max(len(h) for _, h in dd["hS_seq"]), max(len(h) for _, h in dd["hSm_seq"]))
    hist = np.zeros(Lh - 1)
    pref = dd.get("primary_power", float(np.mean(d ** 2)))
    div = False; n_done = nb; failure = None
    plant_state = model_state = (None, None)
    same_model = (len(dd["hSm_seq"]) == len(dd["hS_seq"])
                  and all(b == c and np.array_equal(h, m)
                          for (b, h), (c, m) in zip(dd["hS_seq"], dd["hSm_seq"])))
    for i in range(nb):
        s0 = i * B
        Xb = swv[s0 + N - M + 1:s0 + N - M + 1 + B][:, ::-1]
        yb = Xb @ w
        y_all[s0:s0+B] = yb
        ys, plant_state = filtered_block(plant_state, i, hist, yb, dd["hS_seq"])
        eb = d[s0:s0 + B] + ys
        e_all[s0:s0 + B] = eb
        if error_frame is None:
            E = p.ML.frame_E(e_all, s0, M)
        else:
            if same_model:
                ym = ys
            else:
                ym, model_state = filtered_block(model_state, i, hist, yb, dd["hSm_seq"])
            E = error_frame(i, w, e_all, ym)
        if hist.size and (len(dd["hS_seq"]) > 1 or len(dd["hSm_seq"]) > 1):
            hist = np.concatenate([hist, yb])[-hist.size:]
        c = T.Ctx(i, Rall[i], E, st, w)
        rule.mu_formula(i)
        mu, dv = rule.block(i, c)
        if hook is not None:
            hook(i, c, eb, mu)
        w = c.w_override if c.w_override is not None else w - 2 * mu * B / M * dv
        div = not np.isfinite(w).all() or diverged(eb, pref)
        if div:
            failure = dict(error_peak(e_all, s0), sample=s0 + B - 1)
            e_all[s0:] = np.nan; n_done = i
            break
    return dict(e=e_all, y=y_all, w=w, div=div, nb=nb, n_done=n_done, failure=failure, ex=np.full(nb, np.nan), tr=rule.traces())


class SFfix(p.Rule):
    """SF direction with a fixed step (no 1/g, no early phase, no restart), ordinary error frame."""

    def __init__(s, mu):
        super().__init__(); s.mu = float(mu); s.trials = []; s.restarts = []

    def block(s, i, c):
        s.tr["mu_app"].append(s.mu)
        return s.mu, p.SFm.sf_dir(c.R, c.E, c.Srr, c.Pbar, c.eps, 1e-2)


class AlwaysTrial(q5.P5):
    """Reuse P5 judgment and re-arm on each decision block, without rsN2."""

    def block(self, i, c):
        mu, dv = super().block(i, c)
        if self.bg is None:
            foreground = c.w if c.w_override is None else c.w_override
            self.bg = dict(block=i, w=foreground.copy())
            self.trials.append(dict(block=i, decide=None, rho=None, ok=None, status="pending"))
        return mu, dv


GE_CAP = 0.55      # floor of g in steps taken on rebuilt frames: a = s h / g <= 0.25 / 0.55 = 0.455 (< 0.47, E7: 0/20)


class _CapMixin:
    """Floor GE_CAP on g for the early step (foreground early phase and background trial); late step unchanged.
    Without it the floor is GMIN = 0.25, so the early step can reach a = 1.0, above the measured limit 0.55-0.625."""

    def block(s, i, c):
        g0 = c.g
        early = s.in_early(i)
        bg = getattr(s, "bg", None)
        capped = early or bg is not None
        if capped:
            c.g = max(g0, GE_CAP)
        try:
            mu, dv = super().block(i, c)
        finally:
            c.g = g0
        rolled = s.tr.get("rollback", [False])[-1]
        if capped and not early and dv is not None and c.w_override is None and not rolled:
            mu = s.late / max(g0, s.gmin)
            s.tr["mu_app"][-1] = mu
        bg_updated = bg is not None and s.bg is bg
        s.tr.setdefault("mu_bg_app", []).append(s.c["mu_e"] / max(g0, GE_CAP, s.gmin) if bg_updated else np.nan)
        return mu, dv


class P5Cap(_CapMixin, q5.P5):
    pass


class P4Cap(_CapMixin, q4.P4):
    pass


class SwitchedFrame(q4.Frame4):
    """Use each sample's active model for the synthesis of frozen errors."""

    def __init__(self, data, rule, always=False):
        super().__init__(data, rule)
        self.always = always
        self.references = [(block * B, np.pad(r, (2 * M - 1, 0)))
                           for block, r in data["filtered"]]

    def __call__(self, i, w, e_all, ys):
        self.end, self.e_all = (i + 1) * B, e_all
        self.sec[:-B] = self.sec[B:]
        self.sec[-B:] = ys
        seg = e_all[max(0, self.end - M):self.end]
        self.energy = float(seg @ seg)
        if not (self.always or self.rule.in_early(i)):
            return p.ML.frame_E(e_all, i * B, M)
        self.n_rebuilt += 1
        rebuilt = self.trial_frame(w)
        frame = np.zeros(N)
        frame[-len(rebuilt):] = rebuilt
        return np.fft.rfft(frame)

    def trial_frame(self, w):
        end = self.end
        start = max(0, end - M)
        for switch, reference in reversed(self.references):
            if switch <= start:
                self.ref = reference
                break
        result = super().trial_frame(w)
        for switch, reference in self.references:
            if start < switch < end:
                self.ref = reference
                result[switch - start:] = super().trial_frame(w)[switch - start:]
        return result


def run_rule(label, dd, Rall, st, X4, nb=None, hook=None):
    nbr = min(len(dd["d"]) // B, len(Rall)) if nb is None else nb
    frame = True
    if label in ("P5-early", "P5-early-noG", "TP-always"):
        cfg = q5.config(early_ok=(4, 0.25))
        if label == "TP-always":
            cfg["restart"] = False
            rule = AlwaysTrial(cfg, dd, nbr, X4)
        else:
            rule = q5.P5(cfg, dd, nbr, X4)
        if label == "P5-early-noG":
            st = copy.copy(st); st.g = np.ones_like(st.g)
    elif label == "P4":
        rule = q4.P4(q4.config(K=16, win="B", abort=4.0), dd, nbr, X4)
    elif label == "P5-early-cap":
        rule = P5Cap(q5.config(early_ok=(4, 0.25)), dd, nbr, X4)
    elif label == "nors-cap":
        rule = P4Cap(q4.config(restart=False), dd, nbr, X4)
    elif label in ("nors", "ND-always"):
        rule = q4.P4(q4.config(restart=False), dd, nbr, X4)
    elif label == "SF-fix":
        rule = SFfix(q.mu_late(0.03)); frame = False
    else:
        raise ValueError(label)
    if frame:
        rule.frame = (SwitchedFrame(dd, rule, always=label == "ND-always")
                      if len(dd["hSm_seq"]) > 1 or label == "ND-always" else q4.Frame4(dd, rule))
    t0 = time.perf_counter()
    sim = simulate(dd, Rall, st, rule, nb=nb, error_frame=rule.frame if frame else None, hook=hook)
    sim["seconds"] = time.perf_counter() - t0
    return rule, sim


# ----------------------------------------------------------------------------------------------
# Per-sample simulator for the external methods
# ----------------------------------------------------------------------------------------------
def run_baseline(name, mu, params, dd, n_tot=None):
    x, d, r = dd["x"], dd["d"], dd["r"]
    T_ = len(d) if n_tot is None else min(n_tot, len(d))
    model = dd["hSm_seq"]; plant = [(b0 * B, h[::-1].copy()) for b0, h in dd["hS_seq"]]
    par = dict(params); par.setdefault("c_impulse", model[0][1]); par.setdefault("order_control", M)
    algo = make_algo(name, par)
    xr = np.concatenate([x[:T_][::-1], np.zeros(M - 1)])        # xr[T_-1-n : T_-1-n+M] = x[n], x[n-1], ...
    rr = np.concatenate([r[:T_][::-1], np.zeros(M - 1)])
    L = max(len(h) for _, h in plant)
    y2 = np.zeros(2 * L); head = 0
    w = np.zeros(M); e = np.zeros(T_)
    hrev = plant[0][1]; lh = len(hrev)
    switches = {n0: h for n0, h in plant[1:]}
    mswitch = {b0 * B: m for b0, m in model[1:]}
    pref = dd.get("primary_power", float(np.mean(d ** 2))); div = False; n_done = T_; failure = None
    t0 = time.perf_counter()
    for n in range(T_):
        if n in switches:
            hrev = switches[n]; lh = len(hrev)
        if n in mswitch and hasattr(algo, "Hc"):
            algo.Hc = np.fft.rfft(mswitch[n], n=algo.N)
        a = T_ - 1 - n
        xv = xr[a:a + M]; rv = rr[a:a + M]
        y = float(w @ xv)
        y2[head] = y; y2[head + L] = y
        head = head + 1 if head + 1 < L else 0
        en = d[n] + float(y2[head + L - lh:head + L] @ hrev)
        e[n] = en
        finite = math.isfinite(en)
        if finite:
            w, _ = algo.update(w, en, rv, xv, mu)
        if not finite or ((n + 1) % B == 0
                                  and (diverged(e[n + 1 - B:n + 1], pref) or not np.isfinite(w).all())):
            div = True; n_done = n // B * B
            failure = dict(error_peak(e, n_done), sample=n)
            e[n_done:] = np.nan
            break
    return dict(e=e, w=w, div=div, n_done=n_done, failure=failure, seconds=time.perf_counter() - t0)


# ----------------------------------------------------------------------------------------------
# Measures
# ----------------------------------------------------------------------------------------------
def ratio_db(e, d, lo, hi):
    return float(10 * np.log10(np.sum(e[lo:hi] ** 2) / np.sum(d[lo:hi] ** 2)))


def recovery(e, cb, target_power, win=16, hold=BPS):
    """Start of the first sustained run below twice the post-only error power."""
    nb = len(e) // B
    power = np.mean(e[:nb * B].reshape(nb, B) ** 2, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(power[cb:])])
    limit = target_power * 10 ** (3 / 10)
    ok = (cumulative[win:] - cumulative[:-win]) / win <= limit
    cnt = 0
    for j, accept in enumerate(ok):
        cnt = cnt + 1 if accept else 0
        if cnt >= hold:
            return j - hold + win
    return None


def measures(sim, dd, post=None):
    e, d = sim["e"], dd["d"][:len(sim["e"])]
    out = dict(div=int(bool(sim["div"])), seconds=round(float(sim["seconds"]), 2))
    if sim["div"]:
        out["div_block"] = int(sim["n_done"] if "nb" in sim else sim["n_done"] // B)
        out["failure"] = sim["failure"]
        return out
    anr = p.O.anr_curve(e, d); ts = (63 + 64 * np.arange(len(anr))) / FS
    win = lambda a, b: float(anr[(ts >= a) & (ts < b)].mean())
    tend = len(e) / FS
    out.update(L1_whole=float(anr.mean()), L1_last10=win(max(0.0, tend - 10), tend), L2_last5=ratio_db(e, d, len(e) - 5 * FS, len(e)))
    ev = [b / BPS for b in (dd["changes"] or dd["events"]) if b / BPS + 2 <= tend]
    if ev:
        out["L1_ev"] = float(np.mean([win(t, t + 2) for t in ev]))
        out["L1_pre"] = float(np.mean([win(t - 2, t) for t in ev]))
    if dd["changes"] and post is not None and not post["div"]:
        ep = post["e"]
        target = float(np.mean(ep[-5 * FS:] ** 2))
        cb = dd["changes"][0]
        rec = recovery(e, cb, target)
        out.update(rec_target=target, rec_blocks=rec, rec_cens=int(rec is None),
                   rec_limit=max(0, len(e) // B - cb - BPS + 1))
    return out


def trial_stats(rule, dd):
    tr = getattr(rule, "trials", None) or []
    ch = dd["changes"]
    true = lambda t: any(cb <= t["block"] < cb + p.RS_TRUE_WIN for cb in ch)
    conf = [t for t in tr if t["status"] == "confirmed"]
    scored = getattr(rule, "det", None) is not None or isinstance(rule, AlwaysTrial)
    rs = p.restart_stats([t["block"] for t in conf], ch, dd.get("scored_blocks", len(dd["d"]) // B))
    false_blocks = set(rs["false_blocks"])
    def trial_row(t):
        rho = t.get("rho")
        return dict(block=t["block"], decide=t.get("decide"), status=t["status"],
                    rho=rho if rho is None or math.isfinite(rho) else str(rho),
                    false=t["block"] in false_blocks)
    return dict(trials=[trial_row(t) for t in tr], n_trial=len(tr), n_confirm=len(conf), false_confirm=rs["n_false"],
                missed=rs["n_missed"] if scored else None,
                detection_blocks=[t["block"] for t in tr if true(t)],
                decide_blocks=[t["decide"] - t["block"] for t in conf if true(t) and t.get("decide") is not None])


# ----------------------------------------------------------------------------------------------
# Jobs
# ----------------------------------------------------------------------------------------------
PILOT_CLASSES = ("G", "a145", "II300") + REAL
PILOT_SCEN = ("stat", "RS-near", "RS-far", "RM-10", "RM-10s", "EN0")
PILOT_SEEDS = (10, 11)
PILOT_LS = (256, 512)
POST_FOR = ("RS-near", "RS-far", "RM-10", "RM-10s")


def out_dir(smoke):
    return HERE / "out" / ("final_pilot2_smoke" if smoke else "final_pilot2")


def run_job(job):
    cls, scen, seed, Ls, smoke = job
    f = out_dir(smoke) / f"{cls}_{scen}_s{seed}_L{Ls}.json"
    if f.exists() and json.loads(f.read_text()).get("version") == VERSION:
        return
    t0 = time.monotonic()
    X4 = p.parse(["c5", "--preset", "final"]).X4
    nb = 20 * BPS if smoke else None
    n_tot = None if nb is None else nb * B
    with_post = scen in POST_FOR and not smoke
    rows = []
    data = {m: build(cls, scen, seed, Ls, m) for m in RMODES}
    posts = {m: build(cls, "post:" + scen, seed, Ls, m) for m in RMODES} if with_post else {}
    ref = {}
    for rmode in RMODES:
        dd, Rall, st = data[rmode]
        for label in RULES:
            rule, sim = run_rule(label, dd, Rall, st, X4, nb)
            post = None
            if with_post:
                pd_, pR, pst = posts[rmode]
                _, ps = run_rule(label, pd_, pR, pst, X4, nb)
                post = dict(e=ps["e"], d=pd_["d"][:len(ps["e"])], div=ps["div"])
            row = dict(method=label, kind="rule", rmode=rmode, **measures(sim, dd, post),
                       **trial_stats(rule, dict(dd, scored_blocks=sim["n_done"])))
            if post is not None:
                row["post_div"] = int(post["div"])
                row["post_seconds"] = ps["seconds"]
                row["post_failure"] = ps["failure"]
            if label == "nors":
                ref[rmode] = (sim["e"], sim["y"])
            elif rmode in ref and not sim["div"]:
                if np.isfinite(ref[rmode][0]).all():
                    diff = float(np.max(np.abs(sim["e"] - ref[rmode][0])))
                    speaker = float(np.max(np.abs(sim["y"] - ref[rmode][1])))
                    different = (sim["e"] != ref[rmode][0]) | (sim["y"] != ref[rmode][1])
                    row.update(same_as_nors=int(diff == 0 and speaker == 0),
                               max_diff_nors=diff, max_diff_y_nors=speaker,
                               first_diff_sample=int(np.argmax(different)) if different.any() else None)
            rows.append(row)
    dd = data["fd"][0]
    for name, (mu, par) in BASE.items():
        sim = run_baseline(name, mu, par, dd, n_tot)
        post = None
        if with_post:
            pd_ = posts["fd"][0]
            ps = run_baseline(name, mu, par, pd_, n_tot)
            post = dict(e=ps["e"], d=pd_["d"][:len(ps["e"])], div=ps["div"])
        row = dict(method=name, kind="base", rmode="-", mu=mu, **measures(sim, dd, post))
        if post is not None:
            row.update(post_div=int(post["div"]), post_seconds=ps["seconds"], post_failure=ps["failure"])
        rows.append(row)
    for r in rows:
        r.update(cls=cls, scen=scen, seed=int(seed), Ls=int(Ls))
    f.parent.mkdir(parents=True, exist_ok=True)
    ab.atomic_write(f, lambda h: json.dump(dict(version=VERSION, rows=rows, wall_seconds=time.monotonic()-t0), h))
    print(f"{f.stem}: {len(rows)} rows, div={sum(r['div'] for r in rows)}, {time.monotonic() - t0:.0f} s", flush=True)


# ----------------------------------------------------------------------------------------------
# S1c: floor on g for the early step (design seeds 10-14, S-hat 256, rules only)
# ----------------------------------------------------------------------------------------------
CAP_RULES = ("nors", "P5-early", "nors-cap", "P5-early-cap")
CAP_REF = {"P5-early": "nors", "P5-early-cap": "nors-cap"}
CAP_SEEDS = (10, 11, 12, 13, 14)
CAP_DIR = HERE / "out" / "final_cap"
CAP_VERSION = 1
H_STEP = 0.25                                   # h = (2B/M)(M_e/N); a = mu_app * h



def cap_steps(tr):
    mu = np.asarray(tr["mu_app"], float)
    early = np.asarray(tr["early"], bool)
    adaptive = ~(np.asarray(tr.get("rollback", np.zeros(len(mu))), bool)
                 | np.asarray(tr.get("transfer", np.zeros(len(mu))), bool))
    def maximum(values):
        return float(np.max(values) * H_STEP) if len(values) else None
    bg = np.array(tr.get("mu_bg_app", []))
    return dict(a_max_early=maximum(mu[early & adaptive]), a_max_late=maximum(mu[~early & adaptive]),
                a_max_bg=maximum(bg[np.isfinite(bg)]))

def run_cap_job(job):
    cls, scen, seed = job
    f = CAP_DIR / f"{cls}_{scen}_s{seed}.json"
    if f.exists() and json.loads(f.read_text()).get("version") == CAP_VERSION:
        return
    t0 = time.monotonic()
    X4 = p.parse(["c5", "--preset", "final"]).X4
    dd, Rall, st = build(cls, scen, seed, 256)
    post = build(cls, "post:" + scen, seed, 256) if scen in POST_FOR else None
    rows, ref = [], {}
    for label in CAP_RULES:
        rule, sim = run_rule(label, dd, Rall, st, X4)
        pm = None
        if post is not None:
            _, ps = run_rule(label, post[0], post[1], post[2], X4)
            pm = dict(e=ps["e"], d=post[0]["d"][:len(ps["e"])], div=ps["div"])
        row = dict(method=label, cls=cls, scen=scen, seed=int(seed), **measures(sim, dd, pm),
                   **trial_stats(rule, dict(dd, scored_blocks=sim["n_done"])))
        row.update(cap_steps(sim["tr"]))
        row["g_min32"] = float(np.min(st.g[:min(32, len(sim["tr"]["mu_app"]))]))
        if pm is not None:
            row["post_div"] = int(pm["div"])
        ref[label] = sim
        r0 = ref.get(CAP_REF.get(label))
        if r0 is not None and not sim["div"] and not r0["div"]:
            row["same_as_ref"] = int(all(np.array_equal(sim[k], r0[k]) for k in ("e", "y")))
        rows.append(row)
    f.parent.mkdir(parents=True, exist_ok=True)
    ab.atomic_write(f, lambda h: json.dump(dict(version=CAP_VERSION, rows=rows), h))
    print(f"{f.stem}: div={[r['div'] for r in rows]}, {time.monotonic() - t0:.0f} s", flush=True)


def cap(procs):
    jobs = [(c, s, sd) for c in PILOT_CLASSES for s in PILOT_SCEN for sd in CAP_SEEDS]
    with Pool(procs) as pool:
        list(pool.imap_unordered(run_cap_job, jobs))
    files = [json.loads((CAP_DIR / f"{c}_{s}_s{sd}.json").read_text()) for c, s, sd in jobs]
    if any(f["version"] != CAP_VERSION or tuple(r["method"] for r in f["rows"]) != CAP_RULES for f in files):
        raise ValueError("Incomplete or stale S1c results")
    rows = [r for f in files for r in f["rows"]]

    def med(values):
        if not values:
            return "—"
        mid = float(np.median(values))
        return "打切" if np.isinf(mid) else fmt(mid, 1)

    L = ["# S1c：前半のステップに使う g の下限（設計用 seed 10–14、Ŝ 256）", "",
         f"cap = 作り直した誤差で進む更新（前半と裏の試行）だけ g の下限を {GE_CAP} にする（a ≤ {0.25 / GE_CAP:.3f}）。後半は変更なし。",
         "表は自動生成。L1 は完走した実行の平均（dB）。rec は回復までのブロック数の中央値（打ち切りを含む）。", ""]
    for scen in PILOT_SCEN:
        L += [f"## {scen}", "", "| 手法 | 試行 | 発散 | L1 whole | L1 last10 | L1 直後2秒 | rec 中央値 | 打切 | 誤確定 | 見逃し | 基準と一致 | a 最大（前半） |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for m in CAP_RULES:
            rm = [r for r in rows if r["scen"] == scen and r["method"] == m]; ok = [r for r in rm if not r["div"]]
            recs = [float("inf") if r["rec_blocks"] is None else r["rec_blocks"] for r in ok if r.get("rec_cens") is not None]
            cnt = lambda k: "—" if all(r.get(k) is None for r in rm) else str(sum(r.get(k) or 0 for r in rm))
            am = [r["a_max_early"] for r in rm if r.get("a_max_early") is not None]
            L.append(f"| {m} | {len(rm)} | {sum(r['div'] for r in rm)} | {fmt(agg(ok, 'L1_whole'))} | {fmt(agg(ok, 'L1_last10'))} | {fmt(agg(ok, 'L1_ev'))} | "
                     f"{med(recs)} | {sum(v == float('inf') for v in recs) if recs else '—'} | {cnt('false_confirm')} | {cnt('missed')} | "
                     f"{cnt('same_as_ref')}/{sum(r.get('same_as_ref') is not None for r in ok)} | {fmt(max(am), 3) if am else '—'} |")
        L.append("")
    L += ["## 発散した実行", "", "| クラス | 条件 | seed | 手法 | 発散ブロック | a 最大（前半） | g 最小（先頭32ブロック） |", "|---|---|---:|---|---:|---:|---:|"]
    L += [f"| {r['cls']} | {r['scen']} | {r['seed']} | {r['method']} | {r.get('div_block', '—')} | {fmt(r.get('a_max_early'), 3)} | {fmt(r.get('g_min32'), 3)} |"
          for r in rows if r["div"]]
    L += ["", "## 対応のある差（cap − 変更前、両方完走した実行）", "", "| 組 | 条件 | L1 whole の差 | L1 直後2秒の差 | whole / 直後 対数 | 回復差（ブロック） | 回復対数 |", "|---|---|---:|---:|---:|---:|---:|"]
    key = lambda r: (r["cls"], r["scen"], r["seed"])
    for a, b in (("nors-cap", "nors"), ("P5-early-cap", "P5-early")):
        for scen in PILOT_SCEN + ("全条件",):
            A = {key(r): r for r in rows if r["method"] == a and not r["div"] and scen in (r["scen"], "全条件")}
            Bm = {key(r): r for r in rows if r["method"] == b and not r["div"] and scen in (r["scen"], "全条件")}
            ks = sorted(set(A) & set(Bm))
            dif = lambda k_: [A[k][k_] - Bm[k][k_] for k in ks if A[k].get(k_) is not None and Bm[k].get(k_) is not None]
            L.append(f"| {a} − {b} | {scen} | {fmt(float(np.mean(dif('L1_whole'))) if dif('L1_whole') else None, 3)} | "
                     f"{fmt(float(np.mean(dif('L1_ev'))) if dif('L1_ev') else None, 3)} | {len(ks)} / {len(dif('L1_ev'))} | "
                     f"{fmt(float(np.mean(dif('rec_blocks'))) if dif('rec_blocks') else None, 1)} | {len(dif('rec_blocks'))} |")
    L += ["", "## クラスごとの a 最大（前半、変更前の P5-early）と発散", "", "| クラス | a 最大 | a>0.55 の実行 | 発散（P5-early） | 発散（P5-early-cap） |", "|---|---:|---:|---:|---:|"]
    for c in PILOT_CLASSES:
        rc = [r for r in rows if r["cls"] == c and r["method"] == "P5-early"]; am = [r["a_max_early"] for r in rc if r.get("a_max_early") is not None]
        L.append(f"| {c} | {fmt(max(am), 3) if am else '—'} | {sum(v > 0.55 for v in am)}/{len(am)} | {sum(r['div'] for r in rc)} | "
                 f"{sum(r['div'] for r in rows if r['cls'] == c and r['method'] == 'P5-early-cap')} |")
    (HERE / "RESULTS_final_cap.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote RESULTS_final_cap.md", len(rows), "rows")


def pilot(procs, smoke):
    seeds = PILOT_SEEDS[:1] if smoke else PILOT_SEEDS
    classes = ("G", "a145", "machinegun") if smoke else PILOT_CLASSES
    scen = ("stat", "RS-far", "RM-10s") if smoke else PILOT_SCEN
    jobs = [(c, s, sd, L, smoke) for L in (PILOT_LS[:1] if smoke else PILOT_LS) for c in classes for s in scen for sd in seeds]
    if procs == 1:
        for j in jobs:
            run_job(j)
    else:
        with Pool(procs) as pool:
            list(pool.imap_unordered(run_job, jobs))


# ----------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------
def fmt(v, nd=2):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def agg(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(v)) if v else None


def report(smoke):
    d = out_dir(smoke)
    jobs = [json.loads(f.read_text()) for f in sorted(d.glob("*.json"))]
    rows = [r for job in jobs if job["version"] == VERSION for r in job["rows"]]
    if not rows:
        raise ValueError(f"No pilot rows in {d}")
    L = ["# 最終評価 S1b：100 Hz 高域通過を入れた試行（設計用 seed）", "",
         "表は自動生成。値は dB（小さいほど良い）、発散した試行は平均から除き件数を示す。rec は回復までのブロック数の中央値"
         "（打ち切りは件数で表示。打ち切りは観測上限より後として順位に含め、中央値が確定しない場合は「打切」）。", ""]
    methods = list(RULES) + list(BASE)
    for Ls in sorted({r["Ls"] for r in rows}):
        for scen in PILOT_SCEN:
            for rmode in RMODES:
                rs = [r for r in rows if r["Ls"] == Ls and r["scen"] == scen and r["rmode"] in (rmode, "-" if rmode == "fd" else "")]
                if not rs:
                    continue
                L += [f"## Ŝ {Ls} タップ、{scen}、参照 {rmode}", "",
                      "| 手法 | 試行 | 発散 | L1 whole | L1 last10 | L1 直後2秒 | rec 中央値 | 打切 | 試行回数 | 誤確定 | 見逃し | nors と一致 | 秒 |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
                for m in methods:
                    rm = [r for r in rs if r["method"] == m]
                    if not rm:
                        continue
                    okr = [r for r in rm if not r["div"]]
                    recs = [r for r in okr if "rec_cens" in r]
                    if recs:
                        vals = sorted(float("inf") if r["rec_blocks"] is None else r["rec_blocks"] for r in recs)
                        med = (vals[(len(vals) - 1) // 2] + vals[len(vals) // 2]) / 2
                        med = "打切" if med == float("inf") else f"{med:g}"
                    else:
                        med = "—"
                    cnt = lambda k: "—" if all(r.get(k) is None for r in rm) else str(sum(r.get(k) or 0 for r in rm))
                    L.append(f"| {m} | {len(rm)} | {sum(r['div'] for r in rm)} | {fmt(agg(okr, 'L1_whole'))} | {fmt(agg(okr, 'L1_last10'))} | "
                             f"{fmt(agg(okr, 'L1_ev'))} | {med} | {sum(r['rec_cens'] for r in recs) if recs else '—'} | {cnt('n_trial')} | "
                             f"{cnt('false_confirm')} | {cnt('missed')} | {cnt('same_as_nors')} | {fmt(agg(rm, 'seconds'), 1)} |")
                L.append("")
    L += ["## クラスごとの L1 last10（Ŝ ごと、stat、参照 fd）", ""]
    for Ls in sorted({r["Ls"] for r in rows}):
        classes = sorted({r["cls"] for r in rows})
        L += [f"### Ŝ {Ls} タップ", "", "| 手法 | " + " | ".join(classes) + " |", "|---|" + "---:|" * len(classes)]
        for m in methods:
            cells = [fmt(agg([r for r in rows if r["Ls"] == Ls and r["scen"] == "stat" and r["method"] == m and r["cls"] == c
                              and r["rmode"] in ("fd", "-") and not r["div"]], "L1_last10")) for c in classes]
            L.append(f"| {m} | " + " | ".join(cells) + " |")
        L.append("")
    dest = (d if smoke else HERE) / "RESULTS_final_pilot2.md"
    dest.write_text("\n".join(L), encoding="utf-8")
    print("wrote", dest, len(rows), "rows")


# ----------------------------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------------------------
def check():
    X4 = p.parse(["c5", "--preset", "final"]).X4
    rng = np.random.default_rng(0)
    # 1. Block FIR = lfilter, across a switch of the response.
    h1, h2 = rng.standard_normal(300), rng.standard_normal(200); y = rng.standard_normal(6 * B)
    hist = np.zeros(299); out = []
    for i in range(6):
        yb = y[i * B:(i + 1) * B]; out.append(fir_block(hist, yb, h1 if i < 3 else h2)); hist = np.concatenate([hist, yb])[-299:]
    ref = np.concatenate([p.signal.lfilter(h1, [1.0], y)[:3 * B], p.signal.lfilter(h2, [1.0], y)[3 * B:]])
    np.testing.assert_allclose(np.concatenate(out), ref, atol=1e-9)
    print("check 1 ok: block FIR with a switch")
    # 2. Reserved seeds are refused.
    try:
        guard_seed(100)
    except SystemExit:
        print("check 2 ok: seeds 100-119 refused")
    else:
        raise AssertionError("seed guard")
    # 3. Input power of the real recordings equals that of the Gaussian class.
    pg = g_power()
    for cls in REAL:
        pm = float(np.mean(excitation(cls, 10) ** 2))
        assert abs(pm / pg - 1) < 1e-9, (cls, pg, pm)
    print(f"check 3 ok: input power {pg:.3f}")
    # 4. Scenario bookkeeping.
    dd, Rall, st = build("G", "RS-far", 10, 256)
    assert dd["changes"] == [CB] and len(dd["hS_seq"]) == 1 and len(dd["hSm_seq"]) == 1
    d2, _, _ = build("G", "RM-10", 10, 256); d3, _, _ = build("G", "RM-10s", 10, 256)
    assert len(d2["hS_seq"]) == 2 and len(d2["hSm_seq"]) == 2 and len(d3["hSm_seq"]) == 1
    dp, _, _ = build("G", "post:RS-far", 10, 256)
    np.testing.assert_allclose(dp["d"][CB * B + 5000:CB * B + 6000], dd["d"][CB * B + 5000:CB * B + 6000], atol=1e-9)
    print("check 4 ok: scenarios")
    # 5. The block simulator and the per-sample simulator give the same error for frozen (zero) coefficients,
    #    and both reduce the error on a short stationary run.
    dd, Rall, st = build("G", "stat", 10, 256)
    rule, sim = run_rule("P5-early", dd, Rall, st, X4, nb=600)
    bs = run_baseline("FxNLMS", 0.3, {}, dd, 600 * B)
    assert not sim["div"] and not bs["div"]
    a = ratio_db(sim["e"], dd["d"], 500 * B, 600 * B); b = ratio_db(bs["e"], dd["d"], 500 * B, 600 * B)
    assert a < -3 and b < -1, (a, b)
    print(f"check 5 ok: after 4.8 s, P5-early {a:.2f} dB, FxNLMS {b:.2f} dB ({sim['seconds']:.1f} s, {bs['seconds']:.1f} s)")
    # 6. Recovery measure on a hand-made curve.
    d = np.ones(400 * B); e = np.full(400 * B, 0.1); e[100 * B:140 * B] = 1.0
    assert recovery(e, 100, 0.01) == 55, recovery(e, 100, 0.01)      # first 16-block window after block 139
    assert recovery(np.ones(400 * B), 100, 0.01) is None
    print("check 6 ok: recovery", recovery(e, 100, 0.01))
    # 7. Without a change and without bursts, P5-early equals the rule without restart unless a trial was confirmed.
    rn, sn = run_rule("nors", dd, Rall, st, X4, nb=600)
    if not rule.restarts:
        np.testing.assert_allclose(sim["e"], sn["e"], rtol=1e-9, atol=1e-12)
    print("check 7 ok: no confirmed trial -> same output as no restart; trials", [t["status"] for t in rule.trials])


    # 8. Reference simulators on the repository's default paths.
    standard, R0, st0 = q.prep("G_s10", 256)
    standard.update(hS_seq=[(0, standard["hS"])], hSm_seq=[(0, standard["hS"])])
    for label, cfg in (("nors", q4.config(restart=False)), ("P5-early", q5.config(early_ok=(4, 0.25)))):
        reference = q4.run_p4 if label == "nors" else q5.run_p5
        _, expected = reference(standard, R0, st0, cfg, 256, X4)
        _, actual = run_rule(label, standard, R0, st0, X4, 256)
        assert not actual["div"] and not expected["div"]
        np.testing.assert_array_equal(actual["e"], expected["e"])
        np.testing.assert_array_equal(actual["w"], expected["w"])
    print("check 8 ok: block errors and coefficients exactly match vssT_lib on default paths")

    from anc import pad_or_trim, step_td
    hP, hS = pad_or_trim(T.BS.hP, M), standard["hS"]
    xvec, weights, yhist = np.zeros(M), np.zeros((1, M)), np.zeros((1, len(hS)))
    rb, yc, ys = np.zeros((1, M)), np.zeros(1), np.zeros(1)
    algo = make_algo("FxNLMS", {"order_control": M})
    reference_e = np.empty(8 * B)
    for i, x in enumerate(standard["x"][:len(reference_e)]):
        d0 = step_td(float(x), xvec, weights, yhist, hP, hS, len(hS), rb, yc, ys)
        reference_e[i] = d0 + ys[0]
        weights[0], _ = algo.update(weights[0], reference_e[i], rb[0], xvec, 0.3)
    actual = run_baseline("FxNLMS", 0.3, {}, standard, len(reference_e))
    assert not actual["div"]
    np.testing.assert_allclose(actual["e"], reference_e, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual["w"], weights[0], rtol=1e-10, atol=1e-10)
    print("check 9 ok: FxNLMS agrees with anc.step_td; max error difference",
          np.max(np.abs(actual["e"] - reference_e)))

    # 10. Independent FIR reference at a model switch with a frozen controller.
    x = rng.standard_normal(8 * B)
    w = rng.standard_normal(M)
    y = np.convolve(x, w)[:len(x)]
    models = [(0, h1), (3, h2)]
    refs = [(b, p.signal.lfilter(h, [1.0], x)) for b, h in models]
    frame = SwitchedFrame(dict(r=refs[0][1], filtered=refs), None, always=True)
    history = np.zeros(len(h1) - 1)
    state = (None, None)
    d_only = rng.standard_normal(len(x))
    actual_e = d_only.copy()
    expected_e = d_only + np.concatenate([p.signal.lfilter(h1, [1.0], y)[:3*B],
                                           p.signal.lfilter(h2, [1.0], y)[3*B:]])
    for i in range(8):
        block = slice(i * B, (i + 1) * B)
        secondary, state = filtered_block(state, i, history, y[block], models)
        actual_e[block] += secondary
        frame(i, w, actual_e, secondary)
        np.testing.assert_allclose(frame.trial_frame(w), expected_e[max(0, (i+1)*B-M):(i+1)*B], atol=1e-9)
        history = np.concatenate([history, y[block]])[-len(history):]
    print("check 10 ok: model switch preserves frozen-error reconstruction")


    # 11. Match every real input to filtered G and verify the high-pass response.
    np.testing.assert_equal(g_power(), np.mean(excitation("G", 10) ** 2))
    for cls in (*REAL, "G", "a145", "II300"):
        raw = _excitation(cls, 10)
        filtered = highpass(raw)
        if cls in REAL:
            filtered *= np.sqrt(g_power() / np.mean(filtered ** 2))
        np.testing.assert_array_equal(excitation(cls, 10), filtered)
    ts = np.arange(FS) / FS
    tones = [np.sin(2 * np.pi * hz * ts) for hz in (40, 250)]
    assert np.std(highpass(tones[0])[FS//2:]) < 0.02
    assert np.std(highpass(tones[1])[FS//2:]) > 0.70
    assert error_peak(tones[1], FS)["band"] == "250–500"
    print("check 11 ok: filtered-G power, high-pass response and band diagnostic")


    # 12. g above the cap leaves both foreground and trial arithmetic unchanged.
    fixtures = ("nors", "P5-early")
    dd, Rall, st = build("G", "RS-far", 10, 256)
    above = copy.copy(st)
    above.g = np.maximum(st.g, GE_CAP)
    for label in fixtures:
        _, uncapped = run_rule(label, dd, Rall, above, X4, nb=CB + 160)
        _, capped = run_rule(label + "-cap", dd, Rall, above, X4, nb=CB + 160)
        assert not uncapped["div"] and not capped["div"]
        np.testing.assert_array_equal(uncapped["e"], capped["e"])
        np.testing.assert_array_equal(uncapped["tr"]["mu_app"], capped["tr"]["mu_app"])
    print("check 12 ok: capped and uncapped errors and steps match exactly above the floor")

    # 13. Force the floor to engage; compare logged adaptive steps with the formulas.
    low = copy.copy(st)
    low.g = np.full_like(st.g, 0.50)
    for label in fixtures:
        rule, sim = run_rule(label + "-cap", dd, Rall, low, X4, nb=CB + 160)
        assert not sim["div"]
        steps = cap_steps(sim["tr"])
        np.testing.assert_allclose(steps["a_max_early"], H_STEP / GE_CAP)
        if label == "P5-early":
            np.testing.assert_allclose(steps["a_max_bg"], H_STEP / GE_CAP)
        tr = sim["tr"]
        late = (tr["early"] == 0) & (tr.get("transfer", np.zeros(len(tr["early"]))) == 0)
        assert late.any()
        np.testing.assert_array_equal(tr["mu_app"][late], np.full(late.sum(), rule.late / 0.50))
    print("check 13 ok: foreground/background cap and original late-step log")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("cmd", choices=("check", "pilot", "report", "cap"))
    a.add_argument("--procs", type=int, default=os.cpu_count() or 1)
    a.add_argument("--smoke", action="store_true")
    o = a.parse_args()
    if o.cmd == "check":
        check()
    elif o.cmd == "pilot":
        pilot(o.procs, o.smoke)
    elif o.cmd == "cap":
        cap(o.procs)
    else:
        report(o.smoke)


if __name__ == "__main__":
    main()
