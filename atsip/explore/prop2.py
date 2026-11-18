"""Proposed method, version 2 (P2), and the dev1 experiments E1-E7.

P2 = SF direction, applied step s_n / max(g_n, g_min), and
  early phase (cold start, and from the block after a restart detection): s = mu_e; the error frame is either the
      ordinary one (early=SF) or rebuilt with the current coefficients (early=ND, modified filtered-x, only here);
  end of the early phase: fixed length, or qstop(theta): q_n = ||y_n / max(g_n, g_min)||^2, Q4_n = mean of the last
      4 q; from the 8th early block on, the phase ends when Q4_n > theta * Q4_{n-4}; upper limit NE_MAX blocks;
  late phase: s = mu_L(delta) = 2 delta / (h (tau + (2M/B) delta)), delta = admissible steady-state excess ratio;
  restart: the rsN2 detector of protoC (error energy over the 2M-window reference energy), unchanged.

Commands (run in this order; every step can be resumed, finished jobs are skipped):
  python prop2.py check
  python prop2.py run --exp all [--procs K] [--smoke]
  python prop2.py report [--smoke]
"""
import argparse
import csv
import json
import math
import os
import re
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import ablate as ab
import nodelay as nd
import protoC as p

np, T = p.np, p.T
B, M, N, FS, BPS = p.B, p.M, p.N, p.FS, p.BPS
HERE = Path(__file__).resolve().parent
SEEDS = (10, 11, 12, 13, 14)
H = T.H                       # 0.25
TAU = M / B                   # error-window overlap (M_e = M)
NE_MAX = 64
BLOCKS = (8, 16, 32, 64, 125)
CURVE_LEN = 250
BASE8 = ("G", "BU", "PS", "II300", "a145", "a120", "pc", "pcb")
BASE6 = BASE8[:6]
E5_CLASSES = ("ENm20", "ENm10", "EN0", "EV", "EVpc", "SC")
EVENTS = {"pc": (15,), "pcb": (10, 20), "ENm20": (10, 20), "ENm10": (10, 20), "EN0": (10, 20), "EVpc": (15,),
          "SC": (15,)}
SHATS = ("g0.7", "g1.4", "d1", "n20")
FINAL = p.FINAL_RULE
BASELINES = (FINAL, "SF early+fix+g+rsN2 mu0.05", "SF-V2", "SF fix mu0.1")
COLUMNS = ("tag", "class", "seed", "rule", "L1_whole", "L1_first5", "L1_last10", "ex_0_1s", "ex_0_5s", "ex_last10",
           "L1_pc15_17", "L1_pcb10_12", "L1_pcb20_22", "div_num", "amp", "false_rs", "missed_rs", "exp",
           "L1_ev", "ex_ev", "n_restart", "early_mean", "early_min", "early_max", "n_rebuilt",
           "ex_b8", "ex_b16", "ex_b32", "ex_b64", "ex_b125")


def mu_late(delta):
    """Late step from the small-step steady state X_ss / N = mu h tau / (2 - mu h 2M/B) = delta."""
    return 2.0 * delta / (H * (TAU + (2.0 * M / B) * delta))


# ----------------------------------------------------------------------------------------------
# Rule
# ----------------------------------------------------------------------------------------------
def config(early, mu_e, end, delta=0.03, restart=True, eps=1e-2, mu_l=None):
    """end: ('fixed', n) or ('qstop', theta).  mu_l overrides mu_late(delta) (used by check only)."""
    return dict(early=early, mu_e=float(mu_e), end=tuple(end), delta=float(delta), restart=bool(restart),
                eps=float(eps), mu_l=mu_l)


def name_of(c):
    end = f"fixed{c['end'][1]}" if c["end"][0] == "fixed" else f"qstop{c['end'][1]:g}"
    s = f"P2 early={c['early']} mue{c['mu_e']:g} end={end} d{c['delta']:g}"
    if not c["restart"]:
        s += " nors"
    if c["eps"] != 1e-2:
        s += f" | eps{c['eps']:g}"
    return s


class Frame:
    """Error frame for early=ND: rebuilt with the current coefficients while the rule is in its early phase
    (same arithmetic as nodelay.ReconstructedError), the ordinary frame otherwise.  The secondary-path output
    passed by the simulator is that of the model (S-hat) when data['hS_model'] is set."""

    def __init__(s, data, rule):
        s.ref = np.pad(data["r"], (2 * M - 1, 0)); s.sec = np.zeros(M)
        s.rule, s.energy, s.n_rebuilt = rule, 0.0, 0

    def __call__(s, i, w, e_all, ys):
        end = (i + 1) * B; start = max(0, end - M)
        s.sec[:-B] = s.sec[B:]; s.sec[-B:] = ys
        seg = e_all[start:end]
        s.energy = float(seg @ seg)                 # actual error energy, for the restart detector
        if not s.rule.in_early(i):
            return p.ML.frame_E(e_all, i * B, M)
        s.n_rebuilt += 1
        n = end - start
        frozen = np.fft.irfft(np.fft.rfft(s.ref[end:end + 2 * M - 1], N) * np.fft.rfft(w, N), N)
        frame = np.zeros(N)
        frame[-n:] = (seg - s.sec[-n:]) + frozen[2 * M - 1 - n:2 * M - 1]
        return np.fft.rfft(frame)


class P2(p.Rule):
    def __init__(s, c, data, nb, X4, gmin=T.GMIN):
        super().__init__()
        s.c, s.gmin = c, gmin
        s.late = mu_late(c["delta"]) if c["mu_l"] is None else c["mu_l"]
        s.length = c["end"][1] if c["end"][0] == "fixed" else NE_MAX
        s.theta = c["end"][1] if c["end"][0] == "qstop" else None
        s.rw = p.rwin_energy(data, nb, 2 * M) if c["restart"] else None
        s.det = p.RSNDetector(X4) if c["restart"] else None
        s.until = s.length; s.q = []; s.restarts = []; s.frame = None
        s.phases = [[0, None, None]]                # [first block, first block after the phase, reason]
        s.tr.update(mu_state=[], restart_apply_block=[], early=[])

    def in_early(s, i):
        return i < s.until

    def mu_formula(s, i):
        return s.c["mu_e"] if s.in_early(i) else s.late

    def _close(s, end, reason):
        if s.phases and s.phases[-1][1] is None:
            s.phases[-1][1:] = [int(end), reason]

    def block(s, i, c):
        early = s.in_early(i)
        if not early:
            s._close(s.until, "fixed" if s.theta is None else "limit")
        internal = s.c["mu_e"] if early else s.late
        gain = max(c.g, s.gmin)
        # Match the reference operation order for bitwise SF and ND comparisons.
        mu = internal * (1.0 / gain) if s.c["early"] == "SF" else internal / gain
        dv, x, _ = p.SFm.sf_dir(c.R, c.E, c.Srr, c.Pbar, c.eps, s.c["eps"], ret_y=True)
        applied = bool(s.restarts) and s.restarts[-1] == i - 1
        restarted = False
        if s.det is not None:
            rw = float(s.rw[i])
            z = p.rsn_stat(c.E, rw) if s.frame is None else (s.frame.energy / rw if rw > 0 else np.nan)
            if s.det.update(i, z):
                restarted = True
                s.restarts.append(i); s._close(i + 1, "restart")
                s.until = i + 1 + s.length; s.q = []; s.phases.append([i + 1, None, None])
        if early and not restarted and s.theta is not None:
            u = T.KD * x[:N // 2] / gain
            s.q.append(float(u @ u))
            if len(s.q) >= 8 and np.mean(s.q[-4:]) > s.theta * np.mean(s.q[-8:-4]):
                s.until = i + 1; s._close(i + 1, "qstop")
        s.tr["mu_app"].append(mu); s.tr["mu_state"].append(internal)
        s.tr["restart_apply_block"].append(i if applied else -1); s.tr["early"].append(float(early))
        return mu, dv

    def phase_lengths(s, nb):
        return [(nb if e is None else min(e, nb)) - a for a, e, _ in s.phases if a < nb]


def run_p2(data, Rall, st, c, nb, X4):
    rule = P2(c, data, nb, X4)
    frame = None
    if c["early"] == "ND":
        frame = Frame(data, rule); rule.frame = frame
    sim = p.simulate(data, Rall, st, rule, nb=nb, error_frame=frame)
    return rule, sim


# ----------------------------------------------------------------------------------------------
# Inputs: the protoC classes, the E5 classes, and secondary-path model errors (E6)
# ----------------------------------------------------------------------------------------------
def error_noise(dd, seed, level_db, spans=None):
    """Noise at the error microphone only (uncorrelated with the reference), level relative to the power of d over
    the first 10 s.  It is added to d, so that L1 (ANR) is measured against the uncontrolled error including it."""
    rng = np.random.default_rng(50_000 + seed)
    d = dd["d"]; P = float(np.mean(d[:10 * FS] ** 2))
    v = rng.standard_normal(len(d)) * math.sqrt(P * 10 ** (level_db / 10))
    if spans is not None:
        m = np.zeros(len(d), bool)
        for a, b in spans:
            m[int(a * FS):int(b * FS)] = True
        v = v * m
    dd["d"] = d + v; dd["v"] = v


def new_data(cls, seed):
    x = p.IC.make_input("A", seed)                              # the Gaussian input of class G
    if cls == "SC":                                             # colour change at 15 s, same power
        n0 = 15 * FS
        low = p.signal.lfilter([math.sqrt(1 - 0.9 ** 2)], [1.0, -0.9], x)
        dd = p.D.build(np.concatenate([x[:n0], low[n0:]])); dd["pc"] = False
    elif cls == "EVpc":
        dd = p.MFm.build_pc(x); dd["pc"] = True
        error_noise(dd, seed, -10.0)
    elif cls == "EV":
        dd = p.D.build(x); dd["pc"] = False
        error_noise(dd, seed, -10.0)
    else:
        level = {"ENm20": -20.0, "ENm10": -10.0, "EN0": 0.0}[cls]
        dd = p.D.build(x); dd["pc"] = False
        error_noise(dd, seed, level, spans=((10.0, 10.5), (20.0, 20.5)))
    return dd


def with_model(dd, shat, seed):
    """Replace the secondary-path model used by the controller (filtered reference, statistics, frame rebuild);
    the plant keeps the true hS.  g<a>: gain a;  d1: one-sample delay;  n<L>: white tap errors at -L dB."""
    hS = dd["hS"]
    if shat[0] == "g":
        hm = float(shat[1:]) * hS
    elif shat == "d1":
        hm = np.concatenate([[0.0], hS])
    elif shat[0] == "n":
        rng = np.random.default_rng(70_000 + seed)
        e = rng.standard_normal(len(hS))
        hm = hS + e * math.sqrt(float(hS @ hS) * 10 ** (-float(shat[1:]) / 10) / float(e @ e))
    else:
        raise ValueError(shat)
    out = dict(dd)
    out["hS_model"] = hm; out["Sf"] = np.fft.rfft(hm, N); out["r"] = p.signal.lfilter(hm, [1.0], dd["x"])
    q = np.abs(out["Sf"]) ** 2; out["S2"] = q / q.mean()
    return out


def prep(tag, nb=None, shat=None):
    cls, seed = tag.rsplit("_s", 1)
    if cls not in E5_CLASSES and shat is None:
        return p.prep3(tag, nb)
    if cls in E5_CLASSES:
        dd = new_data(cls, int(seed)); dd["tag"] = tag
    else:
        dd = p.data_pcb(tag) if cls == "pcb" else T.data(tag)
    if shat is not None:
        dd = with_model(dd, shat, int(seed))
    Rall = p.ML.prepass(dd)["Rall"]
    if nb is not None:
        Rall = Rall[:min(len(Rall), nb + 64)]
    return dd, Rall, p.Stats(dd, Rall)


# ----------------------------------------------------------------------------------------------
# Scores
# ----------------------------------------------------------------------------------------------
def db(v):
    return float(10 * np.log10(v)) if v is not None and np.isfinite(v) and v > 0 else None


def score(tag, name, rule, sim, data, keep_curve):
    cls, seed = tag.rsplit("_s", 1)
    m = p.metrics(sim, dict(data, pc=False))
    div = bool(m["div"]); nb, ex = sim["nb"], sim["ex"]
    row = {k: None for k in COLUMNS}
    row.update(tag=tag, seed=int(seed), rule=name, div_num=int(div), amp=int((not div) and m["amp"]))
    row["class"] = cls
    if not div:
        row.update(L1_whole=m["L1"][0], L1_first5=m["L1"][1], L1_last10=m["L1"][2],
                   ex_0_1s=p.dbm(ex[:BPS]) if nb >= BPS else None,
                   ex_0_5s=p.dbm(ex[:5 * BPS]) if nb >= 5 * BPS else None,
                   ex_last10=p.dbm(ex[-max(1, nb // 10):]))
        anr = p.O.anr_curve(sim["e"], data["d"][:len(sim["e"])]); ts = (63 + 64 * np.arange(len(anr))) / FS

        def win(a, b):
            return float(anr[(ts >= a) & (ts < b)].mean()) if nb >= b * BPS else None
        if data.get("pc"):
            row["L1_pc15_17"] = win(15, 17)
        if data.get("pseg"):
            row["L1_pcb10_12"], row["L1_pcb20_22"] = win(10, 12), win(20, 22)
        ev = [t for t in EVENTS.get(cls, ()) if nb >= (t + 2) * BPS]
        if ev:
            row["L1_ev"] = float(np.mean([win(t, t + 2) for t in ev]))
            row["ex_ev"] = p.dbm(np.concatenate([ex[t * BPS:(t + 2) * BPS] for t in ev]))
    for b in BLOCKS:
        if b <= len(ex):
            row[f"ex_b{b}"] = db(ex[b - 1])
    rst = getattr(rule, "restarts", None)
    has_det = rule.det is not None if isinstance(rule, P2) else getattr(rule, "detector", True) is not None
    if rst is not None and has_det:
        tr = sim["tr"]
        rs = p.restart_stats(rst, p.change_blocks(data), len(tr["mu_app"]), tr.get("restart_apply_block", ()))
        row.update(false_rs=rs["n_false"], missed_rs=rs["n_missed"], n_restart=len(rst))
    if isinstance(rule, P2):
        ln = rule.phase_lengths(len(sim["tr"]["mu_app"]))
        if ln:
            row.update(early_mean=float(np.mean(ln)), early_min=int(min(ln)), early_max=int(max(ln)))
        row["n_rebuilt"] = rule.frame.n_rebuilt if rule.frame is not None else 0
    if keep_curve:
        row["curve"] = [db(v) for v in ex[:CURVE_LEN]]
    for key, value in row.items():
        if isinstance(value, float) and not math.isfinite(value):
            row[key] = None
    return row


# ----------------------------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------------------------
def out_dir(smoke):
    return HERE / "out" / ("dev1_smoke" if smoke else "dev1")


def sel_path(smoke):
    return out_dir(smoke) / "selection.json" if smoke else HERE / "RESULTS_dev1_selection.json"


def load_sel(smoke):
    f = sel_path(smoke)
    return json.loads(f.read_text()) if f.exists() else {}


def base_cfgs(sel):
    end = tuple(sel["end"])
    return config("ND", sel["mu_e_ND"], end), config("SF", sel["mu_e_SF"], end)


def plan(exp, sel):
    """Returns [(classes, shat, [spec, ...])], spec = ('P2', cfg) or ('base', rule name)."""
    P = lambda *a, **k: ("P2", config(*a, **k))
    if exp == "E1":
        specs = [P("SF", m, ("fixed", 64), restart=False) for m in (0.3, 0.4, 0.5, 0.6, 0.678)]
        specs += [P("ND", m, ("fixed", 64), restart=False) for m in (0.7, 1.0, 1.25, 1.5)]
        return [(BASE6, None, specs)]
    if exp == "E2":
        ends = (("fixed", 32), ("fixed", 64), ("qstop", 0.7), ("qstop", 0.8), ("qstop", 0.9))
        specs = [P(k, sel[f"mu_e_{k}"], e) for k in ("SF", "ND") for e in ends]
        return [(BASE8, None, specs)]
    nd_, sf_ = base_cfgs(sel)
    if exp == "E3":
        return [(BASE8, None, [("P2", dict(nd_, delta=d)) for d in (0.01, 0.02, 0.03, 0.05, 0.1)])]
    if exp == "E4":
        return [(BASE8, None, [("P2", nd_), ("P2", sf_)] + [("base", b) for b in BASELINES])]
    if exp == "E5":
        return [(E5_CLASSES, None, [("P2", nd_), ("P2", sf_), ("P2", dict(nd_, restart=False))])]
    if exp == "E6":
        cl = ("G", "a145", "pc")
        return [(cl, s, [("P2", nd_), ("P2", sf_)]) for s in (None,) + SHATS]
    if exp == "E7":
        specs = [("P2", dict(c, eps=e)) for c in (nd_, sf_) for e in (1e-2, 1e-3, 1e-4)]
        specs += [("P2", dict(c, mu_e=c["mu_e"] * k)) for c in (nd_, sf_) for k in (1.5, 2.0, 3.0)]
        return [(("G", "BU", "II300", "a145"), None, specs)]
    raise ValueError(exp)


def spec_name(spec, shat):
    nm = name_of(spec[1]) if spec[0] == "P2" else spec[1]
    return nm if shat is None else f"{nm} | Shat {shat}"


def job_file(exp, tag, shat, smoke):
    return out_dir(smoke) / exp / (tag + ("" if shat is None else "_" + shat) + ".json")


def run_job(job):
    exp, tag, shat, specs, smoke = job
    f = job_file(exp, tag, shat, smoke)
    names = [spec_name(s, shat) for s in specs]
    if f.exists():
        old = json.loads(f.read_text())
        if [r["rule"] for r in old["rows"]] == names:
            return old["rows"]
    t0 = time.monotonic()
    short = 1250 if smoke else None
    data, Rall, st = prep(tag, short, shat)
    nb = short or len(data["d"]) // B
    o = p.parse(["c5", "--preset", "final"])
    base = ab.factories(data, nb, o) if any(s[0] == "base" for s in specs) else {}
    rows = []
    for spec, nm in zip(specs, names):
        if spec[0] == "P2":
            rule, sim = run_p2(data, Rall, st, spec[1], nb, o.X4)
        else:
            rule = base[spec[1]](); sim = p.simulate(data, Rall, st, rule, nb=nb)
        row = score(tag, nm, rule, sim, data, exp in ("E1", "E4")); row["exp"] = exp
        rows.append(row)
    f.parent.mkdir(parents=True, exist_ok=True)
    ab.atomic_write(f, lambda h: json.dump(dict(nb=nb, rows=rows), h))
    print(f"{exp} {f.stem}: {len(rows)} rules, div={sum(r['div_num'] for r in rows)}, {time.monotonic() - t0:.0f} s",
          flush=True)
    return rows


def run_exp(exp, procs, smoke):
    sel = load_sel(smoke)
    seeds = SEEDS[:1] if smoke else SEEDS
    jobs = [(exp, f"{c}_s{s}", shat, specs, smoke) for classes, shat, specs in plan(exp, sel)
            for c in classes for s in seeds]
    if procs == 1:
        batches = [run_job(j) for j in jobs]
    else:
        with Pool(procs) as pool:
            batches = list(pool.imap(run_job, jobs))
    rows = [r for b in batches for r in b]
    if exp == "E1":
        sel.update(select_e1(rows))
    elif exp == "E2":
        sel.update(select_e2(rows))
    if exp in ("E1", "E2"):
        sel_path(smoke).parent.mkdir(parents=True, exist_ok=True)
        ab.atomic_write(sel_path(smoke), lambda h: h.write(json.dumps(sel, indent=1) + "\n"))
        print("selection:", sel, flush=True)
    return rows


# ----------------------------------------------------------------------------------------------
# Aggregation (class-equal weights: mean over seeds within a class, then over classes)
# ----------------------------------------------------------------------------------------------
def finite(value):
    return value not in (None, "") and math.isfinite(float(value))


def cmean(rows, key):
    by = defaultdict(list)
    for r in rows:
        if finite(r.get(key)):
            by[r["class"]].append(float(r[key]))
    return float(np.mean([np.mean(v) for v in by.values()])) if by else None


def csum(rows, key):
    return int(sum(int(float(r[key])) for r in rows if finite(r.get(key))))


def by_rule(rows):
    g = defaultdict(list)
    for r in rows:
        g[r["rule"]].append(r)
    return g


def clean(rows):
    return all(r.get(k) == 0 for r in rows for k in ("div_num", "amp"))


def cfg_from_name(nm):
    m = re.match(r"P2 early=(SF|ND) mue([\d.]+) end=(fixed|qstop)([\d.]+) d([\d.]+)", nm)
    return m.group(1), float(m.group(2)), (m.group(3), float(m.group(4)) if m.group(3) == "qstop" else int(m.group(4)))


def select_e1(rows):
    """Per direction: smallest class-mean Phi-excess at block 32 among the steps with no divergence/amplification."""
    out = {}
    for kind in ("SF", "ND"):
        cand = {n: cmean(r, "ex_b32") for n, r in by_rule(rows).items()
                if cfg_from_name(n)[0] == kind and clean(r) and all(finite(t.get("ex_b32")) for t in r)}
        if not cand:
            raise ValueError(f"E1: no eligible {kind} setting with a complete block-32 excess")
        best = min(cand, key=cand.get)
        out[f"mu_e_{kind}"] = cfg_from_name(best)[1]
    return out


def select_e2(rows):
    """End rule from the early=ND results: no divergence / amplification / false or missed restart; smallest
    class-mean L1 whole; a qstop within 0.02 dB of the best is preferred."""
    val = {n: cmean(r, "L1_whole") for n, r in by_rule(rows).items()
           if cfg_from_name(n)[0] == "ND" and clean(r)
           and all(t.get("false_rs") == 0 and t.get("missed_rs") == 0 and finite(t.get("L1_whole")) for t in r)}
    if not val:
        raise ValueError("E2: no eligible ND setting with zero divergence, amplification, false or missed restarts")
    best = min(val, key=val.get)
    qs = [n for n in val if cfg_from_name(n)[2][0] == "qstop" and val[n] <= val[best] + 0.02]
    pick = min(qs, key=val.get) if qs else best
    return dict(end=list(cfg_from_name(pick)[2]), end_rule=pick)


# ----------------------------------------------------------------------------------------------
# Report: RESULTS_dev1.csv, RESULTS_dev1_curves.csv, RESULTS_dev1.md (tables only)
# ----------------------------------------------------------------------------------------------
def fmt(v, nd_=2):
    return f"{v:.{nd_}f}" if finite(v) else "—"


def md_cell(label):
    return label.replace("|", r"\|")


def table(rows, fields, counts=("div_num", "amp", "false_rs", "missed_rs")):
    head = ["rule", "n"] + list(fields) + list(counts)
    out = ["| " + " | ".join(head) + " |", "|---|" + "---:|" * (len(head) - 1)]
    for nm, rs in by_rule(rows).items():
        cells = [md_cell(nm), str(len(rs))] + [fmt(cmean(rs, f)) for f in fields]
        cells += [str(csum(rs, c)) if any(finite(r.get(c)) for r in rs) else "—" for c in counts]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def per_class(rows, field):
    classes = list(dict.fromkeys(r["class"] for r in rows))
    out = ["| rule | " + " | ".join(classes) + " |", "|---|" + "---:|" * len(classes)]
    for nm, rs in by_rule(rows).items():
        out.append("| " + md_cell(nm) + " | " + " | ".join(fmt(cmean([r for r in rs if r["class"] == c], field))
                                                 for c in classes) + " |")
    return "\n".join(out)


def paired(rows, a, b, field):
    """a - b per seed (mean over the classes where both are finite); mean [min, max] over seeds, wins of a."""
    g = by_rule(rows)
    ia = {(r["class"], r["seed"]): r.get(field) for r in g.get(a, [])}
    ib = {(r["class"], r["seed"]): r.get(field) for r in g.get(b, [])}
    per = defaultdict(list)
    for k in ia:
        if finite(ia[k]) and finite(ib.get(k)):
            per[k[1]].append(float(ia[k]) - float(ib[k]))
    d = [float(np.mean(v)) for v in per.values()]
    if not d:
        return "—"
    return f"{np.mean(d):+.3f} [{min(d):+.3f}, {max(d):+.3f}] ({sum(x < 0 for x in d)}/{len(d)})"


def load_rows(smoke):
    rows = []
    for exp in ("E1", "E2", "E3", "E4", "E5", "E6", "E7"):
        d = out_dir(smoke) / exp
        for f in sorted(d.glob("*.json")) if d.exists() else []:
            rows += json.loads(f.read_text())["rows"]
    return rows


def report(smoke):
    rows = load_rows(smoke); sel = load_sel(smoke)
    dest = out_dir(smoke) if smoke else HERE
    dest.mkdir(parents=True, exist_ok=True)

    def write_csv(h):
        w = csv.DictWriter(h, fieldnames=COLUMNS, lineterminator="\n", extrasaction="ignore"); w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None or (isinstance(v, float) and not math.isfinite(v)) else v)
                        for k, v in r.items()})
    ab.atomic_write(dest / "RESULTS_dev1.csv", write_csv)

    def write_curves(h):
        h.write("exp,class,rule,block,ex_db\n")
        g = defaultdict(list)
        for r in rows:
            if r.get("curve"):
                g[(r["exp"], r["class"], r["rule"])].append(r["curve"])
        for (exp, cls, rule), cs in g.items():
            for b in range(min(len(c) for c in cs)):
                v = [c[b] for c in cs if finite(c[b])]
                if v:
                    h.write(f"{exp},{cls},\"{rule}\",{b + 1},{np.mean(v):.3f}\n")
    ab.atomic_write(dest / "RESULTS_dev1_curves.csv", write_curves)

    E = lambda e: [r for r in rows if r["exp"] == e]
    main = ("L1_whole", "L1_first5", "L1_last10", "ex_0_1s", "ex_0_5s", "ex_last10")
    blk = tuple(f"ex_b{b}" for b in BLOCKS)
    chg = ("L1_pc15_17", "L1_pcb10_12", "L1_pcb20_22")
    L = ["# dev1：提案法（第2版 P2）の検証 E1–E7", "",
         "表は自動生成。値は dB（小さいほど良い）。クラス内で seed 平均、クラス間で等重み平均。件数は合計。",
         "対応差は「左 − 右」を seed ごとにクラス平均し、seed 間の平均 [最小, 最大]（左が良かった seed 数）。",
         f"後半のステップ μ_L(δ) = 2δ/(h(τ + (2M/B)δ))、δ = 0.03 で {mu_late(0.03):.4f}。", "",
         "## 選んだ設定", "", "```json", json.dumps(sel, indent=1, ensure_ascii=False), "```", ""]
    if E("E1"):
        L += ["## E1 前半のステップ（end=fixed64、再始動なし、6 クラス）", "", table(E("E1"), blk + main, ("div_num", "amp")), ""]
    if E("E2"):
        L += ["## E2 前半の終了（8 クラス、再始動あり）", "",
              table(E("E2"), main + chg + ("early_mean", "early_min", "early_max")), "",
              "### クラスごとの L1 whole", "", per_class(E("E2"), "L1_whole"), "",
              "### クラスごとの前半の長さ（平均ブロック数）", "", per_class(E("E2"), "early_mean"), ""]
    if E("E3"):
        L += ["## E3 後半の設計式（P2-ND、δ を変える）", "", table(E("E3"), main + chg), "",
              "### クラスごとの ex last10", "", per_class(E("E3"), "ex_last10"), ""]
    if E("E4"):
        r4 = E("E4"); names = list(by_rule(r4))
        L += ["## E4 基準との比較（8 クラス）", "", table(r4, main + chg + blk), "",
              "### クラスごとの L1 whole", "", per_class(r4, "L1_whole"), "",
              "### クラスごとの ex 0–1 s", "", per_class(r4, "ex_0_1s"), "", "### 対応差", "",
              "| 左 − 右 | L1 whole | L1 first5 | ex 0–1 s | L1 pc 15–17 | L1 pcb 10–12 | L1 pcb 20–22 |", "|---|---|---|---|---|---|---|"]
        for b in names[1:]:
            L.append(f"| {md_cell(names[0])} − {md_cell(b)} | " + " | ".join(
                paired(r4, names[0], b, f) for f in ("L1_whole", "L1_first5", "ex_0_1s") + chg) + " |")
        L.append("")
    if E("E5"):
        r5 = E("E5"); names = list(by_rule(r5))
        L += ["## E5 再始動の頑健性", "",
              "ENm20 / ENm10 / EN0：誤差マイク側の無相関な外乱（10–10.5 s、20–20.5 s、d に対して −20 / −10 / 0 dB）。"
              "EV：同じ外乱を −10 dB で常時。EVpc：EV ＋ 15 s の経路変化。SC：15 s で入力の色が変わる（経路は不変。"
              "SC の 15 s 以降の Φ 超過は白色入力の Wiener 解が基準なので参考値）。",
              "false_rs は経路変化のない時刻の再始動。L1_ev / ex_ev は各イベントの直後 2 秒。", ""]
        for c in E5_CLASSES:
            rc = [r for r in r5 if r["class"] == c]
            if rc:
                L += [f"### {c}", "", table(rc, ("L1_whole", "L1_last10", "ex_last10", "L1_ev", "ex_ev", "L1_pc15_17",
                                                 "early_mean", "early_min", "n_restart")), ""]
        if len(names) >= 3:
            L += ["### 再始動あり − 再始動なし（P2-ND）", "", "| クラス | L1 whole | L1_ev | ex_ev |", "|---|---|---|---|"]
            for c in E5_CLASSES:
                rc = [r for r in r5 if r["class"] == c]
                L.append(f"| {c} | " + " | ".join(paired(rc, names[0], names[2], f) for f in ("L1_whole", "L1_ev", "ex_ev")) + " |")
            L.append("")
    if E("E6"):
        r6 = E("E6")
        L += ["## E6 二次経路の推定誤差（G、a145、pc）", "",
              "g0.7 / g1.4：利得誤差。d1：1 サンプルの遅れ。n20：各タップに −20 dB の白色誤差。", "",
              table(r6, main + ("L1_pc15_17",) + blk), ""]
        base = [n for n in by_rule(r6) if "Shat" not in n]
        L += ["### Ŝ ≠ S − Ŝ = S", "", "| 規則 | L1 whole | L1 first5 | ex 0–1 s | L1 pc 15–17 |", "|---|---|---|---|---|"]
        for b in base:
            for s in SHATS:
                a = f"{b} | Shat {s}"
                L.append(f"| {md_cell(a)} | " + " | ".join(paired(r6, a, b, f) for f in ("L1_whole", "L1_first5", "ex_0_1s", "L1_pc15_17")) + " |")
        L.append("")
    if E("E7"):
        L += ["## E7 正則化 ε と安定の余裕（G、BU、II300、a145）", "", table(E("E7"), main + blk, ("div_num", "amp", "false_rs")), "",
              "### クラスごとの L1 whole", "", per_class(E("E7"), "L1_whole"), ""]
    L += ["## 再現手順", "", "```sh", "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1",
          "python prop2.py check", "python prop2.py run --exp all --procs <コア数>", "python prop2.py report", "```", ""]
    (dest / "RESULTS_dev1.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote", dest / "RESULTS_dev1.md", len(rows), "rows")


# ----------------------------------------------------------------------------------------------
# Check
# ----------------------------------------------------------------------------------------------
def check():
    assert abs(mu_late(0.03) - 0.0566) < 5e-5, mu_late(0.03)
    o = p.parse(["c5", "--preset", "final"])
    # 1. early=SF, fixed 64, mu_e 0.678, late 0.05 == ablation rule "SF early+fix+g+rsN2 mu0.05" (path change at 15 s)
    nb = 2100
    data, Rall, st = prep("pc_s10", nb)
    ref_rule = ab.factories(data, nb, o)["SF early+fix+g+rsN2 mu0.05"]()
    ref = p.simulate(data, Rall, st, ref_rule, nb=nb)
    rule, sim = run_p2(data, Rall, st, config("SF", o.mu_e, ("fixed", 64), mu_l=0.05), nb, o.X4)
    for k in ("e", "w", "ex"):
        np.testing.assert_array_equal(sim[k], ref[k])
    for k in ("mu_app", "restart_apply_block"):
        np.testing.assert_array_equal(sim["tr"][k], ref["tr"][k])
    assert rule.restarts == ref_rule.restarts and len(rule.restarts) == 1, (rule.restarts, ref_rule.restarts)
    print("check 1 ok: early=SF equals the ablation rule; restart at block", rule.restarts)
    # 2. early=ND: the early blocks equal nodelay's SF-ND+g+early; no rebuild afterwards
    nb = 96
    data, Rall, st = prep("G_s10", nb)
    ref = nd.simulate(data, Rall, st, "SF-ND+g+early", 0.05, 1.0, nb)
    rule, sim = run_p2(data, Rall, st, config("ND", 1.0, ("fixed", 64), restart=False, mu_l=0.05), nb, o.X4)
    np.testing.assert_array_equal(sim["ex"][:64], ref["ex"][:64])
    np.testing.assert_array_equal(sim["e"][:65 * B], ref["e"][:65 * B])
    np.testing.assert_array_equal(sim["tr"]["mu_app"][:64], ref["tr"]["mu_app"][:64])
    assert rule.frame.n_rebuilt == 64 and not np.allclose(sim["ex"][70:], ref["ex"][70:], rtol=1e-6)
    print("check 2 ok: early=ND equals SF-ND for the 64 early blocks; frames rebuilt:", rule.frame.n_rebuilt)
    # 3. a model equal to the true secondary path changes nothing; qstop ends before the limit and is recorded
    d2, R2, s2 = prep("G_s10", nb, "g1")
    _, sim2 = run_p2(d2, R2, s2, config("ND", 1.0, ("fixed", 64), restart=False, mu_l=0.05), nb, o.X4)
    for k in ("e", "w", "ex"):
        np.testing.assert_array_equal(sim2[k], sim[k])
    np.testing.assert_array_equal(sim2["tr"]["mu_app"], sim["tr"]["mu_app"])
    nb = 400
    data, Rall, st = prep("G_s10", nb)
    rule, sim = run_p2(data, Rall, st, config("ND", 1.0, ("qstop", 0.8)), nb, o.X4)
    a, e, why = rule.phases[0]
    assert a == 0 and why == "qstop" and 8 <= e < NE_MAX and rule.frame.n_rebuilt == e, rule.phases
    assert np.all(sim["tr"]["mu_state"][e:] == mu_late(0.03))
    print(f"check 3 ok: model = S is neutral; qstop phase {rule.phases}, excess at block 125 {db(sim['ex'][124]):.2f} dB")
    # 4. the E5 classes and the model errors build and run
    for tag, shat in (("ENm10_s10", None), ("EVpc_s10", None), ("SC_s10", None), ("G_s10", "d1"), ("G_s10", "n20")):
        data, Rall, st = prep(tag, 64, shat)
        _, sim = run_p2(data, Rall, st, config("ND", 1.0, ("qstop", 0.8)), 64, o.X4)
        assert np.isfinite(sim["ex"]).all(), (tag, shat)
    print("check 4 ok: E5 inputs and secondary-path model errors run")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("cmd", choices=("check", "run", "report"))
    a.add_argument("--exp", default="all")
    a.add_argument("--procs", type=int, default=os.cpu_count() or 1)
    a.add_argument("--smoke", action="store_true")
    o = a.parse_args()
    if o.cmd == "check":
        check()
    elif o.cmd == "run":
        for exp in (("E1", "E2", "E3", "E4", "E5", "E6", "E7") if o.exp == "all" else o.exp.split(",")):
            run_exp(exp, o.procs, o.smoke)
    else:
        report(o.smoke)


if __name__ == "__main__":
    main()
