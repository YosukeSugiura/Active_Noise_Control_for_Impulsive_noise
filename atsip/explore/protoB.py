"""protoB: prototype experiment B (P3 of EXPERIMENTS_vssT_v2.md; THEORY_vss2.md Sec. 3 and Sec. 9.5).

Concentration law with a KNOWN scale trajectory, fixed normaliser, fixed coefficients, SF direction.
  x_t = SIG0 * sigma_t * xi_t, xi white Gaussian; R independent xi realisations (the same xi for every condition,
  common random numbers); one target block n whose reference frame is x[F0, F0 + N) and whose error frame is the
  last M samples of it, [F0 + N - M, F0 + N).
  Normaliser: the same frozen Gaussian steady state as protoA (time average of the G_s0 run), for all conditions.
  Coefficients fixed: w = w_fp (SF model fixed point) -> zeta = Delta(w_fp) (noise);
                      v = -w_fp (w = 0)               -> A v = Delta of the coherent error r * v (mean + H~v).
  Every condition has the same energy sum sigma^2 = N in the reference frame of block n (g_sigma = 1), except
  where stated.

Per condition (Euclidean and Phi): mean action h_eff = <E A v, v>/|v|^2 and |E A v|^2/(h^2|v|^2); fluctuation
  E|A v - E A v|^2 / (h^2 |v|^2) (= f); noise E|zeta - E zeta|^2 (centred) and E|zeta|^2 (raw), |E zeta|^2;
  pi (true, from sigma) over the N-sample reference frame (pi_x) and over the last-M error window (pi_xM);
  actual frame gain g = mean|R|^2/Pbar (E g, E g^2).
Predictions (Sec. 3 / Sec. 9.5), relative to the stationary uniform condition: mean ~ g; noise ~ g^2 (1 - pi);
  fluctuation ~ g^2 (Sec. 9.5) or ~ g^2 (1 - pi) (old Sec. 3).

Subcommands: check | run [--R 200] [--seed] [--smoke] | report [--smoke]
Outputs: out/protoB/protoB.json (full), out/protoB_smoke/ (smoke).  Report: RESULTS_protoB.md.
"""
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vssT_lib import *        # noqa: F401,F403
import protoA as PA

OUT_FULL = os.path.join(HERE, "out", "protoB")
OUT_SMOKE = os.path.join(HERE, "out", "protoB_smoke")
SIG0 = PA.SIG0
F0 = 9 * B                      # start of the reference frame of the target block (local samples)
LLOC = F0 + N                   # local signal length; the target block ends at the end of the signal
EW0 = N - M                     # frame offset where the error window starts (512)


def cond_list():
    """[(name, group, sigma^2 array (units of SIG0^2))]."""
    C = []
    s = np.ones(LLOC); C.append(("uniform (stationary)", "ref", s))
    s = np.zeros(LLOC); s[F0:] = 1.0; C.append(("uniform (frame only)", "uniform", s))
    for o in (0, 256, 511, 512, 640, 768, 1023):
        s = np.zeros(LLOC); s[F0 + o] = N
        C.append((f"1 pulse @{o}{' (in err. window)' if o >= EW0 else ''}", "pulse1 position", s))
    for sp in (1, 16, 128, 512):
        s = np.zeros(LLOC); s[F0 + 768] = N / 2; s[F0 + 768 - sp] = N / 2
        C.append((f"2 pulses spacing {sp} (@{768 - sp},768)", "pulse2 spacing", s))
    for o in (0, 248, 504, 640, 1007):
        s = np.zeros(LLOC); s[F0 + o] = N / 2; s[F0 + o + 16] = N / 2
        C.append((f"2 pulses spacing 16 @{o},{o + 16}", "pulse2 position", s))
    for k in (8, 4, 2, 1):
        s = np.ones(LLOC); s[F0 - k * B] += N
        C.append((f"stationary + earlier pulse {k} block(s) before", "tail (stationary base)", s))
    for d in (127, 64, 16):
        s = np.ones(LLOC); s[F0 - d] += N
        C.append((f"stationary + earlier pulse {d} samples before", "tail (stationary base)", s))
    for k in (8, 4, 2, 1):
        s = np.zeros(LLOC); s[F0 + 768] = N; s[F0 - k * B] = N
        C.append((f"1 pulse @768 + earlier pulse {k} block(s) before", "tail (pulse base)", s))
    for d in (127, 64, 16):
        s = np.zeros(LLOC); s[F0 + 768] = N; s[F0 - d] = N
        C.append((f"1 pulse @768 + earlier pulse {d} samples before", "tail (pulse base)", s))
    return C


def pis(s2):
    fr = s2[F0:F0 + N]; ew = s2[F0 + EW0:F0 + N]
    px = float(np.sum(fr ** 2) / np.sum(fr) ** 2)
    pm = float(np.sum(ew ** 2) / np.sum(ew) ** 2) if np.sum(ew) > 0 else float("nan")
    return px, pm, float(np.sum(fr) / N), float(np.sum(ew) / M)


class Probe:
    def __init__(s, S, R, seed):
        s.S = S; s.R = R
        s.xi = np.random.default_rng(seed).standard_normal((R, LLOC))
        s.dirf = PA.FixedSF(S["Srr"], S["Pbar"], S["eps"])
        s.wfp = S["wfp"]; s.v = -S["wfp"]

    def measure(s, s2, keep=False):
        S = s.S
        x = SIG0 * np.sqrt(s2)[None] * s.xi
        d = signal.lfilter(S["hP"], [1.0], x, axis=1); r = signal.lfilter(S["hS"], [1.0], x, axis=1)
        Rf = S["Sf"] * np.fft.rfft(x[:, F0:F0 + N], axis=1)
        efp = d + signal.lfilter(s.wfp, [1.0], r, axis=1)
        ec = signal.lfilter(s.v, [1.0], r, axis=1)
        fr = lambda e: np.fft.rfft(np.concatenate([np.zeros((s.R, N - M)), e[:, F0 + EW0:F0 + N]], 1), axis=1)
        z = KD * s.dirf(Rf, fr(efp)); Av = KD * s.dirf(Rf, fr(ec))
        g = np.mean(np.abs(Rf) ** 2, axis=1) / S["Pbar"]
        out = dict(Eg=float(g.mean()), Eg2=float((g ** 2).mean()))
        v = s.v
        for ip in ("E", "P"):
            W = (lambda a: a) if ip == "E" else (lambda a: a @ Phi)
            vv = float(v @ W(v))
            m = Av.mean(0); dm = Av - m
            zm = z.mean(0); dz = z - zm
            out[ip] = dict(h_eff=float(m @ W(v)) / vv, mean2=float(m @ W(m)) / (H ** 2 * vv),
                           fluct=float(np.mean(np.sum(dm * W(dm), 1))) / (H ** 2 * vv),
                           noise_c=float(np.mean(np.sum(dz * W(dz), 1))), noise_raw=float(np.mean(np.sum(z * W(z), 1))),
                           zeta_mean2=float(zm @ W(zm)))
        if keep:
            out["_z"], out["_Av"] = z, Av
        return out


def job(o):
    t0 = time.time()
    S = PA.setup(states=False)
    pr = Probe(S, o.R, o.seed)
    res = dict(R=o.R, seed=o.seed, F0=F0, N=N, M=M, h=H, Jo=Jo, conds=[])
    for name, grp, s2 in cond_list():
        px, pm, gs, gw = pis(s2)
        r = pr.measure(s2)
        r.update(name=name, group=grp, pi_x=px, pi_xM=pm, g_sigma=gs, g_sigma_errwin=gw)
        res["conds"].append(r)
    res["sec"] = time.time() - t0
    d = OUT_SMOKE if o.smoke else OUT_FULL
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "protoB.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"protoB: {len(res['conds'])} conditions, R = {o.R} ({res['sec']:.0f} s)", flush=True)


def check(o):
    ok = True

    def rep(name, cond, msg=""):
        nonlocal ok
        ok &= bool(cond); print(f"[{'OK ' if cond else 'FAIL'}] {name} {msg}", flush=True)

    C = {n: s2 for n, g, s2 in cond_list()}
    p1 = pis(C["1 pulse @768 (in err. window)"])
    rep("pi of a single pulse = 1", abs(p1[0] - 1) < 1e-15 and abs(p1[1] - 1) < 1e-15, f"{p1[0]}, {p1[1]}")
    pu = pis(C["uniform (stationary)"])
    rep("pi of the uniform frame = 1/N, 1/M", abs(pu[0] - 1 / N) < 1e-15 and abs(pu[1] - 1 / M) < 1e-15)
    p2 = pis(C["2 pulses spacing 16 (@752,768)"])
    rep("pi of two equal pulses = 0.5", abs(p2[0] - 0.5) < 1e-15)
    gs = [pis(s2)[2] for s2 in C.values()]
    rep("same frame energy in every condition (g_sigma = 1)", np.max(np.abs(np.array(gs) - 1)) < 1e-12)
    S = PA.setup(states=False)
    pr = Probe(S, 8, 1)
    base = pr.measure(C["uniform (stationary)"], keep=True)
    for k in (1, 2, 4, 8):
        t = pr.measure(C[f"stationary + earlier pulse {k} block(s) before"], keep=True)
        dz = np.max(np.abs(t["_z"] - base["_z"])); da = np.max(np.abs(t["_Av"] - base["_Av"]))
        rep(f"earlier pulse {k} block(s) before the frame: directions unchanged (fixed normaliser)", dz == 0 and da == 0,
            f"max diff {max(dz, da):.1e}")
    t = pr.measure(C["stationary + earlier pulse 16 samples before"], keep=True)
    rep("earlier pulse 16 samples before: the residual tail reaches the error window (directions change)",
        np.max(np.abs(t["_z"] - base["_z"])) > 0)
    # bilinearity: sigma x 2 -> mean x 4, noise x 16
    b2 = pr.measure(4 * C["uniform (stationary)"])
    rep("bilinear scaling (sigma x2: h_eff x4, noise x16, fluct x16)",
        abs(b2["E"]["h_eff"] / base["E"]["h_eff"] - 4) < 1e-9 and abs(b2["E"]["noise_c"] / base["E"]["noise_c"] - 16) < 1e-9
        and abs(b2["E"]["fluct"] / base["E"]["fluct"] - 16) < 1e-9)
    # one realisation of the uniform stationary condition == FixedSF on a frame built by protoA.Ens (same signal)
    ens = PA.Ens(S, 1, LLOC // B, None, x=SIG0 * pr.xi[:1].copy())
    bn = LLOC // B - 1
    z1 = ens.Dd(bn)[0] + ens.A(S["wfp"], bn)[0]
    rep("Probe zeta == protoA.Ens direction at w_fp (same signal)", np.max(np.abs(z1 - base["_z"][0])) < 1e-9 * np.max(np.abs(z1)),
        f"{np.max(np.abs(z1 - base['_z'][0])):.1e}")
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED", flush=True)
    return ok


def report(o):
    d = OUT_SMOKE if o.smoke else OUT_FULL
    p = os.path.join(d, "protoB.json")
    fm = PA.fm
    L = ["# RESULTS protoB: concentration law with known scale, fixed normaliser, fixed coefficients (THEORY_vss2.md Sec. 3, 9.5)", "",
         f"Generated by `python protoB.py report{' --smoke' if o.smoke else ''}` from {os.path.relpath(p, HERE)}. Prototype diagnostics.", ""]
    if not os.path.exists(p):
        open(os.path.join(HERE, "RESULTS_protoB.md") if not o.smoke else os.path.join(d, "RESULTS_protoB_smoke.md"), "w").write("\n".join(L + ["(no outputs)"]) + "\n")
        return
    J = json.load(open(p))
    L += [f"R = {J['R']} xi realisations (same xi in every condition), seed {J['seed']}. Target block: reference frame = "
          f"x[{J['F0']}, {J['F0'] + N}) (local samples), error window = its last M = {M} samples (frame offsets {EW0}..{N - 1}). "
          "Normaliser frozen at the Gaussian steady state. Coefficients fixed: noise zeta at w = w_fp; mean and fluctuation of "
          "A v for v = -w_fp. g_sigma = sum sigma^2 over the frame / N (= 1 in every condition). Ratios are relative to "
          "'uniform (stationary)'. pi_x: over the N-sample reference frame; pi_xM: over the M-sample error window "
          "(both from sigma, not from the sample).", "",
          "Predictions of the ratios (g_sigma = 1): mean 1; noise (1 - pi)/(1 - pi_ref) [Sec. 3]; fluctuation 1 [Sec. 9.5] "
          "or (1 - pi)/(1 - pi_ref) [old Sec. 3]; pred columns use pi_x.", ""]
    ref = J["conds"][0]
    for ip in ("E", "P"):
        rE = ref[ip]
        L += [f"## Inner product {ip}", "",
              f"Reference (uniform stationary): h_eff/h = {fm(rE['h_eff'] / H)}, f = {fm(rE['fluct'])}, noise (centred) = {fm(rE['noise_c'])}, "
              f"|E zeta|^2 = {fm(rE['zeta_mean2'])}.", "",
              "| condition | group | pi_x | pi_xM | g_sigma (err. win.) | E g | E g^2 | mean/g ratio | fluct/g^2 ratio | noise/g^2 ratio | raw noise ratio | |E zeta|^2/noise_ref | pred noise (1-pi) | pred fluct 9.5 | pred fluct old |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in J["conds"]:
            x = c[ip]; gs = c["g_sigma"]
            pnoise = gs ** 2 * (1 - c["pi_x"]) / (1 - ref["pi_x"])
            L.append("| " + " | ".join([c["name"], c["group"], fm(c["pi_x"]), fm(c["pi_xM"]), fm(c["g_sigma_errwin"]), fm(c["Eg"]), fm(c["Eg2"]),
                                         fm(x["h_eff"] / gs / rE["h_eff"]), fm(x["fluct"] / gs ** 2 / rE["fluct"]),
                                         fm(x["noise_c"] / gs ** 2 / rE["noise_c"]), fm(x["noise_raw"] / gs ** 2 / rE["noise_raw"]),
                                         fm(x["zeta_mean2"] / rE["noise_c"]), fm(pnoise), fm(gs ** 2), fm(pnoise)]) + " |")
        L.append("")
    path = os.path.join(HERE, "RESULTS_protoB.md") if not o.smoke else os.path.join(d, "RESULTS_protoB_smoke.md")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", path)


def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "run", "report"])
    ap.add_argument("--R", type=int, default=200); ap.add_argument("--seed", type=int, default=54321)
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args(argv)


if __name__ == "__main__":
    o = parse(sys.argv[1:])
    if o.cmd == "check":
        sys.exit(0 if check(o) else 1)
    elif o.cmd == "report":
        report(o)
    else:
        job(o)
