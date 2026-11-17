"""protoA: prototype experiment A (P1 + P2 of EXPERIMENTS_vssT_v2.md, THEORY_vss2.md Sec. 9.1-9.4).

Branched short segments, white Gaussian input, SF direction, FIXED normaliser (Srr, Pbar, eps and the SF
minimum-phase factor frozen at the time average of a normal G_s0 run, blocks 500..), g treated as 1.

For one coefficient state W0 (i: w = 0, i.e. v = -wM; ii / iii: w of a fixed-mu 0.1 SF run at blocks 30 / 2000):
  R independent Gaussian realisations (common random numbers across mu).  Per realisation the local signal has
  WARM + PRE + T blocks; w is frozen at W0 for the first WARM + PRE blocks (the frame before the branch point is
  generated consistently with W0), the adaptation starts at local block r0 = WARM + PRE and runs T blocks with
  w <- w - mu Delta, Delta = KD * SF-direction (E Delta = h v, h = 0.25).  The first WARM = 8 blocks are discarded
  (reference frame and filters filling up); the PRE recorded frozen blocks give the EMA history and the lag
  products that reach back before the branch point.

Recorded (all exact ensemble averages over the R realisations, no fitted constants):
  * Gram matrices E<Delta_t, Delta_s>, E<v_t, Delta_s>, E<v_t, v_s> over all recorded blocks (Euclidean and Phi),
    for v relative to wM and to the SF model fixed point.  Everything in Sec. 9.1 (C_l, Q, the EMA versions for any
    (L, beta_d), the outer EMA, the four-way split h^2<v,v> + h<v,eta> + h<eta,v> + <eta,eta>) is computed from
    these offline in `report`.
  * frame delay (Sec. 9.2): Delta^re_t = direction from the error frame re-generated with w_t only;
    |E Delta - E Delta^re| / |E Delta|; occupancy weights c_k (a) at mu = 0 from the error frame split by the
    block whose coefficients produced each output sample (exact operator split, k = 0..4), (b) by regressing
    E(Delta_t - Delta^re_t) on h (E w_{t-k} - E w_t), k = 1..4, over mu > 0.
  * frozen decomposition at mu = 0: Delta_t = A_t w + dir(d); eta = Delta - E Delta; zeta (ref) = dir(d) + A_t w_ref,
    H~V = (A_t - A_bar) (W0 - w_ref); Gamma_k, K_eta, N, S, f, tau_a, tau_m, cross terms.
  * P2 (Sec. 9.4) at mu = 0: T1(m) = sum_{k=1..m} E<d_{n-k}, H~_n V>, T2(m) = sum_{k=1..m} E<V, H~_n d_{n-k}>
    (the two terms of the RHS with n - r = m; at mu = 0 the record is stationary, so the sum over j = r..n-1 is
    evaluated as a lag sum averaged over all origins n in the record).  d_j - E d_j = eta_j is used in T1
    (E H~_n V = 0) and E<V, H~_n d> = E<V, A_n (d - E d)> in T2 (A_bar deterministic), both exact identities that
    only remove variance.  Also the finite-difference slope of E<v_n, H~_n v_n> from the mu > 0 branches
    (A_bar applied to E v_n is the same-sample mean, the second-order term E<dv, A_bar dv> uses the operator of
    another realisation).

Subcommands:  check | run [--states i,ii,iii] [--R 64] [--T 40] [--pre 48] [--procs 2] [--smoke] [--force] | report [--smoke]
Outputs: out/protoA/protoA_<state>.npz + .json (full), out/protoA_smoke/ (smoke).  Report: RESULTS_protoA.md.
"""
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vssT_lib import *        # noqa: F401,F403  np, signal, linalg, M, N, B, KD, H, NBIN, wM, Jo, Phi, O, SFm, ...
import vssT_lib as T
from numpy.lib.stride_tricks import sliding_window_view
from multiprocessing import Pool

OUT_FULL = os.path.join(HERE, "out", "protoA")
OUT_SMOKE = os.path.join(HERE, "out", "protoA_smoke")
SIG0 = float(np.sqrt(2.0))      # std of the G input (SaS alpha = 2, scale 1 -> variance 2)
MUS = (0.0, 0.0125, 0.025, 0.05, 0.1)
STATES = ("i", "ii", "iii")
STATE_DESC = {"i": "w = 0 (v = -wM)", "ii": "w of a fixed-mu 0.1 SF run (G_s0) at block 30",
              "iii": "w of a fixed-mu 0.1 SF run (G_s0) at block 2000"}
STATE_BLOCK = {"ii": 30, "iii": 2000}
REFS_ = ("fp", "wM")
IPS = ("E", "P")                # Euclidean, Phi
WARM = 8                        # first valid local block (reference frame N = 8 blocks; d / r*w filled)
LMAX = 16                       # raw lags of C_l and P2 horizon
KC = 4                          # occupancy k = 0..KC (M-sample frame = 4 blocks + secondary-path tail)
NCK = 16                        # mu = 0 blocks used for the masked c_k
EPS_REG = 1e-2
NORM_FROM = 500                 # frozen normaliser = mean over blocks >= NORM_FROM of the G_s0 run statistics


def outdir(smoke):
    d = OUT_SMOKE if smoke else OUT_FULL
    os.makedirs(d, exist_ok=True)
    return d


# ----------------------------------------------------------------------------------------------
# Fixed normaliser, paths, states
# ----------------------------------------------------------------------------------------------
def paths():
    hP = O.pad_or_trim(np.loadtxt(os.path.join(O.ROOT, "impulse_response/primary_ir.dat"), ndmin=1), M)
    hS = O.pad_or_trim(np.loadtxt(os.path.join(O.ROOT, "impulse_response/secondary_ir.dat"), ndmin=1), O.C_SEC)
    return hP, hS


def setup(states=True):
    """Frozen normaliser from G_s0 (time average of Srr, Pbar, eps over blocks >= NORM_FROM), w states, refs."""
    dd, Rall, st = T.prep("G_s0")
    Srr = st.Srr[NORM_FROM:].mean(0); Pbar = float(st.Pbar[NORM_FROM:].mean()); eps = float(st.eps[NORM_FROM:].mean())
    hP, hS = paths()
    S = dict(Srr=Srr, Pbar=Pbar, eps=eps, hP=hP, hS=hS, Sf=dd["Sf"], wM=wM.copy(),
             wfp=wM + fixed_point(("SF",), 0), g_run=float(st.g[NORM_FROM:].mean()))
    if states:
        snaps = {}

        def hk(i, w, *a, **k):
            if i in STATE_BLOCK.values():
                snaps[i] = w.copy()
        simulate(dd, Rall, st, Fixed("SF", 0.1), nb=max(STATE_BLOCK.values()) + 1, hook=hk)
        S["W"] = {"i": np.zeros(M), "ii": snaps[STATE_BLOCK["ii"]], "iii": snaps[STATE_BLOCK["iii"]]}
    S["_dd"], S["_Rall"], S["_st"] = dd, Rall, st
    return S


class FixedSF:
    """SFm.sf_dir with the normaliser (Lam and its minimum-phase factor F) frozen; batched over leading axes."""

    def __init__(s, Srr, Pbar, eps, eps_reg=EPS_REG):
        s.args = (Srr, Pbar, eps, eps_reg)
        Lam = np.maximum(Srr + eps + eps_reg * Pbar, 1e-300)
        full = np.concatenate([Lam, Lam[-2:0:-1]])
        c = np.fft.ifft(0.5 * np.log(full)).real
        c[1:N // 2] *= 2.0
        c[N // 2 + 1:] = 0.0
        s.F = np.exp(np.fft.rfft(c)); s.cF = np.conj(s.F)

    def __call__(s, R, E):
        x = np.fft.irfft(np.conj(R) * E / s.cF, N, axis=-1)
        x[..., N // 2:] = 0.0
        return np.fft.irfft(np.fft.rfft(x, axis=-1) / s.F, N, axis=-1)[..., :M]


# ----------------------------------------------------------------------------------------------
# Ensemble of local realisations
# ----------------------------------------------------------------------------------------------
class Ens:
    def __init__(s, S, R, nblk, seed, x=None):
        s.S, s.R, s.nb = S, R, nblk
        L = nblk * B
        s.x = SIG0 * np.random.default_rng(seed).standard_normal((R, L)) if x is None else x
        s.d = signal.lfilter(S["hP"], [1.0], s.x, axis=1)
        s.r = signal.lfilter(S["hS"], [1.0], s.x, axis=1)
        s.hS = S["hS"]
        s.dirf = FixedSF(S["Srr"], S["Pbar"], S["eps"])
        s.Rf = np.zeros((R, nblk, NBIN), complex)
        for b in range(WARM - 1, nblk):
            s.Rf[:, b] = S["Sf"] * np.fft.rfft(s.x[:, b * B + B - N:b * B + B], axis=1)
        s.g = np.mean(np.abs(s.Rf) ** 2, axis=2) / S["Pbar"]          # (R, nblk), actual gain (recorded only)

    @staticmethod
    def win(b):
        return b * B + B - M, b * B + B

    @staticmethod
    def frameE(seg):
        z = np.zeros(seg.shape[:-1] + (N,)); z[..., N - M:] = seg
        return np.fft.rfft(z, axis=-1)

    def D(s, b, seg, roll=0):
        """Delta = KD * SF direction for block b from error-frame samples seg (R, M)."""
        Rb = s.Rf[:, b] if roll == 0 else np.roll(s.Rf[:, b], roll, axis=0)
        return KD * s.dirf(Rb, s.frameE(seg))

    def rF(s, b, roll=0):
        a, e = s.win(b)
        rr = s.r[:, a - M + 1:e]
        if roll:
            rr = np.roll(rr, roll, axis=0)
        return np.fft.rfft(rr, 2 * N, axis=1)

    def coh(s, w, b, rF=None, roll=0):
        """(r * w) over the error-frame window of block b; w (M,) or (R, M)."""
        rF = s.rF(b, roll) if rF is None else rF
        return np.fft.irfft(rF * np.fft.rfft(w, 2 * N, axis=-1), 2 * N, axis=-1)[:, M - 1:2 * M - 1]

    def A(s, w, b, rF=None, roll=0):
        """A_b w: Delta of the coherent error r * w with w frozen (linear in w)."""
        return s.D(b, s.coh(w, b, rF, roll), roll)

    def Dd(s, b):
        a, e = s.win(b)
        return s.D(b, s.d[:, a:e])

    def branch(s, W0, mu, r0):
        """Actual loop (identical arithmetic to vssT_lib.simulate up to FFT round-off): w frozen at W0 before local
        block r0, then w <- w - mu Delta.  Yields (b, w_b (pre-update), Delta_b) for b >= r0."""
        hS = s.hS; nh = len(hS)
        y = signal.lfilter(W0, [1.0], s.x, axis=1)
        e = s.d + signal.lfilter(hS, [1.0], y, axis=1)
        w = np.repeat(W0[None], s.R, 0)
        for b in range(r0, s.nb):
            s0 = b * B
            seg = s.x[:, s0 - M + 1:s0 + B]
            y[:, s0:s0 + B] = np.fft.irfft(np.fft.rfft(seg, 2 * M, axis=1) * np.fft.rfft(w, 2 * M, axis=1),
                                           2 * M, axis=1)[:, M - 1:M - 1 + B]
            ys = signal.fftconvolve(y[:, s0 - nh + 1:s0 + B], hS[None], mode="valid", axes=1)
            e[:, s0:s0 + B] = s.d[:, s0:s0 + B] + ys
            a, en = s.win(b)
            Dl = s.D(b, e[:, a:en])
            yield b, w.copy(), Dl
            w = w - mu * Dl

    def masked(s, v, b, kmax=KC):
        """Split A_b v by the block (b - k) that produced the output samples y = x * v: returns list over k."""
        y = signal.lfilter(v, [1.0], s.x, axis=1)
        a, e = s.win(b); nh = len(s.hS)
        out = []
        for k in range(kmax + 1):
            ym = np.zeros((s.R, e - a + nh - 1)); lo = a - nh + 1
            c0, c1 = max((b - k) * B, lo), min((b - k + 1) * B, e)
            if c1 > c0:
                ym[:, c0 - lo:c1 - lo] = y[:, c0:c1]
            seg = signal.fftconvolve(ym, s.hS[None], mode="valid", axes=1)
            out.append(s.D(b, seg))
        return out


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------
def ipmat(ip):
    return None if ip == "E" else Phi


def gram(Aa, Bb, ip):
    """G[t, s] = mean_R <A_t, B_s> (A, B: (R, T, M))."""
    if ip == "P":
        Bb = Bb @ Phi
    return np.einsum("rtm,rsm->ts", Aa, Bb, optimize=True) / Aa.shape[0]


def dot(a, b, ip):
    """mean over leading axis of <a, b> (a, b (R, M) or (M,))."""
    if ip == "P":
        b = b @ Phi
    return float(np.mean(np.sum(a * b, axis=-1)))


def lagmean(G, kmax):
    """g[k] = mean_t G[t, t-k], k = 0..kmax (G[t, s] = <X_t, Y_s>)."""
    return np.array([np.mean(np.diagonal(G, -k)) for k in range(kmax + 1)])


def lagmean2(G, kmax):
    """two-sided: g[k + kmax] = mean_t G[t, t-k], k = -kmax..kmax."""
    return np.array([np.mean(np.diagonal(G, -k)) for k in range(-kmax, kmax + 1)])


# ----------------------------------------------------------------------------------------------
# The job (one state)
# ----------------------------------------------------------------------------------------------
def job(args):
    state, o = args
    t0 = time.time()
    S = setup(states=True)
    W0 = S["W"][state]
    T_, PRE, R = o.T, o.pre, o.R
    nblk = WARM + PRE + T_
    r0 = WARM + PRE
    nrec = nblk - WARM
    ens = Ens(S, R, nblk, o.seed)
    refs = {"fp": S["wfp"], "wM": S["wM"]}
    Vr = {k: W0 - w for k, w in refs.items()}
    PV = {k: Phi @ v for k, v in Vr.items()}
    res = dict(state=state, desc=STATE_DESC[state], R=R, T=T_, pre=PRE, warm=WARM, r0=r0, mus=list(MUS), seed=o.seed,
               lmax=LMAX, kc=KC, h=H, Jo=Jo, norm=dict(Pbar=S["Pbar"], eps=S["eps"], g_run=S["g_run"]),
               V={k: dict(xE=float(v @ v), xP=float(v @ PV[k]), exP_dB=float(10 * np.log10(v @ PV[k] / Jo)))
                  for k, v in Vr.items()},
               W0_exP_wM_dB=float(10 * np.log10(exP(W0 - wM, Jo))), g_mean=float(ens.g[:, WARM:].mean()),
               g2_mean=float((ens.g[:, WARM:] ** 2).mean()))
    arr = {}
    # ---------------- mu-independent: dir(d), A_b W0, A_b w_ref for all recorded blocks
    DD = np.empty((R, nrec, M)); AW = np.empty_like(DD); Aref = {k: np.empty_like(DD) for k in refs}
    for ib in range(nrec):
        b = WARM + ib; rF = ens.rF(b)
        DD[:, ib] = ens.Dd(b); AW[:, ib] = ens.A(W0, b, rF)
        for k, w in refs.items():
            Aref[k][:, ib] = ens.A(w, b, rF)
    D0 = DD + AW                                   # Delta at mu = 0 (frozen W0)
    print(f"[A {state}] frozen directions ({time.time() - t0:.0f} s)", flush=True)
    # ---------------- frozen decomposition (mu = 0, stationary record)
    mD = D0.mean((0, 1))
    eta = D0 - mD
    fr = dict()
    for ip in IPS:
        Ge = gram(eta, eta, ip)
        arr[f"Geta_{ip}"] = lagmean(Ge, nrec - 1)
        fr[ip] = dict(EDelta2=dot(mD, mD, ip), Q0=float(np.trace(gram(D0, D0, ip)) / nrec))
    for k in refs:
        zt = DD + Aref[k]; zm = zt.mean((0, 1)); zt = zt - zm
        mv = AW - Aref[k]; mm = mv.mean((0, 1)); mv = mv - mm          # H~ V_ref (mean over R and blocks removed)
        for ip in IPS:
            Gz, Gm, Gzm = gram(zt, zt, ip), gram(mv, mv, ip), gram(zt, mv, ip)
            Gem = gram(eta, mv, ip)                                 # <eta_t, m_s>
            arr[f"Gz_{ip}_{k}"] = lagmean(Gz, nrec - 1); arr[f"Gm_{ip}_{k}"] = lagmean(Gm, nrec - 1)
            arr[f"Gzm_{ip}_{k}"] = lagmean2(Gzm, LMAX)             # E<zeta_n, m_{n-k}>, k=-LMAX..LMAX
            # P2, term 1: c1[k] = E<eta_{n-k}, H~_n V> = mean_t Gem[t - k, t]  (k = 0..LMAX)
            arr[f"c1_{ip}_{k}"] = np.array([np.mean(np.diagonal(Gem, kk)) for kk in range(LMAX + 1)])
            arr[f"c1z_{ip}_{k}"] = np.array([np.mean(np.diagonal(Gzm, kk)) for kk in range(LMAX + 1)])
            v = Vr[k]; X = H ** 2 * (float(v @ v) if ip == "E" else float(v @ PV[k]))
            fr[ip][k] = dict(X=X, h_eff=dot(mm + 0 * v, v, ip) / dot(v, v, ip), Ezeta2=dot(zm, zm, ip),
                             nonscalar=0.0)
            hm = mm - fr[ip][k]["h_eff"] * v
            fr[ip][k]["nonscalar"] = dot(hm, hm, ip) / (fr[ip][k]["h_eff"] ** 2 * dot(v, v, ip))
            fr[ip][k]["mean_vs_hV"] = dot(mm - H * v, mm - H * v, ip) / (H ** 2 * dot(v, v, ip))
        del zt, mv
    res["frozen"] = fr
    # P2, term 2: c2[k] = E<V, A_n eta_{n-k}>  (k = 1..LMAX), all origins n with n-k in the record
    c2 = {(ip, k): np.zeros(LMAX + 1) for ip in IPS for k in refs}
    cnt = 0
    for ib in range(LMAX, nrec):
        b = WARM + ib; rF = ens.rF(b); cnt += 1
        for kk in range(1, LMAX + 1):
            Ae = ens.A(eta[:, ib - kk], b, rF)
            for k in refs:
                c2[("E", k)][kk] += dot(Ae, Vr[k], "E"); c2[("P", k)][kk] += dot(Ae, PV[k], "E")
    for (ip, k), v in c2.items():
        arr[f"c2_{ip}_{k}"] = v / cnt
    print(f"[A {state}] P2 terms ({time.time() - t0:.0f} s)", flush=True)
    # masked occupancy c_k at mu = 0 (exact operator split), first NCK segment blocks
    ck = {(ip, k): np.zeros(KC + 1) for ip in IPS for k in refs}
    chk = 0.0
    nck = min(NCK, T_)
    for tb in range(nck):
        b = r0 + tb
        for k in refs:
            parts = ens.masked(Vr[k], b)
            full = AW[:, b - WARM] - Aref[k][:, b - WARM]
            chk = max(chk, float(np.max(np.abs(sum(parts) - full)) / np.max(np.abs(full))))
            for kk, p in enumerate(parts):
                pm = p.mean(0)
                ck[("E", k)][kk] += dot(pm, Vr[k], "E") / (H * dot(Vr[k], Vr[k], "E")) / nck
                ck[("P", k)][kk] += dot(pm, Vr[k], "P") / (H * dot(Vr[k], Vr[k], "P")) / nck
    for (ip, k), v in ck.items():
        arr[f"ckmask_{ip}_{k}"] = v
    res["ckmask_sum_check"] = chk
    print(f"[A {state}] masked c_k ({time.time() - t0:.0f} s)", flush=True)
    # ---------------- branches
    vrec0 = {k: np.repeat(Vr[k][None, None], R, 0).repeat(nrec, 1) for k in refs}
    regX, regY = [], []
    res["mu"] = {}
    for im, mu in enumerate(MUS):
        Dfull = D0.copy()
        Wseg = np.repeat(W0[None, None], R, 0).repeat(T_, 1)
        fdE = np.zeros(T_); fdP = np.zeros(T_); fdden = np.zeros((2, T_))
        vHv = {(ip, k): np.zeros(T_) for ip in IPS for k in refs}
        mDre = np.zeros((T_, M))
        if mu > 0:
            for b, w, Dl in ens.branch(W0, mu, r0):
                tb = b - r0; ib = b - WARM
                Dfull[:, ib] = Dl; Wseg[:, tb] = w
                rF = ens.rF(b)
                Aw = ens.A(w, b, rF)
                Dre = DD[:, ib] + Aw
                dm = Dl.mean(0); dre = Dre.mean(0); mDre[tb] = dre
                diff = dm - dre
                fdE[tb] = np.sqrt(dot(diff, diff, "E")); fdP[tb] = np.sqrt(dot(diff, diff, "P"))
                fdden[0, tb] = np.sqrt(dot(dm, dm, "E")); fdden[1, tb] = np.sqrt(dot(dm, dm, "P"))
                Ew = w.mean(0)
                AEw = ens.A(Ew, b, rF).mean(0)
                dv = w - Ew
                Ash = ens.A(dv, b, roll=1)                          # operator of realisation r+1 applied to dv_r
                for k in refs:
                    v = w - refs[k]; Av = Aw - Aref[k][:, ib]
                    Abar_Ev = AEw - Aref[k][:, ib].mean(0)
                    for ip in IPS:
                        vHv[(ip, k)][tb] = dot(v, Av, ip) - dot(Ew - refs[k], Abar_Ev, ip) - dot(dv, Ash, ip)
        else:
            mDre[:] = D0[:, r0 - WARM:].mean(0)
        # Gram matrices over the whole record (pre-roll frozen + segment)
        vfull = {}
        for k in refs:
            vv = vrec0[k].copy(); vv[:, r0 - WARM:] = Wseg - refs[k]; vfull[k] = vv
        for ip in IPS:
            arr[f"GDD_{ip}_m{im}"] = gram(Dfull, Dfull, ip)
            for k in refs:
                arr[f"GvD_{ip}_{k}_m{im}"] = gram(vfull[k], Dfull, ip)
                arr[f"Gvv_{ip}_{k}_m{im}"] = gram(vfull[k], vfull[k], ip)
                arr[f"vHv_{ip}_{k}_m{im}"] = vHv[(ip, k)]
        arr[f"fd_m{im}"] = np.vstack([fdE, fdP, fdden])
        # effective delay (Sec. 9.2): E(Delta_t - Delta^re_t) = h sum_k c_k (E w_{t-k} - E w_t) ~ d_eff h (E w_{t-1} - E w_t).
        # (The c_k themselves are not identifiable from the mean trajectory: E w_{t-k} - E w_t are nearly collinear.)
        if mu > 0:
            Ews = np.concatenate([W0[None], Wseg.mean(0)], 0)            # index tb + 1
            num = den = 0.0
            for tb in range(1, T_):
                y = Dfull[:, r0 - WARM + tb].mean(0) - mDre[tb]
                xk = H * (Ews[tb] - Ews[tb + 1])
                num += float(xk @ y); den += float(xk @ xk)
            arr[f"deff_m{im}"] = np.array([num / den if den > 0 else np.nan])
            regX.append(num); regY.append(den)
        res["mu"][str(mu)] = dict(div=bool(not np.isfinite(Dfull).all()))
        print(f"[A {state}] mu={mu} ({time.time() - t0:.0f} s)", flush=True)
        del Dfull, Wseg, vfull
    if regX:
        arr["deff_all"] = np.array([sum(regX) / sum(regY)])
    arr["g_blocks"] = ens.g[:, WARM:].mean(0)
    res["sec"] = time.time() - t0
    d = outdir(o.smoke)
    np.savez_compressed(os.path.join(d, f"protoA_{state}.npz"), **arr)
    with open(os.path.join(d, f"protoA_{state}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"[A {state}] done ({res['sec']:.0f} s)", flush=True)
    return state


# ----------------------------------------------------------------------------------------------
# Theory (Sec. 2, Sec. 9.1) evaluated from the measured quantities; used by `report`
# ----------------------------------------------------------------------------------------------
def gam_fn(Gk, kg):
    """Gamma_|k| truncated at |k| <= kg (0 beyond)."""
    def g(k):
        k = abs(int(k))
        return Gk[k] if k <= kg else 0.0
    return g


def gf_exo(Gk, kg):
    """Exogenous residual (Sec. 9.1 assumption): E<eta_a, eta_b> = Gamma_|a-b|, independent of the state."""
    g = gam_fn(Gk, kg)
    return lambda a, b: g(a - b)


def gf_scaled(Gz, Gm, Gzm, kg, Xser, X0):
    """Variant 'S ~ X' (Sec. 0: S = f h^2 x): the H~v part of eta scales with the size of v,
    E<eta_a, eta_b> = Gz_k + r Gm_k + sqrt(r) (c_k + c_-k),  r = sqrt(X_a X_b) / X0,  k = |a - b|.
    Xser(t): X at segment time t (t <= 0: X0).  Gzm two-sided (index mid + k = E<zeta_n, m_{n-k}>)."""
    gz, gm = gam_fn(Gz, kg), gam_fn(Gm, kg); mid = len(Gzm) // 2

    def gc(k):
        k = abs(int(k))
        return Gzm[mid + k] + Gzm[mid - k] if k <= min(kg, mid) else 0.0

    def f(a, b):
        r = np.sqrt(Xser(a) * Xser(b)) / X0
        return gz(a - b) + r * gm(a - b) + np.sqrt(r) * gc(a - b)
    return f


def pred_branch(gf, X_tau, tau, l, mu, h=H):
    """First-order branched-segment version of Sec. 9.1 (v = V before the branch point, adaptation from tau = 0):
    C(tau, l) = (1 + h mu min(l, tau)) X_tau + Gamma(tau, s) - h mu (S1 + S2),  s = tau - l,
    S1 = sum_{j=0}^{tau-1} Gamma(j, s)  [from h E<v_tau, eta_s>],  S2 = sum_{j=0}^{s-1} Gamma(tau, j)  [h E<eta_tau, v_s>].
    gf(a, b) = E<eta_a, eta_b> (gf_exo: Gamma_|a-b|).  Returns the four parts."""
    s = tau - l
    S1 = sum(gf(j, s) for j in range(tau))
    S2 = sum(gf(tau, j) for j in range(max(s, 0)))
    return dict(vv=(1 + h * mu * min(l, tau)) * X_tau, veta=-h * mu * S1, etav=-h * mu * S2, etaeta=gf(tau, s))


def pred_steady(Gk, kg, X, l, mu, h=H):
    """Sec. 9.1 as written: C_l = (1 + h mu l) X + Gamma_l - h mu (K_eta - Gamma_l) (Q = C_0)."""
    g = gam_fn(Gk, kg); K = g(0) + 2 * sum(g(k) for k in range(1, kg + 1))
    return (1 + h * mu * l) * X + g(l) - h * mu * (K - g(l))


def pred_old(Gk, kg, X, l, mu, h=H):
    """Sec. 2 (2nd edition): C = X - h mu K_eta (lag beyond the correlation), Q = X + Gamma_0."""
    g = gam_fn(Gk, kg); K = g(0) + 2 * sum(g(k) for k in range(1, kg + 1))
    return X + g(0) if l == 0 else X - h * mu * K


def ema_weights(m, bd):
    """ubar_m = sum_j p_j u_{m-j}, EMA initialised with the first record (index 0)."""
    if m < 0:
        return np.zeros(0)
    p = bd * (1 - bd) ** np.arange(m + 1)
    p[m] = (1 - bd) ** m
    return p


def ema_c(G, t, L, bd):
    """E c_t = E<u_t, ubar_{t-L}> from the Gram matrix G (record indices); 0 if t < L (DirectEst)."""
    if t < L:
        return 0.0
    p = ema_weights(t - L, bd)
    return float(sum(p[j] * G[t, t - L - j] for j in range(len(p))))


def ema_out(vals, t, L, beta):
    """Outer EMA as DirectEst: starts at record index L + 4 with the first value."""
    t0 = L + 4
    if t < t0:
        return float("nan")
    acc = vals[t0]
    for i in range(t0 + 1, t + 1):
        acc = (1 - beta) * acc + beta * vals[i]
    return float(acc)


def solve91(C, Q, mu, L, bd, f, tm, ta, gbar_z, gbar_m, h=H):
    """Sec. 9.1 EMA forms solved for (X, N):
      C = [1 + h mu (lbar - tm f)] X - h mu ta N + (1 + h mu) Gbar,   Gbar = N gbar_z + f X gbar_m  (*)
      Q = (1 + f) X + N - h mu [(tm - 1) f X + (ta - 1) N]
    (*) Gbar is not given in terms of (X, N) in THEORY 9.1; here it is written with the normalised mu = 0 profiles
    gbar_z = sum_j p_j rho^zeta_{L+j}, gbar_m = sum_j p_j rho^m_{L+j} (measured)."""
    lbar = L + (1 - bd) / bd
    a1 = 1 + h * mu * (lbar - tm * f) + (1 + h * mu) * f * gbar_m
    a2 = -h * mu * ta + (1 + h * mu) * gbar_z
    b1 = (1 + f) - h * mu * (tm - 1) * f
    b2 = 1 - h * mu * (ta - 1)
    det = a1 * b2 - a2 * b1
    X = (C * b2 - a2 * Q) / det; Nn = (a1 * Q - b1 * C) / det
    return max(X, 0.0), max(Nn, 0.0)


# ----------------------------------------------------------------------------------------------
# Self-check
# ----------------------------------------------------------------------------------------------
class _St:
    def __init__(s, Srr, Pbar, eps, g, nb):
        s.Srr = np.repeat(Srr[None], nb, 0); s.Pbar = np.full(nb, Pbar); s.eps = np.full(nb, eps); s.g = g; s.pi = {}


def check(o):
    ok = True

    def rep(name, cond, msg=""):
        nonlocal ok
        ok &= bool(cond); print(f"[{'OK ' if cond else 'FAIL'}] {name} {msg}", flush=True)

    S = setup(states=True)
    dd = S["_dd"]
    # 1. paths reproduce the data builder
    rep("d, r == data builder (G_s0)", np.array_equal(signal.lfilter(S["hP"], [1.0], dd["x"][:5000]), dd["d"][:5000])
        and np.array_equal(signal.lfilter(S["hS"], [1.0], dd["x"][:5000]), dd["r"][:5000]))
    # 2. FixedSF == SFm.sf_dir
    rng = np.random.default_rng(3)
    Rr = rng.standard_normal(NBIN) + 1j * rng.standard_normal(NBIN); Ee = rng.standard_normal(NBIN) + 1j * rng.standard_normal(NBIN)
    a = FixedSF(S["Srr"], S["Pbar"], S["eps"])(Rr, Ee); b = SFm.sf_dir(Rr, Ee, S["Srr"], S["Pbar"], S["eps"], EPS_REG)
    rep("FixedSF == SFm.sf_dir", np.max(np.abs(a - b)) < 1e-12 * np.max(np.abs(b)), f"{np.max(np.abs(a - b)):.1e}")
    # 3. branch loop == vssT_lib.simulate with the frozen normaliser (mu = 0 before r0, then mu)
    W0 = S["W"]["ii"]; nblk = 40; r0 = 24; mu = 0.1
    ens = Ens(S, 2, nblk, 7)
    rec = {b: Dl for b, w, Dl in ens.branch(W0, mu, r0)}
    x0 = ens.x[0]; xq = np.concatenate([np.zeros(N), x0])
    data = dict(d=ens.d[0], xq=xq, hS=S["hS"], r=ens.r[0], pc=False)
    Rall = np.array([S["Sf"] * np.fft.rfft(xq[i * B + B:i * B + B + N]) for i in range(nblk)])
    st = _St(S["Srr"], S["Pbar"], S["eps"], np.ones(nblk), nblk)
    dv = []
    simulate(data, Rall, st, Fixed("SF", np.where(np.arange(nblk) < r0, 0.0, mu)), nb=nblk, w0=W0,
             hook=lambda i, w, m, d_, *a_, **k: dv.append(KD * d_))
    dmax = max(np.max(np.abs(rec[b][0] - dv[b])) / np.max(np.abs(dv[b])) for b in range(r0, nblk))
    rep("branch loop == vssT_lib.simulate (frozen normaliser)", dmax < 1e-9, f"rel diff {dmax:.1e}")
    # 4. mu = 0: frozen operator == frozen probes (vssT_lib.frozen_dirs) and == the branch loop with mu = 0
    fd = frozen_dirs(data, Rall, st, W0, range(WARM, nblk), "SF")
    d1 = max(np.max(np.abs(ens.Dd(b)[0] + ens.A(W0, b)[0] - KD * fd[b - WARM])) / np.max(np.abs(KD * fd[b - WARM]))
             for b in range(WARM, nblk))
    rep("mu=0: dir(d) + A W0 == frozen_dirs", d1 < 1e-9, f"rel diff {d1:.1e}")
    rec0 = {b: Dl for b, w, Dl in ens.branch(W0, 0.0, WARM)}
    d2 = max(np.max(np.abs(rec0[b] - (ens.Dd(b) + ens.A(W0, b)))) / np.max(np.abs(rec0[b])) for b in range(WARM, nblk))
    rep("re-generated frame == actual frame when w is constant", d2 < 1e-9, f"rel diff {d2:.1e}")
    # 4b. frame-delay identity: Delta_b - Delta^re_b = sum_k A^(k)_b (w_{b-k} - w_b)  (exact, realisation 0)
    e1 = Ens(S, 1, nblk, None, x=ens.x[:1].copy())
    ws = {b: w[0] for b, w, Dl in e1.branch(W0, mu, r0)}
    d4 = 0.0
    for b in range(r0 + 1, nblk, 3):
        Dre = e1.Dd(b)[0] + e1.A(ws[b], b)[0]
        corr = sum(e1.masked(ws.get(b - k, W0), b)[k][0] - e1.masked(ws[b], b)[k][0] for k in range(KC + 1))
        d4 = max(d4, float(np.max(np.abs(rec[b][0] - Dre - corr)) / np.max(np.abs(rec[b][0] - Dre))))
    rep("frame delay: Delta - Delta^re == sum_k A^(k) (w_b-k - w_b)", d4 < 1e-8, f"rel diff {d4:.1e}")
    # 5. masked split sums to the full operator
    parts = ens.masked(W0 - S["wfp"], r0 + 3); full = ens.A(W0 - S["wfp"], r0 + 3)
    d3 = np.max(np.abs(sum(parts) - full)) / np.max(np.abs(full))
    rep("masked occupancy split sums to A_b v", d3 < 1e-9, f"rel diff {d3:.1e}")
    # 6. offline EMA from the Gram matrix == DirectEst online (one realisation, g = 1), and C_0 == Q
    Ds = np.array([rec[b][0] if b >= r0 else (ens.Dd(b) + ens.A(W0, b))[0] for b in range(WARM, nblk)])
    G = Ds @ Ds.T
    for L, bd in ((4, 0.1), (8, 0.3)):
        de = T.DirectEst("E", L=L, bd=bd, b=0.05, gmin=0.25)
        for i, Dl in enumerate(Ds):
            de.update(i, Dl / KD, 1.0)
        n = len(Ds) - 1
        c_off = ema_out([ema_c(G, t, L, bd) for t in range(n + 1)], n, L, 0.05)
        q_off = ema_out([G[t, t] for t in range(n + 1)], n, L, 0.05)
        rep(f"offline EMA == DirectEst (L={L}, bd={bd})", abs(c_off - de.C) < 1e-9 * abs(de.Q) and abs(q_off - de.Q) < 1e-9 * de.Q,
            f"C {c_off:.6g} / {de.C:.6g}, Q {q_off:.6g} / {de.Q:.6g}")
    rep("C_0 == Q (lag 0 of the Gram matrix)", abs(ema_c(G, 10, 0, 1.0) - G[10, 10]) < 1e-12 * G[10, 10])
    # 7. branched prediction reduces to Sec. 9.1 for tau -> infinity
    Gk = np.array([1.0, 0.6, 0.3, 0.1] + [0.0] * 30)
    for l in (0, 1, 3, 6):
        pb = pred_branch(gf_exo(Gk, 16), 2.0, 200, l, 0.05); s1 = sum(pb.values())
        s2 = pred_steady(Gk, 16, 2.0, l, 0.05)
        ok_l = abs(s1 - s2) < 1e-12
        rep(f"pred_branch(tau large) == Sec. 9.1 (l={l})", ok_l, f"{s1:.6f} vs {s2:.6f}")
    # 8. solve91 inverts its own forward model
    f, tm, ta, gz, gm = 0.9, 2.5, 3.8, 0.05, 0.1
    X, Nn, mu_, L, bd = 0.7, 1.3, 0.05, 4, 0.1
    lbar = L + (1 - bd) / bd; Gb = Nn * gz + f * X * gm
    C = (1 + H * mu_ * (lbar - tm * f)) * X - H * mu_ * ta * Nn + (1 + H * mu_) * Gb
    Q = (1 + f) * X + Nn - H * mu_ * ((tm - 1) * f * X + (ta - 1) * Nn)
    Xh, Nh = solve91(C, Q, mu_, L, bd, f, tm, ta, gz, gm)
    rep("solve91 inverts the Sec. 9.1 EMA forms", abs(Xh - X) < 1e-12 and abs(Nh - Nn) < 1e-12)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED", flush=True)
    return ok


# ----------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------
def fm(x, p=3):
    try:
        return "NaN" if x is None or not np.isfinite(x) else (f"{x:.{p}g}" if abs(x) < 1e-3 and x != 0 else f"{x:.{p}f}")
    except TypeError:
        return str(x)


class Loaded:
    def __init__(s, js, A):
        s.js, s.A = js, A
        s.T, s.pre, s.r0, s.warm = js["T"], js["pre"], js["r0"], js["warm"]
        s.ib0 = s.r0 - s.warm                     # record index of the branch point
        s.mus = js["mus"]

    def G(s, key, im):
        return s.A[f"{key}_m{im}"]

    def gk(s, ip):
        return s.A[f"Geta_{ip}"]


def windows(T_):
    early = list(range(1, min(8, T_)))
    late = list(range(min(16, T_ // 2), T_))
    return {"early (tau 1-7)": early, f"late (tau {late[0]}-{late[-1]})": late}


def mk_gf(A, ip, kg, im=None, X0=None, t_of=None):
    """Gamma functions for state-time pairs: exogenous (Sec. 9.1) and 'S ~ X' (H~v part scaled with X_t)."""
    exo = gf_exo(A[f"Geta_{ip}"], kg)
    if im is None:
        return exo, None
    Gvv = A[f"Gvv_{ip}_fp_m{im}"]

    def Xser(t):
        return X0 if t <= 0 else H ** 2 * Gvv[t_of(t), t_of(t)]
    sc = gf_scaled(A[f"Gz_{ip}_fp"], A[f"Gm_{ip}_fp"], A[f"Gzm_{ip}_fp"], kg, Xser, X0)
    return exo, sc


def rep_state(Ld, kg, L_):
    js, A = Ld.js, Ld.A
    st = js["state"]; T_ = Ld.T; ib0 = Ld.ib0
    W = windows(T_)
    tof = lambda tau: ib0 + tau
    out = [f"## State {st}: {js['desc']}", "",
           f"R = {js['R']}, T = {T_}, pre-roll {js['pre']} frozen blocks, seed {js['seed']}. Excess of W0 rel. wM "
           f"{fm(js['W0_exP_wM_dB'], 2)} dB; |v|^2_Phi/Jo rel. fp {fm(js['V']['fp']['exP_dB'], 2)} dB. "
           f"Mean actual g = {fm(js['g_mean'])} (E g^2 = {fm(js['g2_mean'])}).", ""]
    # ---- Table 0: frozen decomposition
    out += [f"### {st}.0 Frozen decomposition at mu = 0 (Gamma truncated at |k| <= {kg})", "",
            "| ip | ref | X = h^2|V|^2 | |E Delta|^2 / X | h_eff/h | nonscalar | |E zeta|^2 | N | S | f = S/X | tau_a | tau_m | cross sum / sqrt(NS) | Gamma_0 | K_eta | K_eta/Gamma_0 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    consts = {}
    for ip in IPS:
        Ge = A[f"Geta_{ip}"]; K = Ge[0] + 2 * Ge[1:kg + 1].sum()
        for k in REFS_:
            fr = js["frozen"][ip][k]
            Gz, Gm, Gzm = A[f"Gz_{ip}_{k}"], A[f"Gm_{ip}_{k}"], A[f"Gzm_{ip}_{k}"]
            Nn, S_ = Gz[0], Gm[0]
            ta = (Gz[0] + 2 * Gz[1:kg + 1].sum()) / Nn; tm = (Gm[0] + 2 * Gm[1:kg + 1].sum()) / S_
            mid = len(Gzm) // 2
            cr = Gzm[mid - kg:mid + kg + 1].sum() / np.sqrt(Nn * S_)
            f = S_ / fr["X"]
            consts[(ip, k)] = dict(N=Nn, S=S_, f=f, ta=ta, tm=tm, X=fr["X"], Gz=Gz, Gm=Gm)
            out.append("| " + " | ".join([ip, k, fm(fr["X"]), fm(js["frozen"][ip]["EDelta2"] / fr["X"]), fm(fr["h_eff"] / H), fm(fr["nonscalar"]),
                                           fm(fr["Ezeta2"]), fm(Nn), fm(S_), fm(f), fm(ta, 2), fm(tm, 2), fm(cr), fm(Ge[0]), fm(K),
                                           fm(K / Ge[0], 2)]) + " |")
    out += ["", "X, N, S in the inner product of the row (E: Euclidean, P: Phi). N, S, tau from the reference's zeta and "
            "H~V (mean over realisations and blocks removed); Gamma, K_eta of eta = Delta - E Delta do not depend on the "
            "reference. nonscalar = |A_bar V - h_eff V|^2 / (h_eff^2 |V|^2). |E Delta|^2 / X != 1 is the scalar-mean "
            "approximation error (E Delta = h v assumed); it shifts every C_l by a constant at mu = 0.", ""]
    # ---- Table 1: C_l and Q vs mu and l
    lsel = [0, 1, 2, 4, 8, 16]
    for ip in IPS:
        Gk = A[f"Geta_{ip}"]; X0 = consts[(ip, "fp")]["X"]
        out += [f"### {st}.1 ({ip}) Raw lagged products C_l = E<Delta_t, Delta_t-l> (C_0 = Q): measured vs predictions, ref fp", "",
                "Columns per lag: meas | 9.1 = Sec. 9.1 steady form | br = same first-order theory with the finite sums of the "
                "branched segment (exogenous Gamma) | brS = br with the H~v part of Gamma scaled with X_t (S = f h^2 x) | "
                "brSA = brS with the measured mean operator (X -> kappa X, kappa = (h_eff/h)^2 (1 + nonscalar); h -> h_eff in the "
                f"h mu factors) | old = Sec. 2. All values / Gamma_0 (Gamma_0 = {fm(Gk[0])}). Window means over tau.", ""]
        frA = js["frozen"][ip]["fp"]; hA = frA["h_eff"]; kapA = (hA / H) ** 2 * (1 + frA["nonscalar"])
        hdr = "| window | mu | " + " | ".join(f"l={l} meas | 9.1 | br | brS | brSA | old" for l in lsel) + " |"
        out += [hdr, "|---|---|" + "---|" * (6 * len(lsel))]
        for wn, taus in W.items():
            for im, mu in enumerate(Ld.mus):
                GDD = A[f"GDD_{ip}_m{im}"]; Gvv = A[f"Gvv_{ip}_fp_m{im}"]
                gex, gsc = mk_gf(A, ip, kg, im, X0, tof)
                cells = []
                for l in lsel:
                    me, p91, pbr, pbs, pba, pol = [], [], [], [], [], []
                    for tau in taus:
                        t = ib0 + tau; X = H ** 2 * Gvv[t, t]
                        me.append(GDD[t, t - l]); p91.append(pred_steady(Gk, kg, X, l, mu))
                        pbr.append(sum(pred_branch(gex, X, tau, l, mu).values()))
                        pbs.append(sum(pred_branch(gsc, X, tau, l, mu).values())); pol.append(pred_old(Gk, kg, X, l, mu))
                        pba.append(sum(pred_branch(gsc, kapA * X, tau, l, mu, h=hA).values()))
                    cells += [fm(np.mean(q) / Gk[0]) for q in (me, p91, pbr, pbs, pba, pol)]
                out.append(f"| {wn} | {mu:g} | " + " | ".join(cells) + " |")
        out.append("")
        # four-way split (measured vs branch predictions)
        out += [f"#### {st}.1b ({ip}) Split C_l = h^2<v_t,v_s> + h<v_t,eta_s> + h<eta_t,v_s> + <eta_t,eta_s> "
                "(eta := Delta - h v, ref fp; s = t - l): measured / br / brS, / Gamma_0. At mu = 0 the v-eta and eta-v "
                "entries are the scalar-mean error h<V, (A_bar - h) V>, and eta-eta contains |(A_bar - h)V|^2.", ""]
        for wn, taus in W.items():
            out += [f"{wn}:", "", "| mu | l | vv meas | vv pred | v-eta meas | br | brS | eta-v meas | br | brS | eta-eta meas | br | brS |",
                    "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for im, mu in enumerate(Ld.mus):
                GDD = A[f"GDD_{ip}_m{im}"]; Gvv = A[f"Gvv_{ip}_fp_m{im}"]; GvD = A[f"GvD_{ip}_fp_m{im}"]
                gex, gsc = mk_gf(A, ip, kg, im, X0, tof)
                for l in (0, 4, 16):
                    m = {q: [] for q in ("vv", "veta", "etav", "etaeta")}; p = {q: [] for q in m}; ps = {q: [] for q in m}
                    for tau in taus:
                        t = ib0 + tau; s_ = t - l; X = H ** 2 * Gvv[t, t]
                        vv = H ** 2 * Gvv[t, s_]; ve = H * GvD[t, s_] - vv; ev = H * GvD[s_, t] - vv
                        m["vv"].append(vv); m["veta"].append(ve); m["etav"].append(ev); m["etaeta"].append(GDD[t, s_] - vv - ve - ev)
                        pb = pred_branch(gex, X, tau, l, mu); pq = pred_branch(gsc, X, tau, l, mu)
                        for q in m:
                            p[q].append(pb[q]); ps[q].append(pq[q])
                    g0 = Gk[0]
                    row = [fm(np.mean(m["vv"]) / g0), fm(np.mean(p["vv"]) / g0)]
                    for q in ("veta", "etav", "etaeta"):
                        row += [fm(np.mean(m[q]) / g0), fm(np.mean(p[q]) / g0), fm(np.mean(ps[q]) / g0)]
                    out.append(f"| {mu:g} | {l} | " + " | ".join(row) + " |")
            out.append("")
        # EMA versions
        out += [f"#### {st}.1c ({ip}) Implementation EMA versions at the last block (tau = {T_ - 1}): E c = E<u_t, ubar_t-L> "
                "(inner) and, for (L, beta_d) = (4, 0.1), the outer EMA (beta 0.05) of c and q. / Gamma_0. "
                "br / brS: EMA weights applied to the branch predictions of C_l; 9.1 EMA form: "
                "(1 + h mu lbar) X - h mu K_eta + (1 + h mu) Gbar.", "",
                "| mu | (L, beta_d) | E c meas | br | brS | Sec. 9.1 EMA form | Sec. 2 | outer C meas | outer Q meas | Q pred 9.1 | Q pred br | Q pred brS |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        t_last = ib0 + T_ - 1
        for im, mu in enumerate(Ld.mus):
            GDD = A[f"GDD_{ip}_m{im}"]; Gvv = A[f"Gvv_{ip}_fp_m{im}"]
            X = H ** 2 * Gvv[t_last, t_last]
            gex, gsc = mk_gf(A, ip, kg, im, X0, tof)
            for (L, bd) in L_:
                cm = ema_c(GDD, t_last, L, bd)
                p = ema_weights(t_last - L, bd)
                pbr = sum(p[j] * sum(pred_branch(gex, X, T_ - 1, L + j, mu).values()) for j in range(len(p)))
                pbs = sum(p[j] * sum(pred_branch(gsc, X, T_ - 1, L + j, mu).values()) for j in range(len(p)))
                lbar = L + (1 - bd) / bd
                pinf = bd * (1 - bd) ** np.arange(200)
                Gbar = sum(pinf[j] * (Gk[L + j] if L + j <= kg else 0.0) for j in range(200))
                K = Gk[0] + 2 * Gk[1:kg + 1].sum()
                p91 = (1 + H * mu * lbar) * X - H * mu * K + (1 + H * mu) * Gbar
                pol = pred_old(Gk, kg, X, 4, mu)
                if (L, bd) == (4, 0.1):
                    cs = [ema_c(GDD, t, L, bd) for t in range(t_last + 1)]
                    co = ema_out(cs, t_last, L, 0.05); qo = ema_out([GDD[t, t] for t in range(t_last + 1)], t_last, L, 0.05)
                    q91 = pred_steady(Gk, kg, X, 0, mu)
                    qbr = sum(pred_branch(gex, X, T_ - 1, 0, mu).values()); qbs = sum(pred_branch(gsc, X, T_ - 1, 0, mu).values())
                else:
                    co = qo = q91 = qbr = qbs = float("nan")
                out.append(f"| {mu:g} | ({L}, {bd:g}) | " + " | ".join(fm(x / Gk[0]) for x in (cm, pbr, pbs, p91, pol, co, qo, q91, qbr, qbs)) + " |")
        out.append("")
    # ---- Table 2: slopes at mu = 0
    im1 = 1; mu1 = Ld.mus[im1]
    out += [f"### {st}.2 Slopes d/dmu at mu = 0 (finite difference mu = {mu1:g} vs 0, common random numbers), ref fp, / Gamma_0", "",
            "slope (C - X): measured slope of C_l - X_tau (removes the measured shrinkage of X and the mu = 0 offsets). "
            "Predicted slope of C - X: br = h min(l,tau) X - h (S1 + S2) (exogenous Gamma); brS = br + slope of the "
            "X-scaled H~v part of Gamma_l (uses the measured slope of X); 9.1 = h l X - h (K_eta - Gamma_l); "
            "Sec. 2 = -h K_eta (l > 0), 0 (l = 0). Last four columns: measured slopes of the parts (vv - X, v-eta, eta-v, eta-eta).", "",
            "| ip | window | l | slope C | slope X | slope (C - X) | br | brS | 9.1 | Sec. 2 | vv - X | v-eta | eta-v | eta-eta |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for ip in IPS:
        Gk = A[f"Geta_{ip}"]; K = Gk[0] + 2 * Gk[1:kg + 1].sum(); X0c = consts[(ip, "fp")]["X"]
        Gm = A[f"Gm_{ip}_fp"]; Gzm = A[f"Gzm_{ip}_fp"]; mid = len(Gzm) // 2
        G0 = {q: A[f"{q}_{ip}{'_fp' if q != 'GDD' else ''}_m0"] for q in ("GDD", "Gvv", "GvD")}
        G1 = {q: A[f"{q}_{ip}{'_fp' if q != 'GDD' else ''}_m{im1}"] for q in ("GDD", "Gvv", "GvD")}
        gex, _ = mk_gf(A, ip, kg)
        for wn, taus in W.items():
            for l in (0, 1, 4, 8, 16):
                acc = {q: [] for q in ("C", "X", "CX", "pbr", "pbs", "p91", "pol", "vv", "ve", "ev", "ee")}
                for tau in taus:
                    t = ib0 + tau; s_ = t - l
                    parts = []
                    for G in (G0, G1):
                        X = H ** 2 * G["Gvv"][t, t]; vv = H ** 2 * G["Gvv"][t, s_]
                        ve = H * G["GvD"][t, s_] - vv; ev = H * G["GvD"][s_, t] - vv; C = G["GDD"][t, s_]
                        parts.append(dict(C=C, X=X, CX=C - X, vv=vv - X, ve=ve, ev=ev, ee=C - vv - ve - ev,
                                          Xs=H ** 2 * G["Gvv"][s_, s_] if s_ >= ib0 else X0c))
                    for q in ("C", "X", "CX", "vv", "ve", "ev", "ee"):
                        acc[q].append((parts[1][q] - parts[0][q]) / mu1)
                    X0 = parts[0]["X"]; pb = pred_branch(gex, X0, tau, l, 1.0)
                    pbr = H * min(l, tau) * X0 + pb["veta"] + pb["etav"]
                    acc["pbr"].append(pbr)
                    # d/dmu of [r Gm_l + sqrt(r) gc_l], r = sqrt(X_t X_s)/X0
                    dr = 0.5 * ((parts[1]["X"] - parts[0]["X"]) + (parts[1]["Xs"] - parts[0]["Xs"])) / mu1 / X0c
                    gm_l = Gm[l] if l <= kg else 0.0
                    gc_l = (Gzm[mid + l] + Gzm[mid - l]) if l <= min(kg, mid) else 0.0
                    acc["pbs"].append(pbr + dr * gm_l + 0.5 * dr * gc_l)
                    acc["p91"].append(H * l * X0 - H * (K - (Gk[l] if l <= kg else 0.0)))
                    acc["pol"].append(0.0 if l == 0 else -H * K)
                out.append(f"| {ip} | {wn} | {l} | " + " | ".join(fm(np.mean(acc[q]) / Gk[0]) for q in
                                                                  ("C", "X", "CX", "pbr", "pbs", "p91", "pol", "vv", "ve", "ev", "ee")) + " |")
    out.append("")
    # ---- Table 3: one-step oracle vs estimators
    out += [f"### {st}.3 One-step oracle [E<v,Delta>/E|Delta|^2]_+ vs mu (Sec. 9.3: = mu/2 in steady state) and the estimators", "",
            "Oracle columns: ref fp (E, Phi), ref wM (Phi). Estimators from the implementation statistics (Euclidean, L = 4, "
            "beta_d = 0.1; 'in' = expected inner c and q at tau, 'out' = with the outer EMA beta 0.05), constants f, tau_m, "
            "tau_a and the zeta / H~V correlation profiles from this state's mu = 0 decomposition (Euclidean, ref fp), a0 = 1. "
            "(a) Sec. 2 psi_muhat; (b) Sec. 9.1 EMA forms solved for (X, N): mu_deg = (X/h)/((1 + tau_m f) X + tau_a N) "
            "(Sec. 1/2 form), mu_1 = [X/h - mu((tau_a - 1) N + (tau_m - 1) f X)/2]_+ / Q (one-step form). Last column: "
            "mu_deg with the true X_tau and the mu = 0 N. (b) mu_1A: mu_1 with the measured mean operator (h -> h_eff in the "
            "solve; E<v,Delta> = X / (h_eff (1 + nonscalar))).", ""]
    out += ["| window | mu | mu/2 | oracle fp E | oracle fp P | oracle wM P | (a) in | (a) out | (b) mu_deg in | (b) mu_deg out | (b) mu_1 in | (b) mu_1A in | mu_deg true X,N |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    frE = js["frozen"]["E"]["fp"]; hE, nsE = frE["h_eff"], frE["nonscalar"]
    cE = consts[("E", "fp")]
    L, bd = 4, 0.1
    rhoz = cE["Gz"] / cE["Gz"][0]; rhom = cE["Gm"] / cE["Gm"][0]
    pinf = bd * (1 - bd) ** np.arange(200)
    gz = sum(pinf[j] * (rhoz[L + j] if L + j <= kg else 0.0) for j in range(200))
    gm_ = sum(pinf[j] * (rhom[L + j] if L + j <= kg else 0.0) for j in range(200))
    for wn, taus in W.items():
        for im, mu in enumerate(Ld.mus):
            GDDE = A[f"GDD_E_m{im}"]; GDDP = A[f"GDD_P_m{im}"]
            cs = [ema_c(GDDE, t, L, bd) for t in range(GDDE.shape[0])]
            qs = [GDDE[t, t] for t in range(GDDE.shape[0])]
            acc = {q: [] for q in ("oE", "oP", "oWP", "ai", "ao", "bi", "bo", "b1", "b1A", "bt")}
            for tau in taus:
                t = ib0 + tau
                oE = A[f"GvD_E_fp_m{im}"][t, t] / GDDE[t, t]; oP = A[f"GvD_P_fp_m{im}"][t, t] / GDDP[t, t]
                oW = A[f"GvD_P_wM_m{im}"][t, t] / GDDP[t, t]
                acc["oE"].append(max(oE, 0)); acc["oP"].append(max(oP, 0)); acc["oWP"].append(max(oW, 0))
                for nm, (C, Q) in (("i", (cs[t], qs[t])), ("o", (ema_out(cs, t, L, 0.05), ema_out(qs, t, L, 0.05)))):
                    acc["a" + nm].append(psi_muhat(C, Q, mu, 1.0, cE["f"], cE["tm"], cE["ta"])[1])
                    X, Nn = solve91(C, Q, mu, L, bd, cE["f"], cE["tm"], cE["ta"], gz, gm_)
                    den = (1 + cE["tm"] * cE["f"]) * X + cE["ta"] * Nn
                    acc["b" + nm].append((X / H) / den if den > 0 else 0.0)
                    if nm == "i":
                        acc["b1"].append(max(X / H - mu * ((cE["ta"] - 1) * Nn + (cE["tm"] - 1) * cE["f"] * X) / 2, 0) / Q)
                        XA, NA = solve91(C, Q, mu, L, bd, cE["f"], cE["tm"], cE["ta"], gz, gm_, h=hE)
                        acc["b1A"].append(max(XA / (hE * (1 + nsE)) - mu * ((cE["ta"] - 1) * NA + (cE["tm"] - 1) * cE["f"] * XA) / 2, 0) / Q)
                Xt = H ** 2 * A[f"Gvv_E_fp_m{im}"][t, t]
                acc["bt"].append((Xt / H) / ((1 + cE["tm"] * cE["f"]) * Xt + cE["ta"] * cE["N"]))
            out.append(f"| {wn} | {mu:g} | {mu / 2:g} | " + " | ".join(fm(np.nanmean(acc[q]), 4) for q in
                                                                     ("oE", "oP", "oWP", "ai", "ao", "bi", "bo", "b1", "b1A", "bt")) + " |")
    out.append("")
    # ---- Table 4: frame delay
    ckE = A["ckmask_E_fp"]; ckP = A["ckmask_P_fp"]
    kbarE = float(np.sum(np.arange(KC + 1) * ckE)); kbarP = float(np.sum(np.arange(KC + 1) * ckP))
    out += [f"### {st}.4 Frame delay (Sec. 9.2)", "",
            "ratio = |E Delta - E Delta^re| / |E Delta| (window means). Mean-operator prediction: E(Delta - Delta^re) = "
            "h sum_k c_k (E v_t-k - E v_t) ~ h mu (sum_k k c_k) E Delta, i.e. ratio ~ h mu sum_k k c_k with the masked c_k "
            f"(units of h): sum k c_k = {fm(kbarE)} (E), {fm(kbarP)} (Phi). d_eff: least-squares scalar in "
            "E(Delta_t - Delta^re_t) = d_eff h (E w_t-1 - E w_t) (Euclidean); compare with sum k c_k. A ratio or d_eff above "
            "the mean-operator value means the fluctuation of the delayed frame is correlated with the past updates "
            "(the same mechanism as T2 in P2).", "",
            "| window | mu | ratio E | pred E | ratio Phi | pred Phi | d_eff (all tau) | d_eff (h_eff) |", "|---|---|---|---|---|---|---|---|"]
    hE4 = js["frozen"]["E"]["fp"]["h_eff"]
    for wn, taus in W.items():
        for im, mu in enumerate(Ld.mus):
            if mu == 0:
                continue
            fd = A[f"fd_m{im}"]
            de = A[f"deff_m{im}"][0] if f"deff_m{im}" in A.files else np.nan
            out.append(f"| {wn} | {mu:g} | {fm(np.mean(fd[0, taus] / fd[2, taus]))} | {fm(H * mu * kbarE)} | "
                       f"{fm(np.mean(fd[1, taus] / fd[3, taus]))} | {fm(H * mu * kbarP)} | {fm(de)} | {fm(de * H / hE4)} |")
    out += ["", "Occupancy c_k at mu = 0 (exact split of A_b v by the block whose coefficients produced the output samples; "
            "units of h; THEORY 9.2: about 1/4 each, k = 0..3; sum = h_eff/h).", "",
            "| c_k | " + " | ".join(f"k={k}" for k in range(KC + 1)) + " | sum | sum k c_k |", "|---|" + "---|" * (KC + 3)]
    for ip in IPS:
        for k in REFS_:
            c = A[f"ckmask_{ip}_{k}"]
            out.append(f"| {ip}, ref {k} | " + " | ".join(fm(x) for x in c) + f" | {fm(c.sum())} | {fm(float(np.sum(np.arange(KC + 1) * c)))} |")
    if "deff_all" in A.files:
        out += ["", f"d_eff pooled over mu: {fm(A['deff_all'][0])}"]
    out += ["", f"masked split check (sum of parts == A_b v): max rel diff {js['ckmask_sum_check']:.1e}", ""]
    # ---- Table 5: P2
    out += [f"### {st}.5 P2 (Sec. 9.4): T1(m) = sum_k<=m E<d_n-k, H~_n V>, T2(m) = sum_k<=m E<V, H~_n d_n-k>, mu = 0", "",
            "Implied tau_m = 1 + 2 (T1 + T2)/S (Sec. 1: E<v, H~v> = -mu S (tau_m - 1)/2). T2 = T1: 2 tau - 1 type; T2 = 0: "
            "tau_m type. T1zm: zeta-H~V cross part of T1 (ignored by the theory). -FD slope: -d/dmu E<v_n, H~_n v_n> at "
            "n - r = m from the mu = 0.0125 / 0.025 branches (single branch point, R realisations); the theory says it "
            "equals T1 + T2. All / S.", "",
            "| ip | ref | m | T1/S | T1zm/S | T2/S | T2/T1 | (T1+T2)/S | implied tau_m | frozen tau_m | 2 tau_m - 1 | -FD slope/S (mu .0125) | -FD slope/S (mu .025) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for ip in IPS:
        for k in REFS_:
            c = consts[(ip, k)]
            c1 = A[f"c1_{ip}_{k}"]; c1z = A[f"c1z_{ip}_{k}"]; c2 = A[f"c2_{ip}_{k}"]
            for m in (1, 2, 4, 8, 12, 16):
                if m > LMAX:
                    continue
                T1 = c1[1:m + 1].sum(); T1z = c1z[1:m + 1].sum(); T2 = c2[1:m + 1].sum()
                fd = []
                for im in (1, 2):
                    vh = A[f"vHv_{ip}_{k}_m{im}"]
                    fd.append(-vh[m] / Ld.mus[im] / c["S"] if m < len(vh) else float("nan"))
                out.append("| " + " | ".join([ip, k, str(m), fm(T1 / c["S"]), fm(T1z / c["S"]), fm(T2 / c["S"]), fm(T2 / T1 if T1 else np.nan),
                                               fm((T1 + T2) / c["S"]), fm(1 + 2 * (T1 + T2) / c["S"], 2), fm(c["tm"], 2),
                                               fm(2 * c["tm"] - 1, 2), fm(fd[0]), fm(fd[1])]) + " |")
    out.append("")
    return out


def report(o):
    d = OUT_SMOKE if o.smoke else OUT_FULL
    kg = LMAX
    L_ = ((4, 0.1), (4, 0.3), (4, 1.0), (8, 0.1), (8, 0.3), (8, 1.0))
    out = ["# RESULTS protoA: branched segments, Gaussian, SF, fixed normaliser (THEORY_vss2.md Sec. 9.1-9.4)", "",
           f"Generated by `python protoA.py report{' --smoke' if o.smoke else ''}` from {os.path.relpath(d, HERE)}/protoA_*.npz. "
           "Prototype diagnostics, not confirmatory. Delta = KD * SF direction (E Delta = h v, h = 0.25), g treated as 1 "
           "(u = Delta). tau = blocks after the branch point (tau = 0: first adapted block, w = W0). "
           "All predictions use only measured X_tau = h^2 E|v_tau|^2 and Gamma_k of the mu = 0 run of the same state "
           "(no fitted constants).", ""]
    found = False
    for st in STATES:
        p = os.path.join(d, f"protoA_{st}.json")
        if not os.path.exists(p):
            continue
        found = True
        Ld = Loaded(json.load(open(p)), np.load(os.path.join(d, f"protoA_{st}.npz")))
        out += rep_state(Ld, kg, L_)
    if not found:
        out.append("(no outputs)")
    path = os.path.join(HERE, "RESULTS_protoA.md") if not o.smoke else os.path.join(d, "RESULTS_protoA_smoke.md")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    print("wrote", path, f"({len(out)} lines)")


# ----------------------------------------------------------------------------------------------
def main_run(o):
    sts = [s for s in o.states.split(",")]
    todo = []
    for s in sts:
        if os.path.exists(os.path.join(outdir(o.smoke), f"protoA_{s}.json")) and not o.force:
            print("skip (done):", s); continue
        todo.append((s, o))
    t0 = time.time()
    if o.procs <= 1 or len(todo) <= 1:
        for a in todo:
            job(a)
    else:
        with Pool(min(o.procs, 2), maxtasksperchild=1) as pl:
            for r in pl.imap_unordered(job, todo):
                print("finished", r, flush=True)
    print(f"protoA: all done in {time.time() - t0:.0f} s", flush=True)


def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "run", "report"])
    ap.add_argument("--states", default="i,ii,iii")
    ap.add_argument("--R", type=int, default=64); ap.add_argument("--T", type=int, default=40)
    ap.add_argument("--pre", type=int, default=48); ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--procs", type=int, default=2); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    o = ap.parse_args(argv)
    if o.pre < LMAX + 4:
        ap.error("--pre must be >= LMAX + 4 = 20")
    if o.T < 2:
        ap.error("--T must be >= 2")
    return o


if __name__ == "__main__":
    o = parse(sys.argv[1:])
    if o.cmd == "check":
        sys.exit(0 if check(o) else 1)
    elif o.cmd == "report":
        report(o)
    else:
        main_run(o)
