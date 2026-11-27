"""Single-channel APFM simulation kernel, with explicit experiment parameters.

The selection follows Yin et al. (2023), Table 1, and the constrained
frequency-domain update follows equations (10)–(15).
"""
from dataclasses import dataclass
import math
import time

import final as F
from data.diag_band import error_peak

np = F.np
BANK, CURRENT, ZERO = range(3)


@dataclass(frozen=True)
class Settings:
    mu_tilde: float
    mu_max: float
    eta: float
    beta1: float
    beta2: float
    beta3: float

    def __post_init__(self):
        if not all(math.isfinite(x) and x > 0 for x in (self.mu_tilde, self.mu_max, self.eta)):
            raise ValueError("step sizes and regularization must be positive and finite")
        if not all(.9 <= x < 1 for x in (self.beta1, self.beta2, self.beta3)):
            raise ValueError("forgetting factors must lie in [0.9, 1)")


def simulate(dd, settings, *, bank=None, checkpoints=()):
    """Run APFM with a trained bank filter, or just VSS-NFBFxLMS with bank=None."""
    started = time.monotonic()
    m, b, nfft = F.M, F.B, 2*F.M
    x, d, models = dd["x"], dd["d"], dd["hSm_seq"]
    if len(x) != len(d) or len(x) % b:
        raise ValueError("aligned full blocks required")
    if bank is not None:
        bank = np.asarray(bank, dtype=float)
        if bank.shape != (m,) or not np.isfinite(bank).all():
            raise ValueError("invalid fixed bank filter")
        y_bank = F.p.signal.oaconvolve(x, bank)[:len(x)]
        bank_hist = np.zeros(max(len(h) for _,h in models)-1)
        bank_state = (None, None)
    xpad = np.pad(x, (nfft,0))
    model_spectra = [(block,np.fft.rfft(h,nfft)) for block,h in models]
    e = np.zeros_like(d)
    w = np.zeros(m)
    ep = np.zeros(nfft)
    raw_gradient = np.zeros(m)
    norm = np.ones(nfft//2+1)
    px, pe = 0., 0.
    state, model_state = (None,None), (None,None)
    plant = dd["hS_seq"]
    yhist = np.zeros(max(len(h) for _,h in (*plant,*models))-1)
    selections, saved = [], {}
    primary_power = dd.get("primary_power", float(np.mean(d*d)))
    n_done, failure, div = len(d), None, False
    for block, start in enumerate(range(0,len(d),b)):
        stop = start+b
        xwindow = xpad[stop:stop+nfft]
        emitted = np.convolve(xwindow[-(m+b-1):],w,"valid")
        ripple, state = F.filtered_block(state,block,yhist,emitted,plant)
        eb = d[start:stop]+ripple
        e[start:stop] = eb
        if F.diverged(eb,primary_power):
            div, n_done = True, start
            break
        choice = CURRENT
        if bank is not None:
            modeled, model_state = F.filtered_block(model_state,block,yhist,emitted,models)
            candidate = y_bank[start:stop]
            bank_ripple, bank_state = F.filtered_block(bank_state,block,bank_hist,candidate,models)
            if bank_hist.size:
                bank_hist = np.concatenate((bank_hist,candidate))[-bank_hist.size:]
            dhat = eb-modeled
            bank_error = dhat+bank_ripple
            choice = int(np.argmin((bank_error@bank_error, eb@eb, dhat@dhat)))
        if yhist.size:
            yhist = np.concatenate((yhist,emitted))[-yhist.size:]
        X = np.fft.rfft(xwindow)
        R = F.seq_at(model_spectra,block)*X
        ep[m:] = e[max(0,stop-m):stop] if stop >= m else 0.
        if stop < m:
            ep[-stop:] = e[:stop]
        xb = x[start:stop]
        norm = settings.beta1*norm+(1-settings.beta1)*np.abs(X)**2
        px = settings.beta2*px+(1-settings.beta2)*float(xb@xb)
        pe = settings.beta3*pe+(1-settings.beta3)*float(eb@eb)
        mu = min(settings.mu_max,settings.mu_tilde/math.sqrt(px+pe+settings.eta))
        if choice == ZERO:
            raw_gradient = settings.beta1*raw_gradient
            norm.fill(1.)
            px, pe = 0., 0.
            w = -mu*raw_gradient
        else:
            raw_gradient = np.fft.irfft(R.conj()*np.fft.rfft(ep)/norm,nfft)[:m]
            w = (bank if choice == BANK else w)-mu*raw_gradient
        if not np.isfinite(w).all():
            div, n_done = True, start
            break
        selections.append(choice)
        if block+1 in checkpoints:
            saved[block+1] = w.copy()
    if div:
        failure = dict(error_peak(e,n_done),sample=n_done)
        e[n_done:] = np.nan
    return dict(e=e,w=w,div=div,n_done=n_done,failure=failure,seconds=time.monotonic()-started,
                choices=selections,checkpoints=saved)
