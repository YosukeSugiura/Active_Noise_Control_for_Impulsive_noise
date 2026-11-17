"""Generic delayless FD-FxNLMS simulator with arbitrary per-block (mu, a/omega) schedule or an adaptive controller.
Same loop as geosched.sim_sched / specfac.sim_sf / mfblms.sim_new.  Records exact excess v'Phi v / Jo (white model)."""
import os, sys
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal, linalg
EXP=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".."))
sys.path.insert(0,EXP); sys.path.insert(0,os.path.join(EXP,"theory","closedform"))
import dissect as D, explore as X, mixlam as ML, specfac as SFm, mfblms as MFm, inclass as IC
import base as BS
FS,M,N,B,NB=D.FS,D.M,D.N,D.B,D.NB
O=X.O
wM,gM=BS.wiener(M); Jo=gM@gM; Jd=BS.hP@BS.hP
Phi=linalg.toeplitz(BS.xc(BS.hS,BS.hS,np.arange(M)))
def get(tag):
    if tag.startswith("G"):   # Gaussian seed
        return D.build(IC.make_input("A",int(tag[1:])))
    return D.load_data(*D.KIND[tag])
def excess(w,wref=None,PhiM=None):
    v=w-(wM if wref is None else wref); return float(v@(Phi if PhiM is None else PhiM)@v)/Jo
def run(data,Rall,kind,mu,a=0.5,om=None,ctrl=None,rec=8,eps_reg=1e-2,nb=None,w0=None):
    """kind: 'W' (Lam=(1-om)Srr+om Srr^a Pbar^(1-a)), 'SF', 'MF'. mu, a, om: scalar or per-block arrays.
    ctrl: object with .step(i, info)->(mu_i, om_i) called before the update (info: e block, R, E, Srr, w, dir fn)."""
    d,xq,hS,S2=data["d"],data["xq"],data["hS"],data["S2"]
    L=len(d); nb=L//B if nb is None else nb
    swv=sliding_window_view(xq,M); w=np.zeros(M) if w0 is None else w0.copy(); zi=np.zeros(len(hS)-1); e_all=np.zeros(L)
    shared=X.Shared(S2); ex=[]; mus=[]; oms=[]
    arr=lambda v,i: (v[i] if np.ndim(v) else v)
    for i in range(nb):
        s0=i*B
        Xb=swv[s0+N-M+1:s0+N-M+1+B][:,::-1]
        ys,zi=signal.lfilter(hS,[1.0],Xb@w,zi=zi)
        e_all[s0:s0+B]=d[s0:s0+B]+ys
        E=ML.frame_E(e_all,s0,M)
        s=shared.step(Rall[i],E); Srr,Pbar,eps=s["Srr"],s["Pbar"],s["eps"]
        mu_i=arr(mu,i); om_i=1.0 if om is None else arr(om,i); a_i=arr(a,i)
        if ctrl is not None:
            mu_i,om_i=ctrl.step(i,dict(e=e_all[s0:s0+B],R=Rall[i],E=E,Srr=Srr,Pbar=Pbar,eps=eps,w=w,xq=xq,s0=s0,d=d))
        if kind=="W":
            Lam=(1-om_i)*Srr+om_i*Srr**a_i*Pbar**(1-a_i)+eps
            g=np.fft.irfft(np.conj(Rall[i])*E/Lam,N)[:M]
        elif kind=="SF":
            g=SFm.sf_dir(Rall[i],E,Srr,Pbar,eps,eps_reg)
        elif kind=="MF":
            g=MFm.mf_dir(Rall[i],E,Srr+eps)
        if ctrl is not None and hasattr(ctrl,"post"):
            mu_i=ctrl.post(i,g,mu_i)
        w=w-2*mu_i*B/M*g
        mus.append(mu_i); oms.append(om_i)
        if not np.isfinite(w).all(): e_all[s0+B:]=np.nan; break
        if (i+1)%rec==0: ex.append(excess(w,data.get("wref"),data.get("PhiM")))
    return dict(e=e_all,w=w,ex=np.array(ex),mu=np.array(mus),om=np.array(oms))
def L1(sim,data,t0=0,t1=None):
    anr=O.anr_curve(sim["e"],data["d"]); ts=(63+64*np.arange(len(anr)))/FS
    t1=ts[-1]+1 if t1 is None else t1
    return float(anr[(ts>=t0)&(ts<t1)].mean())
def scores(sim,data):
    anr=O.anr_curve(sim["e"],data["d"]); n10=len(anr)//10; n5=5*FS//64
    return np.array([anr.mean(),anr[:n5].mean(),anr[-n10:].mean()])
def ramp(nb,T,m0):
    t=np.arange(nb)*B/FS; wr=np.ones(nb) if T==0 else np.minimum(1,t/T); return wr,1+(m0-1)*(1-wr)
