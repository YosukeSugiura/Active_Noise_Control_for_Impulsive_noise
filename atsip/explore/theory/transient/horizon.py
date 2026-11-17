"""Horizon (30 s) optimum inside the ramp family, evaluated on the model (conditional on g_n):
   om_n = min(1, t/T); mu_n = mu_inf (1+(m0-1)(1-om_n)) / (1 + max(n-n_T,0)/n_d);  Lam = (1-om) Srr + om Srr^a Pbar^(1-a)."""
import sys; sys.argv=["x"]
import numpy as np, itertools
from model import *
from multiprocessing import Pool
nb=3750; setpar(2.0)
GRID=list(itertools.product([0.25,0.5,0.75],[1,2,4,8],[1,3,5],[0.05,0.1],[0,1000]))
def ev(arg):
    tag,(a,T,m0,mu,nd)=arg
    wr,mm=ramp(nb,T,m0); n=np.arange(nb); nT=int(T*FS/B)
    dec=1.0 if nd==0 else 1/(1+np.maximum(n-nT,0)/nd)
    r=simulate(nb,PAR[CLS[tag]],mu=mu*mm*dec,theta=[("W",a,float(np.round(x*20)/20)) for x in wr],gseq=gload(tag),su=False)
    return tag,(a,T,m0,mu,nd),xdb(r["E"])
if __name__=="__main__":
    jobs=[(t,g) for t in ["G0","cms145_s0"] for g in GRID]
    with Pool(2) as pl: res=pl.map(ev,jobs,chunksize=8)
    for tag in ["G0","cms145_s0"]:
        rr=sorted([r for r in res if r[0]==tag],key=lambda r:r[2][0])
        print("==",tag,"best 6 of",len(rr),"(a, T[s], m0, mu_inf, n_d) -> model xdB whole/first5/last10")
        for r in rr[:6]: print("   ",r[1],np.round(r[2],3))
        cur=[r for r in rr if r[1]==(0.5,4,3,0.05,0)][0]; print("    current GS ramp (0.5,4,3,0.05,0):",np.round(cur[2],3)," rank",rr.index(cur)+1)
        for a in [0.25,0.5,0.75]:
            b=[r for r in rr if r[1][0]==a][0]; print("    best with a=%.2f:"%a,b[1],np.round(b[2],3))
        for nd in [0,1000]:
            b=[r for r in rr if r[1][4]==nd][0]; print("    best with n_d=%d:"%nd,b[1],np.round(b[2],3))
