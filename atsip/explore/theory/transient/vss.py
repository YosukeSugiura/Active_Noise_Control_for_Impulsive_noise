"""Practical schedule from the update direction itself (no J_o, J_b, alpha needed):
   dbar <- EMA_bd(dir);  C <- EMA_b(<dir_n, dbar_{n-L}>);  Q <- EMA_b(|dir_n|^2);  q = max(C,0)/Q
   mu_n = clip(kmu*q, mu_lo, mu_hi);   GS: om_n = 1 - min(1, q/q0)
Tests: stationary (L1 whole/first5/last10) and the path change of mfblms.py (15-17 s, 25-30 s)."""
import numpy as np, sys
from simx import *
from multiprocessing import Pool
class Ctrl:
    def __init__(s,kmu=1.0,bd=0.05,b=0.02,L=8,mu_lo=0.0,mu_hi=0.3,q0=None,qinit=0.15):
        s.kmu,s.bd,s.b,s.L,s.lo,s.hi,s.q0=kmu,bd,b,L,mu_lo,mu_hi,q0
        s.dbar=np.zeros(M); s.buf=[np.zeros(M)]*L; s.C=None; s.Q=None; s.q=qinit; s.qs=[]
    def step(s,i,info):
        mu=float(np.clip(s.kmu*s.q,s.lo,s.hi)); om=1.0 if s.q0 is None else 1.0-min(1.0,s.q/s.q0)
        return mu,om
    def post(s,i,g,mu):
        old=s.buf[0]; c=float(g@old); qq=float(g@g)
        if i>=s.L+4:
            if s.C is None: s.C,s.Q=max(c,0.0),qq   # first usable product
            else: s.C=(1-s.b)*s.C+s.b*c; s.Q=(1-s.b)*s.Q+s.b*qq
            s.q=max(s.C,0.0)/(s.Q+1e-300)
        else: s.Q=qq if s.Q is None else s.Q
        s.dbar=(1-s.bd)*s.dbar+s.bd*g if i>0 else g.copy()
        s.buf=s.buf[1:]+[s.dbar.copy()]; s.qs.append(s.q)
        return mu
def cfgs(nb):
    wr,mm=ramp(nb,4,3)
    return {"SF":dict(kind="SF",mu=0.1),"SF+ramp":dict(kind="SF",mu=0.05*ramp(nb,2,3)[1]),"GS":dict(kind="W",mu=0.05*mm,a=0.5,om=wr),
     "SF-V":("SF",dict(kmu=1.0)),"SF-V k2":("SF",dict(kmu=2.0)),"SF-V b.05":("SF",dict(kmu=1.0,b=0.05,bd=0.1)),
     "GS-V q0=.05":("W",dict(kmu=1.0,q0=0.05)),"GS-V q0=.1":("W",dict(kmu=1.0,q0=0.1)),"GS-V q0=.02":("W",dict(kmu=1.0,q0=0.02)),"GS-V k2 q0=.05":("W",dict(kmu=2.0,q0=0.05))}
def unit(arg):
    tag,pc=arg
    if pc:
        data=MFm.build_pc(O.cms_symmetric_stable(1.45,O.N_NOISE,int(tag[-1])))
    else: data=get(tag)
    Rall=ML.prepass(data)["Rall"]; nb=len(data["d"])//B; out={}
    for k,c in cfgs(nb).items():
        if isinstance(c,dict): s=run(data,Rall,**c); qs=None
        else:
            ct=Ctrl(**c[1]); s=run(data,Rall,kind=c[0],mu=0.0,a=0.5,ctrl=ct); qs=np.array(ct.qs)
        if pc: sc=MFm.score_pc(s,data); out[k]=(np.array([sc["L1a"],sc["L1b"],sc["L1w"]]),s["mu"][[1870,1880,1900,1950,2100,3000]],s["om"][[1870,1880,1900,1950,2100,3000]])
        else: out[k]=(scores(s,data),s["mu"][[8,31,125,500,1500,3700]],s["om"][[8,31,125,500,1500,3700]])
    return tag,pc,out
if __name__=="__main__":
    jobs=[(t,False) for t in ["G0","G1","cms145_s0","cms145_s1","cms145_s2","a120_s0","a120_s1"]]+[("pc%d"%i,True) for i in range(3)]
    with Pool(2) as pl: res=pl.map(unit,jobs)
    keys=list(res[0][2].keys())
    for grp,sel in [("Gaussian (2)",["G0","G1"]),("alpha 1.45 (3)",["cms145_s0","cms145_s1","cms145_s2"]),("alpha 1.2 (2)",["a120_s0","a120_s1"]),("path change (3): L1 15-17 s / 25-30 s / whole",["pc0","pc1","pc2"])]:
        print("==",grp)
        for k in keys:
            v=np.mean([r[2][k][0] for r in res if r[0] in sel],0); r0=[r for r in res if r[0]==sel[0]][0][2][k]
            print("  %-16s %s | mu at probes %s om %s"%(k,np.round(v,2),np.round(r0[1],3),np.round(r0[2],2)))
