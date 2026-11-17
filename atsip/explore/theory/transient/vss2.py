"""Direction-based schedule, version 2 (gain-normalised statistics + per-frame cap chi/g_n).
   g_n = mean_k|R_n|^2 / Pbar_n (reference only);  u_n = dir_n / max(g_n,1)
   ubar <- EMA_bd(u);  C <- EMA_b(<u_n, ubar_{n-L}>);  Q <- EMA_b(|u_n|^2);  q = max(C,0)/Q
   mu_n = min( clip(kmu*q, mu_lo, mu_hi), chi/g_n );   GS: om_n = 1 - clip((q-qlo)/(qhi-qlo),0,1)"""
import numpy as np, sys
from simx import *
from multiprocessing import Pool
class Ctrl2:
    def __init__(s,kmu=1.0,bd=0.1,b=0.05,L=8,mu_lo=0.01,mu_hi=0.3,chi=0.5,qlo=None,qhi=None,qinit=0.15,norm=True):
        s.__dict__.update(kmu=kmu,bd=bd,b=b,L=L,lo=mu_lo,hi=mu_hi,chi=chi,qlo=qlo,qhi=qhi,norm=norm)
        s.ubar=np.zeros(M); s.buf=[np.zeros(M)]*L; s.C=None; s.Q=None; s.q=qinit; s.g=1.0; s.qs=[]
    def step(s,i,info):
        s.g=float(np.mean(np.abs(info["R"])**2)/info["Pbar"])
        mu=float(np.clip(s.kmu*s.q,s.lo,s.hi)); mu=min(mu,s.chi/max(s.g,1e-9)) if s.chi else mu
        om=1.0 if s.qlo is None else 1.0-float(np.clip((s.q-s.qlo)/(s.qhi-s.qlo),0,1))
        return mu,om
    def post(s,i,g,mu):
        u=g/max(s.g,1.0) if s.norm else g
        c=float(u@s.buf[0]); qq=float(u@u)
        if i>=s.L+4:
            if s.C is None: s.C,s.Q=s.q*qq,qq
            else: s.C=(1-s.b)*s.C+s.b*c; s.Q=(1-s.b)*s.Q+s.b*qq
            s.q=max(s.C,0.0)/(s.Q+1e-300)
        s.ubar=(1-s.bd)*s.ubar+s.bd*u if i>0 else u.copy()
        s.buf=s.buf[1:]+[s.ubar.copy()]; s.qs.append(s.q)
        return mu
VAR={"SF-V2":("SF",dict()),"SF-V2 nocap":("SF",dict(chi=None)),"SF-V2 nonorm":("SF",dict(norm=False)),"SF-V2 k2":("SF",dict(kmu=2.0)),
     "SF-V2 lo.02":("SF",dict(mu_lo=0.02)),"SF-V2 slow":("SF",dict(b=0.02,bd=0.05)),"SF-V2 chi1":("SF",dict(chi=1.0)),
     "GS-V2 .02/.1":("W",dict(qlo=0.02,qhi=0.1)),"GS-V2 .05/.2":("W",dict(qlo=0.05,qhi=0.2)),"GS-V2 .01/.05":("W",dict(qlo=0.01,qhi=0.05)),"GS-V2 k2 .05/.2":("W",dict(kmu=2.0,qlo=0.05,qhi=0.2))}
def cfgs(nb,var):
    wr,mm=ramp(nb,4,3)
    base={"SF":dict(kind="SF",mu=0.1),"SF+ramp":dict(kind="SF",mu=0.05*ramp(nb,2,3)[1]),"GS":dict(kind="W",mu=0.05*mm,a=0.5,om=wr)}
    base.update(var); return base
def unit(arg):
    tag,pc,var=arg
    data=MFm.build_pc(O.cms_symmetric_stable(1.45,O.N_NOISE,int(tag[-1]))) if pc else get(tag)
    Rall=ML.prepass(data)["Rall"]; nb=len(data["d"])//B; out={}
    for k,c in cfgs(nb,var).items():
        if isinstance(c,dict): s=run(data,Rall,**c)
        else: ct=Ctrl2(**c[1]); s=run(data,Rall,kind=c[0],mu=0.0,a=0.5,ctrl=ct)
        idx=[1870,1880,1900,1950,2100,3000] if pc else [8,31,125,500,1500,3700]
        if pc: sc=MFm.score_pc(s,data); out[k]=(np.array([sc["L1a"],sc["L1b"],sc["L1w"]]),s["mu"][idx],s["om"][idx])
        else: out[k]=(scores(s,data),s["mu"][idx],s["om"][idx])
    return tag,pc,out
def report(res,groups):
    keys=list(res[0][2].keys())
    for grp,sel in groups:
        print("==",grp)
        for k in keys:
            v=np.mean([r[2][k][0] for r in res if r[0] in sel],0); r0=[r for r in res if r[0]==sel[0]][0][2][k]
            print("  %-16s %s | mu %s om %s"%(k,np.round(v,2),np.round(r0[1],3),np.round(r0[2],2)))
if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="final":
        var={k:VAR[k] for k in sys.argv[2].split(",")}
        jobs=[(t,False,var) for t in ["G0","G1","cms145_s0","cms145_s1","cms145_s2","a120_s0","a120_s1"]]+[("pc%d"%i,True,var) for i in range(3)]
        groups=[("Gaussian (2): L1 whole/first5/last10",["G0","G1"]),("alpha 1.45 (3)",["cms145_s0","cms145_s1","cms145_s2"]),("alpha 1.2 (2)",["a120_s0","a120_s1"]),("path change (3): L1 15-17 s / 25-30 s / whole",["pc0","pc1","pc2"])]
    else:
        jobs=[(t,False,VAR) for t in ["G0","cms145_s0","cms145_s1","a120_s0"]]+[("pc%d"%i,True,VAR) for i in range(2)]
        groups=[("G0",["G0"]),("alpha 1.45 (s0,s1)",["cms145_s0","cms145_s1"]),("a120_s0",["a120_s0"]),("path change (2): 15-17 / 25-30 / whole",["pc0","pc1"])]
    with Pool(2) as pl: res=pl.map(unit,jobs)
    report(res,groups)
