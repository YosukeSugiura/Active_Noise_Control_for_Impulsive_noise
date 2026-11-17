"""Measured learning curves (exact excess from weights) for SF/B0/W0.5/GS on Gaussian, alpha 1.45, alpha 1.2. Saves meas.npz."""
import numpy as np, sys
from simx import *
from multiprocessing import Pool
TAGS=["G0","G1","cms145_s0","cms145_s1","a120_s0","a120_s1"]
def unit(tag):
    data=get(tag); Rall=ML.prepass(data)["Rall"]; nb=len(data["d"])//B
    out={}
    wr,mm=ramp(nb,4,3)
    cfg={"SF0.02":dict(kind="SF",mu=0.02),"SF0.05":dict(kind="SF",mu=0.05),"SF0.1":dict(kind="SF",mu=0.1),"SF0.2":dict(kind="SF",mu=0.2),"SF0.4":dict(kind="SF",mu=0.4),
         "SF+ramp":dict(kind="SF",mu=0.05*ramp(nb,2,3)[1]),
         "B0_0.05":dict(kind="W",mu=0.05,a=1.0),"B0_0.1":dict(kind="W",mu=0.1,a=1.0),"B0_0.2":dict(kind="W",mu=0.2,a=1.0),
         "W0.5_0.05":dict(kind="W",mu=0.05,a=0.5),"W0.5_0.1":dict(kind="W",mu=0.1,a=0.5),"W0.5_0.2":dict(kind="W",mu=0.2,a=0.5),
         "GS":dict(kind="W",mu=0.05*mm,a=0.5,om=wr),
         "MF0.05":dict(kind="MF",mu=0.05),"MF0.1":dict(kind="MF",mu=0.1),"MF0.02":dict(kind="MF",mu=0.02)}
    for k,kw in cfg.items():
        s=run(data,Rall,rec=2,**kw); out[tag+"|"+k]=s["ex"]; out[tag+"|"+k+"|sc"]=scores(s,data)
    A=np.abs(Rall)**2; g=IC.gain_seq(Rall)
    # per-bin gain stats with the simulator's EMA (current frame included)
    Srr=A[0].copy(); gb1=[];gb2=[]
    for i in range(len(A)):
        if i: Srr=0.95*Srr+0.05*A[i]
        q=A[i]/Srr; gb1.append(q.mean()); gb2.append((q**2).mean())
    out[tag+"|g"]=g; out[tag+"|gb1"]=np.array(gb1); out[tag+"|gb2"]=np.array(gb2)
    return out
if __name__=="__main__":
    res={}
    with Pool(2) as p:
        for o in p.imap_unordered(unit,TAGS): res.update(o)
    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)),"meas.npz"),**res)
    for t in TAGS:
        g=res[t+"|g"]; print(t,"E[g] %.3f E[g2]/E[g] %.3f | per-bin mean %.3f second %.3f (blocks>200)"%(g.mean(),(g**2).mean()/g.mean(),res[t+"|gb1"][200:].mean(),res[t+"|gb2"][200:].mean()))
        for k in ["SF0.02","SF0.05","SF0.1","SF0.2","SF0.4","SF+ramp","B0_0.05","B0_0.1","B0_0.2","W0.5_0.05","W0.5_0.1","W0.5_0.2","GS","MF0.02","MF0.05","MF0.1"]:
            ex=res[t+"|"+k]; print("   %-10s ex[dB] at blk 16/64/250/1000/end(avg last 300): "%k, np.round(10*np.log10([ex[7],ex[31],ex[124],ex[499],ex[-150:].mean()]),2), np.round(res[t+"|"+k+"|sc"],2))
