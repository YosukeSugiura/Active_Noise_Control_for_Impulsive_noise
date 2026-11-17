"""Check: per-frame direction noise power vs frame gain g and participation ratio pi (Sec. 24 of DERIVATION_vss.md)."""
import sys; sys.argv=["x"]
import numpy as np, os
from simx import *
p=np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),"probe.npz"))
def r2(y,X):
    co=np.linalg.lstsq(X,y,rcond=None)[0]; return 1-np.sum((y-X@co)**2)/np.sum((y-y.mean())**2),co
for tag in ["G0","cms145_s0","a120_s0"]:
    data=get(tag); x=data["x"]; nb=len(data["d"])//B
    g=p[tag+"|g"]; n1=p[tag+"|SF|n1"]; r2s=p[tag+"|SF|init|r2"][100:]; r1=p[tag+"|SF|init|r1"][100:]
    pi=np.array([np.sum(x[i*B:i*B+N]**4)/np.sum(x[i*B:i*B+N]**2)**2 for i in range(nb)])
    g=g[100:100+len(n1)]; pi=pi[100:100+len(n1)]
    # n1 was computed after removing the mean direction; models (through origin)
    m={"g":g[:,None],"g^2":(g**2)[:,None],"g^2(1-pi)":(g**2*(1-pi))[:,None]}
    out=" ".join("%s R2 %.3f"%(k,r2(n1,X)[0]) for k,X in m.items())
    fl=r2s-r1**2   # fluctuation part of the signal-driven direction
    out2=" ".join("%s R2 %.3f"%(k,r2(fl[:len(g)],X[:len(fl)])[0]) for k,X in m.items())
    big=g>np.quantile(g,0.99)
    print("%-10s pi mean %.4f max %.3f | noise n1: %s | fluct: %s | top1%% g: mean g %.1f, n1/g^2 %.4f vs all %.4f, mean pi %.2f"%(tag,pi.mean(),pi.max(),out,out2,g[big].mean(),np.mean(n1[big]/g[big]**2),np.mean(n1/g**2),pi[big].mean()))
