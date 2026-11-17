"""MFBLMS mechanism: (1) where the excess jumps (vs g_i); (2) variants: warm-start Srr, frozen oracle Lam with frame-power scaling."""
import numpy as np, sys
from simx import *
tag=sys.argv[1] if len(sys.argv)>1 else "a120_s0"
data=get(tag); Rall=ML.prepass(data)["Rall"]; nb=1000
g=IC.gain_seq(Rall)
res={}
for nm,kw in [("MF0.05",dict(kind="MF",mu=0.05)),("MF0.1",dict(kind="MF",mu=0.1)),("SF0.1",dict(kind="SF",mu=0.1)),("B0_0.1",dict(kind="W",mu=0.1,a=1.0)),("SF0.2",dict(kind="SF",mu=0.2)),("MF0.2",dict(kind="MF",mu=0.2))]:
    s=run(data,Rall,rec=1,nb=nb,**kw); res[nm]=10*np.log10(s["ex"])
jm=np.diff(res["MF0.1"]); idx=np.argsort(jm)[-8:][::-1]
print(tag,"largest one-block increases of excess [dB] for MF0.1, with g_i and the same-block change of others")
for i in idx: print("  blk %4d g=%.1f g_prev3=%s | MF0.1 %+.1f MF0.05 %+.1f SF0.1 %+.1f SF0.2 %+.1f B0 %+.1f | level MF0.1 %.1f"%(i+1,g[i+1],np.round(g[i-2:i+1],1),jm[i],np.diff(res["MF0.05"])[i],np.diff(res["SF0.1"])[i],np.diff(res["SF0.2"])[i],np.diff(res["B0_0.1"])[i],res["MF0.1"][i]))
for nm in res: 
    d_=np.diff(res[nm]); up=d_[d_>0].sum()
    top=np.argsort(g[1:nb])[-20:]
    print("  %-7s total upward movement %.1f dB; in the 20 largest-g blocks: %.1f dB ; ex at 250/500/1000: %s"%(nm,up,d_[top][d_[top]>0].sum(),np.round(res[nm][[249,499,999]],1)))
