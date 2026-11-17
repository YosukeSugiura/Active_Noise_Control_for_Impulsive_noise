"""Simulator check of the model's horizon prediction: geometric exponent a=0.25 with a long ramp (T=8 s)."""
import numpy as np
from simx import *
from multiprocessing import Pool
def unit(tag):
    data=get(tag); Rall=ML.prepass(data)["Rall"]; nb=len(data["d"])//B; out={}
    for nm,(a,T,m0,mu) in {"GS a.5 T4 m3 mu.05":(0.5,4,3,0.05),"a.25 T8 m5 mu.05":(0.25,8,5,0.05),"a.25 T8 m3 mu.1":(0.25,8,3,0.1),"a.25 T4 m3 mu.05":(0.25,4,3,0.05),"a.5 T2 m5 mu.1":(0.5,2,5,0.1),"a.25 T8 m3 mu.05":(0.25,8,3,0.05)}.items():
        wr,mm=ramp(nb,T,m0); s=run(data,Rall,kind="W",mu=mu*mm,a=a,om=wr); out[nm]=np.r_[scores(s,data),10*np.log10(s["ex"][-40:].mean())]
    return tag,out
if __name__=="__main__":
    tags=["G0","G1","cms145_s0","cms145_s1","cms145_s2","a120_s0","a120_s1","col145_s10"]
    with Pool(2) as pl: res=dict(pl.map(unit,tags))
    for grp,sel in [("Gaussian",["G0","G1"]),("alpha1.45",["cms145_s0","cms145_s1","cms145_s2"]),("alpha1.2",["a120_s0","a120_s1"]),("colored (1)",["col145_s10"])]:
        print("==",grp,"L1 whole/first5/last10, end excess dB")
        for k in res[tags[0]]: print("   %-20s"%k,np.round(np.mean([res[t][k] for t in sel],0),2))
