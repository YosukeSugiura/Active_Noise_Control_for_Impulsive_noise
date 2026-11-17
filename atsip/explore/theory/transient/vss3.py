"""GS-V2 (direction-based schedule) with the geometric exponent a=0.25 instead of 0.5; same sources as vss2 final."""
import numpy as np
from simx import *
from vss2 import Ctrl2
from multiprocessing import Pool
def unit(arg):
    tag,pc=arg
    data=MFm.build_pc(O.cms_symmetric_stable(1.45,O.N_NOISE,int(tag[-1]))) if pc else get(tag)
    Rall=ML.prepass(data)["Rall"]; out={}
    for nm,a in [("GS-V2 a=0.5",0.5),("GS-V2 a=0.25",0.25)]:
        s=run(data,Rall,kind="W",mu=0.0,a=a,ctrl=Ctrl2(kmu=2.0,qlo=0.05,qhi=0.2))
        if pc: sc=MFm.score_pc(s,data); out[nm]=np.array([sc["L1a"],sc["L1b"],sc["L1w"]])
        else: out[nm]=scores(s,data)
    return tag,out
if __name__=="__main__":
    jobs=[(t,False) for t in ["G0","G1","cms145_s0","cms145_s1","cms145_s2","a120_s0","a120_s1"]]+[("pc%d"%i,True) for i in range(3)]
    with Pool(2) as pl: res=dict(pl.map(unit,jobs))
    for grp,sel in [("Gaussian",["G0","G1"]),("alpha1.45",["cms145_s0","cms145_s1","cms145_s2"]),("alpha1.2",["a120_s0","a120_s1"]),("path change 15-17/25-30/whole",["pc0","pc1","pc2"])]:
        for k in res["G0"]: print(grp,k,np.round(np.mean([res[t][k] for t in sel],0),2))
