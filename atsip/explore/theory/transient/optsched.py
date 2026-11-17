"""Model-optimal (greedy) schedules vs the fixed ramps, in the model (predicted) and replayed open loop in the simulator (measured)."""
import sys; sys.argv=["x"]
import numpy as np
from model import *
from multiprocessing import Pool
nb=3750; setpar(2.0)
OMS=[0.0,0.25,0.5,0.75,1.0]; AS=[1.0,0.75,0.5,0.25,0.0]
def unit(tag):
    gs=gload(tag); par=PAR[CLS[tag]]; out={}
    wr,mm=ramp(nb,4,3)
    fx={"SF+ramp":simulate(nb,par,mu=0.05*ramp(nb,2,3)[1],theta=("SF",),gseq=gs,su=False),
        "GS":simulate(nb,par,mu=0.05*mm,theta=[("W",0.5,float(np.round(x,2))) for x in wr],gseq=gs,su=False)}
    gr={"SF*":simulate(nb,par,greedy=[("SF",)],gseq=gs,su=False),
        "SF*(moments)":simulate(nb,par,greedy=[("SF",)],gseq=gs,su=False,know_g=False),
        "GS*(om)":simulate(nb,par,greedy=[("W",0.5,o) for o in OMS],gseq=gs,su=False),
        "GS*(a)":simulate(nb,par,greedy=[("W",a,1.0) for a in AS],gseq=gs,su=False),
        "GS*(om,moments)":simulate(nb,par,greedy=[("W",0.5,o) for o in OMS],gseq=gs,su=False,know_g=False)}
    # replay in the simulator
    data=get(tag); Rall=ML.prepass(data)["Rall"]
    sims={"SF+ramp":run(data,Rall,kind="SF",mu=0.05*ramp(nb,2,3)[1]),"GS":run(data,Rall,kind="W",mu=0.05*mm,a=0.5,om=wr)}
    for k,r in gr.items():
        for sf in (1.0,0.5):
            mu=sf*r["mu"]
            if k.startswith("SF"): s=run(data,Rall,kind="SF",mu=mu)
            else: s=run(data,Rall,kind="W",mu=mu,a=np.array([t[1] for t in r["th"]]),om=np.array([t[2] for t in r["th"]]))
            sims[k+" x%.1f"%sf]=s
    return tag,{k:(xdb(v["E"]),v["mu"],[t[1:] for t in v["th"]]) for k,v in {**fx,**gr}.items()},{k:(scores(s,data),xdb(s["ex"],8)) for k,s in sims.items()}
if __name__=="__main__":
    tags=["G0","cms145_s0","cms145_s1","a120_s0"]
    with Pool(2) as pl: res=pl.map(unit,tags)
    T=[8,31,125,500,1500,3700]
    for tag,mo,si in res:
        print("==",tag)
        for k,(x,mu,th) in mo.items():
            extra=""
            if "*" in k:
                extra=" | mu*(n) at blk %s: %s"%(T,np.round(mu[T],3))
                if k.startswith("GS"): extra+=" | theta*: "+str([ (th[t][1] if "om" in k else th[t][0]) for t in T])
            print("  model %-16s xdB whole/first5/last10 %s%s"%(k,np.round(x,3),extra))
        for k,(sc,x) in si.items(): print("  sim   %-20s L1 whole/first5/last10 %s | xdB %s"%(k,np.round(sc,2),np.round(x,3)))
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),"optsched_res.npy"),np.array(res,dtype=object),allow_pickle=True)
