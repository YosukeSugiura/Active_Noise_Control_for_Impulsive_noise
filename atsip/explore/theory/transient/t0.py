import time,numpy as np
from simx import *
for tag in ["a145","G0"]:
    data=get(tag); Rall=ML.prepass(data)["Rall"]; nb=len(data["d"])//B
    t=time.time()
    wr,mm=ramp(nb,4,3)
    for nm,kw in [("B0",dict(kind="W",mu=0.1,a=1.0)),("W0.5",dict(kind="W",mu=0.1,a=0.5)),("GS",dict(kind="W",mu=0.05*mm,a=0.5,om=wr)),
                  ("SF",dict(kind="SF",mu=0.1)),("SF+ramp",dict(kind="SF",mu=0.05*ramp(nb,2,3)[1])),("MF",dict(kind="MF",mu=0.05))]:
        s=run(data,Rall,**kw); print(tag,nm,np.round(scores(s,data),2),"ex end dB %.2f"%(10*np.log10(s["ex"][-20:].mean())),"exLS %.2f"%(10*np.log10(excess(data["w_opt"]))))
    print(time.time()-t)
