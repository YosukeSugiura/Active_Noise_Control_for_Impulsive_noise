import sys; sys.argv=["x"]
import numpy as np
from vss2 import *
data=get("G0"); Rall=ML.prepass(data)["Rall"]
class Rec:
    def __init__(s): s.us=[]
    def step(s,i,info): return s.mu,1.0
    def post(s,i,g,mu): s.us.append(g.copy()); return mu
for mu in [0.05,0.1,0.2]:
    ct=Rec(); ct.mu=mu; r=run(data,Rall,kind="SF",mu=0.0,ctrl=ct,rec=1)
    U=np.array(ct.us)[1500:]; Q=(U**2).sum(1).mean()
    out=[np.mean(np.sum(U[:len(U)-k]*U[k:],1))/Q for k in [4,5,6,8,12,16,24]]
    UP=U@Phi; QP=np.sum(U*UP,1).mean()
    outP=[np.mean(np.sum(U[:len(U)-k]*UP[k:],1))/QP for k in [4,5,6,8,12,16,24]]
    print("mu %.2f  C_k/Q (k=4,5,6,8,12,16,24): Euclid %s | Phi %s | pred -mu/(2 kappa*) = %.4f"%(mu,np.round(out,4),np.round(outP,4),-mu/2/1.06))
