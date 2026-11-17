import sys; sys.argv=["x"]
import numpy as np
from model import *
sn=s/s.mean(); So_f=np.abs(np.fft.rfft(gM,G_))**2
dof=lambda e: M*e.mean()**2/np.mean(e**2)
eps=1e-2
cases={"SF Euclid":So_f*sn/(sn+eps)**2,"SF Phi (y)":So_f*sn**2/(sn+eps)**2,"B0 Euclid":So_f/(sn+1e-3),"B0 Phi":So_f*sn/(sn+1e-3)*1.0,
       "GS a=.5 Euclid":So_f*sn**0.0,"GS a=.5 Phi":So_f*sn,"a=0 (unnorm) Euclid":So_f*sn}
tau=3.75
for k,e in cases.items():
    D=dof(e)
    for bd,b in [(0.1,0.05)]:
        sq=tau/2*np.sqrt(b*bd/D)
    g=1/(np.e*(4+1/0.1)); jit=tau/2*np.sqrt(g*0.1/D)
    print("%-20s D_eff %6.1f | sigma_q (beta .05, beta_d .1) %.4f | integral rule: gamma %.3f, std(mu) %.4f"%(k,D,sq,g,jit))
# direct: variance of c_n from the saved run
z=np.load("qchk.npz"); U=z["U"][2500:].astype(float); U=U-U.mean(0)
ub=np.zeros(M); buf=[np.zeros(M)]*4; cs=[]; qs=[]
for u in U:
    cs.append(u@buf[0]); qs.append(u@u); ub=0.9*ub+0.1*u; buf=buf[1:]+[ub.copy()]
cs=np.array(cs[50:]); Q=np.mean(qs)
print("measured (Euclid, L=4, beta_d .1): std(c_n)/Q %.4f -> D_eff %.1f ; mean %.4f"%(cs.std()/Q,0.1*tau/2/(cs.std()/Q)**2,cs.mean()/Q))
