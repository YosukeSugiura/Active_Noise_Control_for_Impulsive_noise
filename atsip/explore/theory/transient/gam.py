import sys; sys.argv=["x"]
import numpy as np
from model import *
sn=s/s.mean(); So_f=np.abs(np.fft.rfft(gM,G_))**2
V0=np.abs(np.fft.rfft(-wM,G_))**2
print("s: mean 1, min %.2e max %.2f, geo-mean %.3f"%(sn.min(),sn.max(),np.exp(np.mean(np.log(sn)))))
def gam(a,V,sf=False,eps=1e-2):
    if sf: lam=sn/(sn+eps); num=np.mean(sn*lam*V)/np.mean(lam*lam*V); Nphi=lam*lam*So_f
    else:  lam=sn**(1-a); num=np.mean(sn*lam*V)/np.mean(lam*lam*V); Nphi=lam*lam*So_f
    go=np.mean(Nphi/(sn+1e-6))/np.mean(Nphi)
    return num,go
# late-stage deviation spectra: after t blocks at step eta, |V|^2 ~ V0 * exp(-2 eta t lam c)
for name,a,sf in [("SF",None,True),("B0 a=1",1.0,False),("GS a=0.5",0.5,False),("a=0.25",0.25,False)]:
    out=[]
    for T in [0,20,100,500]:
        lam=(sn/(sn+1e-2)) if sf else sn**(1-a)
        V=V0*np.exp(-2*0.05*0.5*T*lam)
        gv,go=gam(a,V,sf); out.append("T=%d gV %.3f"%(T,gv))
    Vw=np.ones_like(sn); gvw,go=gam(a,Vw,sf)
    print("%-9s gamma_o %.3f | white-V gV %.3f | %s"%(name,go,gvw," ".join(out)))
print("--- regularised")
for eps in [1e-2,1e-3]:
    e=So_f*sn/(sn+eps)**2; p=So_f*sn**2/(sn+eps)**2
    print("SF eps %.0e: gamma_o %.3f ; share of Euclid noise from bins with s<0.1: %.2f ; Phi share %.2f"%(eps,e.mean()/p.mean(),e[sn<0.1].sum()/e.sum(),p[sn<0.1].sum()/p.sum()))
print("fraction of bins s<0.1: %.3f ; So share in those bins %.2f"%(np.mean(sn<0.1),So_f[sn<0.1].sum()/So_f.sum()))
# a Phi-weighted estimator: weight u by F (min-phase) i.e. use <F u, F ubar>: then gamma_o=1, gV=1 for SF
