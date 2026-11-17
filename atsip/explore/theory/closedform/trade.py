import numpy as np
from scipy import linalg
from base import *
dB=lambda x:10*np.log10(x+1e-300)
M,N,B,fs=512,1024,128,16000
kern=lambda spec:np.fft.irfft(spec,G)
w,g=wiener(M); Jo=g@g; Jd=hP@hP
Phi=linalg.toeplitz(xc(hS,hS,np.arange(M))); ch=np.linalg.cholesky(Phi); Pb=Phi[0,0]
LL=4*N; lags=np.arange(-LL,LL+1); p=xc(hS,g,lags); Lm=np.arange(M)[:,None]-lags[None,:]
Sdd=np.abs(Pw)**2; mean=lambda X:np.fft.irfft(X,G)[0]
lh=kern(np.log(s)); u=lh[np.mod(Lm,G)]@p; C2=u@np.linalg.solve(Phi,u)/Jo
agrid=np.round(np.arange(0,1.0001,0.05),3); Ts=[2,5,10,30,60,120]; c=0.42
for mu in [0.05,0.1,0.2]:
    mub=2*mu*B/M
    E=np.zeros((len(agrid),len(Ts))); E1=np.zeros_like(E); Es=np.zeros_like(E); L1=np.zeros_like(E); L1s=np.zeros_like(E)
    for ia,a in enumerate(agrid):
        A=c*linalg.toeplitz(kern(s**(1-a))[:M]); b=c*kern(s**(-a))[np.mod(Lm,G)]@p/Pb
        lam,U=np.linalg.eigh(A); vinf=-U@((U.T@b)/lam); Jb=vinf@Phi@vinf/Jo; c0=U.T@(-w-vinf)
        for it,T in enumerate(Ts):
            nT=int(T*fs/B); ns=np.unique(np.round(np.geomspace(1,nT+1,500)).astype(int))-1
            ex=np.array([np.sum((ch.T@(vinf+U@((1-mub*lam)**n*c0)))**2)/Jo for n in ns])
            exs=np.array([Jb+mean(Sdd*np.exp(-2*mub*c*s**(1-a)*n))/Jo for n in ns])   # Szego approx
            E[ia,it]=ex[-1]; Es[ia,it]=exs[-1]
            L1[ia,it]=np.trapezoid(np.sqrt(1+ex),ns)/nT; L1s[ia,it]=np.trapezoid(np.sqrt(1+exs),ns)/nT
    print("mu=%.2f"%mu)
    print("  a* END exact:",{T:float(agrid[E[:,i].argmin()]) for i,T in enumerate(Ts)}," Szego-approx:",{T:float(agrid[Es[:,i].argmin()]) for i,T in enumerate(Ts)})
    print("  a* L1-avg[0,T] exact:",{T:float(agrid[L1[:,i].argmin()]) for i,T in enumerate(Ts)}," approx:",{T:float(agrid[L1s[:,i].argmin()]) for i,T in enumerate(Ts)})
    for a in [0,.25,.5,.75,1]:
        ia=int(np.argmin(abs(agrid-a)))
        print("   a=%.2f L1-avg ANR[0,T] dB (floor %.2f):"%(a,dB(Jo/Jd)),np.round(2*dB(L1[ia])+dB(Jo/Jd),2),"| approx",np.round(2*dB(L1s[ia])+dB(Jo/Jd),2))
    i30=Ts.index(30); print("   L1-avg over a at T=30:",dict(zip(agrid.tolist(),np.round(2*dB(L1[:,i30])+dB(Jo/Jd),2).tolist())))
