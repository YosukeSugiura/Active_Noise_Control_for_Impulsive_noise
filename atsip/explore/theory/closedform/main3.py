import numpy as np, sys, time
from scipy import linalg
from base import *
dB=lambda x:10*np.log10(x+1e-300)
Pb=xc(hS,hS,np.array([0]))[0]; Jd=hP@hP
kern=lambda spec:np.fft.irfft(spec,G)
two=lambda k,la:k[np.mod(la,G)]
# ================= trade-off (ideal mean model), M=512 ================
M,N,B,fs=512,1024,128,16000
w,g=wiener(M); Jo=g@g
Phi=linalg.toeplitz(xc(hS,hS,np.arange(M)))
LL=4*N; lags=np.arange(-LL,LL+1); p=xc(hS,g,lags); L=np.arange(M)[:,None]-lags[None,:]
ch=np.linalg.cholesky(Phi)
agrid=np.round(np.arange(0,1.0001,0.05),3)
Ts=[1,2,5,10,30,60,120]
for mu in [0.1,0.2]:
    mub=2*mu*B/M; sc=M/N
    end=np.zeros((len(agrid),len(Ts))); avg=np.zeros_like(end); Jb=[]
    for ia,a in enumerate(agrid):
        A=sc*linalg.toeplitz(kern(s**(1-a))[:M]); b=sc*two(kern(s**(-a)),L)@p/Pb
        lam,U=np.linalg.eigh(A); vinf=-U@((U.T@b)/lam); Jb.append(vinf@Phi@vinf/Jo)
        c0=U.T@(-w-vinf)
        for it,T in enumerate(Ts):
            nT=int(T*fs/B); ns=np.unique(np.round(np.linspace(0,nT,400)).astype(int))
            ex=[]
            for n in ns:
                v=vinf+U@((1-mub*lam)**n*c0); ex.append(np.sum((ch.T@v)**2)/Jo)
            ex=np.array(ex); end[ia,it]=ex[-1]; avg[ia,it]=np.trapezoid(ex,ns)/nT
    print("mu=%.2f  argmin_a of END excess at T:"%mu,{T:(float(agrid[end[:,i].argmin()]),round(dB(end[:,i].min()),1)) for i,T in enumerate(Ts)})
    print("        argmin_a of time-AVERAGED total power over [0,T]:",{T:(float(agrid[avg[:,i].argmin()]),round(dB((1+avg[:,i].min())*Jo/Jd),2)) for i,T in enumerate(Ts)})
    for a in [0,.25,.5,.75,1]:
        ia=int(np.argmin(abs(agrid-a)))
        print("   a=%.2f Jb %.1f | end excess dB:"%(a,dB(Jb[ia])),np.round(dB(end[ia]),1),"| avg ANR[0,T] dB:",np.round(dB((1+avg[ia])*Jo/Jd),2))
# two-term analytic approximation: slowest-mode model
