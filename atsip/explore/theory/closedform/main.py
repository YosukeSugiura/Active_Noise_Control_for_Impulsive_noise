import numpy as np, sys, time
from scipy import linalg
from base import *
M,N,B=512,1024,128
dB=lambda x:10*np.log10(x)
w,g=wiener(M); Jo=g@g
lagsM=np.arange(M); rrr=xc(hS,hS,np.arange(-4*N,4*N+1)); 
Phi=linalg.toeplitz(xc(hS,hS,lagsM))
# full cross-correlation p(tau)=E[r(n-tau)e_o(n)], tau in [-LL,LL]
LL=4*N; lags=np.arange(-LL,LL+1); p=xc(hS,g,lags)
print("check p on [0,M): max %.2e ; |p_-|^2 %.3e  |p_tail(>=M)|^2 %.3e  p(M) %.3e"%(np.abs(p[LL:LL+M]).max(),(p[:LL]**2).sum(),(p[LL+M:]**2).sum(),p[LL+M]))
AS=[0,.125,.25,.5,.75,1.0]
obs={0:-27.6,.25:-16.3,.5:-9.9,.75:-5.4,1.0:-1.1}; obsl={0:.0014,.25:.0073,.5:.037,.75:.18,1.0:.77}
# ---------- ideal (infinite frame, exact lags) model ----------
def kern(spec):  # two-sided sequence of a real even spectrum on G grid
    return np.fft.irfft(spec,G)
def two(k,lagarr): return k[np.mod(lagarr,G)]
res={}
print("== IDEAL model: A=T_M(s^(1-a)), b=P[k_a * p]")
for a in AS:
    ka=kern(s**(-a)); ta=kern(s**(1-a))
    A=linalg.toeplitz(ta[:M])
    # b(l)=sum_tau k(l-tau)p(tau)
    b=np.array([two(ka,l-lags)@p for l in range(M)])
    bm=np.array([two(ka,l-lags[:LL])@p[:LL] for l in range(M)])   # from non-causal part only
    v=-np.linalg.solve(A,b); vm=-np.linalg.solve(A,bm)
    Jb=v@Phi@v/Pbar_ if False else None
    ev=np.linalg.eigvalsh(A)
    res[a]=dict(v=v,A=A,ev=ev)
    J=v@Phi@v; Jm=vm@Phi@vm
    print("a=%.3f Jb/Jo %.2f dB (nc part only %.2f) obs %s | lmin/mean %.4g  s_min^(1-a)/mean %.4g obs %s | lmin %.4g smin^(1-a) %.4g"%(a,dB(J/Jo+1e-30),dB(Jm/Jo+1e-30),obs.get(a),ev[0]/ev.mean(),(s**(1-a)).min()/ev.mean(),obsl.get(a),ev[0],s.min()**(1-a)))
# ---------- first-order: C2 ----------
ell=np.log(s); lh=kern(ell)
u=np.array([two(lh,l-lags)@p for l in range(M)])
C2=u@np.linalg.solve(Phi,u)   # units: Phi=T(s*Pbar'), here Phi built from rrr = Pbar*s where Pbar=rrr[0]
Pb=xc(hS,hS,np.array([0]))[0]
# with Lam=Pbar s^a: A=T(s^(1-a)); b=P F^-1[S_re s^-a]/Pbar ; v=-A^-1 b ; a->0: A=Phi/Pbar, b=-a u/Pbar => v=a Phi^-1 u
print("C2/Jo = %.4f (%.2f dB): prediction a^2*C2/Jo:"%(C2/Jo,dB(C2/Jo)), {a:round(dB(a*a*C2/Jo),2) for a in AS if a>0})
np.savez(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)),"ideal.npz"),p=p,lags=lags,w=w,g=g,C2=C2,Jo=Jo)
