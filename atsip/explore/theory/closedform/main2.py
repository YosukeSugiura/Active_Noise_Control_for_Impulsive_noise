import numpy as np, sys, time
from scipy import linalg
from base import *
M,N,B=512,1024,128
dB=lambda x:10*np.log10(x+1e-300)
w,g=wiener(M); Jo=g@g
Pb=xc(hS,hS,np.array([0]))[0]
Phi=linalg.toeplitz(xc(hS,hS,np.arange(M)))
LL=4*N; lags=np.arange(-LL,LL+1); p=xc(hS,g,lags)
AS=[0,.125,.25,.5,.75,1.0]
obs={0:-27.6,.25:-16.3,.5:-9.9,.75:-5.4,1.0:-1.1}
kern=lambda spec:np.fft.irfft(spec,G)
two=lambda k,la:k[np.mod(la,G)]
L=np.arange(M)[:,None]-lags[None,:]
# ---- quantities for bounds: S_re spectrum, coherence
Sre=np.conj(Sw)*np.fft.rfft(g,G)          # cross spectrum r->e_o (finite-M)
See=np.abs(np.fft.rfft(g,G))**2; Srr=S2
gam2=np.abs(Sre)**2/(Srr*See+1e-300)
mean=lambda X:np.fft.irfft(X,G)[0]
print("Jo check",mean(See),Jo," coherent part int|Sre|^2/Srr /Jo = %.3f"%(mean(np.abs(Sre)**2/Srr)/Jo))
ell=np.log(s)
def wmin(fun):  # min over c of int (fun-c)^2 gam2 See
    wgt=gam2*See; c=mean(fun*wgt)/mean(wgt); return mean((fun-c)**2*wgt)
print("== bin-normalised, IDEAL (infinite frame) model")
print("first-order: C2 bound (CS) /Jo = %.3f (%.2f dB)"%(wmin(ell)/Jo,dB(wmin(ell)/Jo)))
lh=kern(ell); u=two(lh,L)@p; C2=u@np.linalg.solve(Phi,u)
print("C2/Jo exact finite M = %.4f (%.2f dB)"%(C2/Jo,dB(C2/Jo)))
out={}
for a in AS:
    ka=kern(s**(-a)); A=linalg.toeplitz(kern(s**(1-a))[:M])
    b=two(ka,L)@p/Pb
    v=-np.linalg.solve(A,b); J=v@Phi@v
    bnd=s.max()**(2*a)*wmin(s**(-a))/Jo
    out[a]=J/Jo
    print("a=%.3f exact Jb/Jo %.2f dB | a^2C2 %.2f | obs %s | bound smax^2a*min_c<(s^-a-c)^2 g2> %.2f dB"%(a,dB(J/Jo),dB(a*a*C2/Jo),obs.get(a),dB(bnd)))
# ---- M->inf closed form: Jb = || f^a [ f^-a q_- ]_+ ||^2
F=minphase(S2,G)                      # |F|^2=S2 (=Srr for unit white x)
f=F/np.sqrt(Pb)
q=np.fft.irfft(np.conj(Sw)*Pw/np.conj(F),G)   # two-sided; index>G/2 = negative lags
qm=q.copy(); qm[:G//2]=0                 # anticausal part = q_-
Jinf=(qm**2).sum(); 
winf=np.fft.irfft(-(np.fft.rfft(q-qm))/F,G)   # causal Wiener (infinite)
print("== M=inf: J_inf/J_d %.2f dB ; J_o(512)/J_inf %.3f dB; |q_-|^2 %.3e"%(dB(Jinf/(hP@hP)),dB(Jo/Jinf),Jinf))
Qm=np.fft.rfft(qm)
for a in AS:
    y=np.fft.irfft(f**(-a)*Qm,G); y[G//2:]=0
    J=(np.abs(np.fft.irfft(f**a*np.fft.rfft(y),G))**2).sum()
    print("a=%.3f closed form (M=inf) Jb/Jinf %.2f dB ; rel Jo(512) %.2f"%(a,dB(J/Jinf),dB(J/Jo)))
# Hankel first-order
lf=np.fft.irfft(np.log(f),G); 
y=np.fft.irfft(np.log(f)*Qm,G); y[G//2:]=0
print("C2_inf=||[log f q_-]_+||^2 /Jinf = %.4f (%.2f dB); var-type bound: min_c sup|log s - c|^2/4?"%((y**2).sum()/Jinf,dB((y**2).sum()/Jinf)))
np.savez(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)),"inf.npz"),winf=winf[:8192],q=q)
