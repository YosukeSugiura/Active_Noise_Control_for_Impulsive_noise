import numpy as np
from scipy import linalg
from base import *
dB=lambda x:10*np.log10(x+1e-300)
n=2500
Pb=np.fft.irfft(S2,G)[0]
F=minphase(S2,G); f=F/np.sqrt(Pb)
q=np.fft.irfft(np.conj(Sw)*Pw/np.conj(F),G); qm=q[G-1:G-1-n:-1].copy()   # q(-1),q(-2),...
Jinf=(q[G//2:]**2).sum(); print("captured q_- energy",(qm**2).sum()/Jinf)
idx=np.arange(n)[:,None]+np.arange(n)[None,:]+1
def hank(spec): c=np.fft.irfft(spec,G); return c[idx]
def low(spec): c=np.fft.irfft(spec,G)[:n]; return np.tril(linalg.toeplitz(c))
lf=np.log(f)
H0=hank(lf); sv=np.linalg.svd(H0,compute_uv=False)
print("||H_logf|| = %.3f ; kappa^2 = %.3f (%.2f dB) ; C2/Jinf = %.4f ; HS norm^2 %.2f"%(sv[0],sv[0]**2,dB(sv[0]**2),((H0@qm)**2).sum()/Jinf,(sv**2).sum()))
ls=np.log(s); th=np.imag(lf)
print("sup|theta| (min-phase angle) %.3f rad ; osc(log s)/4 = %.3f ; std log s %.3f"%(np.abs(th).max(),(ls.max()-ls.min())/4,ls.std()))
for a in [.125,.25,.5,.75,1.0]:
    H=hank(f**(-a)); T=low(f**a)@H
    J=((T@qm)**2).sum()/Jinf
    b2=np.linalg.norm(T,2)**2; k2=np.linalg.norm(H,2)**2
    wq=np.abs(np.fft.rfft(np.concatenate([[0],qm]),G))**2   # |q_-|^2 spectrum
    m=lambda X:np.fft.irfft(X,G)[0]
    fa=f**(-a); c=m(fa*wq)/m(wq); bc=s.max()**a*m(np.abs(fa-c)**2*wq)/Jinf
    print("a=%.3f Jb/Jinf %.2f dB | worst-case beta^2=||f^a H||^2 %.2f dB | smax^a||H_{f^-a}||^2 %.2f dB | a^2 kappa^2 %.2f dB | smax^a min_c<|f^-a-c|^2>_q %.2f dB | ratio Jb/(a^2 C2) %.3f"%(a,dB(J),dB(b2),dB(s.max()**a*k2),dB(a*a*sv[0]**2),dB(bc),J/(a*a*((H0@qm)**2).sum()/Jinf)))
