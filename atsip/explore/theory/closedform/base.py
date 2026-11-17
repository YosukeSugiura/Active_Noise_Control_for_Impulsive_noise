import os
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
import numpy as np
from scipy import linalg
ROOT="/home/claude/active_noise_control_for_impulsive_noise"
def pad(h,n):
    o=np.zeros(n); m=min(n,len(h)); o[:m]=h[:m]; return o
hP=pad(np.loadtxt(ROOT+"/impulse_response/primary_ir.dat",ndmin=1),512)
hS=pad(np.loadtxt(ROOT+"/impulse_response/secondary_ir.dat",ndmin=1),128)
G=1<<17
Sw=np.fft.rfft(hS,G); Pw=np.fft.rfft(hP,G)
S2=np.abs(Sw)**2; Pbar=S2.mean()  # approx mean over rfft bins
s=S2/np.fft.irfft(S2,G)[0]       # s=Srr/Pbar with exact mean
def xc(a,b,lags):  # E[a(n-tau) b(n)] for white x, a,b impulse responses: sum_k a(k) b(k+tau)
    full=np.correlate(b,a,"full")  # index i -> lag i-(len(a)-1)
    out=np.zeros(len(lags)); idx=lags+len(a)-1
    ok=(idx>=0)&(idx<len(full)); out[ok]=full[idx[ok]]; return out
def wiener(M):
    lags=np.arange(M)
    rrr=xc(hS,hS,lags); prd=xc(hS,hP,lags)
    w=linalg.solve_toeplitz(rrr,-prd)
    g=np.zeros(max(len(hP),len(hS)+M-1)); g[:len(hP)]+=hP; g[:len(hS)+M-1]+=np.convolve(hS,w)
    return w,g
def minphase(S,Gn):
    c=np.fft.irfft(0.5*np.log(S),Gn); c[1:Gn//2]*=2; c[Gn//2+1:]=0
    return np.exp(np.fft.rfft(c))
if __name__=="__main__":
    z=np.roots(hS[:np.max(np.nonzero(hS))+1]); az=np.abs(z)
    print("len hP nz",np.count_nonzero(hP),"hS nz",np.count_nonzero(hS))
    print("zeros outside:",(az>1).sum(),"sorted |z| near 1:",np.sort(az)[np.argsort(np.abs(np.sort(az)-1))[:8]])
    refl=np.where(az>1,1/az,az); print("rho=max reflected modulus",refl.max(), "20log10 rho dB/tap",20*np.log10(refl.max()), "top5",np.sort(refl)[-5:])
    print("s min %.3g max %.3g  mean log s %.3f  var log s %.3f"%(s.min(),s.max(),np.log(s).mean(),np.log(s).var()))
    for a in [0,.25,.5,.75,1]:
        t=s**(1-a); print("a",a,"s_min^(1-a)/mean(s^(1-a)) %.4g"%(t.min()/t.mean()))
    for M in [512]:
        w,g=wiener(M); print("M",M,"J_o/J_d dB %.2f"%(10*np.log10(g@g/(hP@hP))), "|w| tail: last 16 taps energy frac %.3g"%((w[-16:]**2).sum()/(w@w)))
