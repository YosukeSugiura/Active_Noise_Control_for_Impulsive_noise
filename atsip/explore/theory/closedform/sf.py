import numpy as np
from scipy import linalg
from base import *
dB=lambda x:10*np.log10(x+1e-300)
Jd=hP@hP
F=minphase(S2,G); q=np.fft.irfft(np.conj(Sw)*Pw/np.conj(F),G); qp=q.copy(); qp[G//2:]=0
winf=np.fft.irfft(-np.fft.rfft(qp)/F,G); Jinf=(q[G//2:]**2).sum()
hinv=np.fft.irfft(1/F,G)
def Jof(wv):
    g=np.zeros(len(hS)+len(wv)+len(hP)); g[:len(hP)]+=hP; c=np.convolve(hS,wv); g[:len(c)]+=c; return g@g,g
print("w_inf energy beyond tap M / total:",{M:float("%.3g"%((winf[M:G//2]**2).sum()/(winf[:G//2]**2).sum())) for M in [256,512,1024,2048]})
print("len of h=1/F: energy beyond lag L:",{L:float("%.3g"%((hinv[L:G//2]**2).sum()/(hinv[:G//2]**2).sum())) for L in [128,256,512,1024,2048]})
def impl(M,N,eps_rel=1e-6):
    w,g=wiener(M); Jo=g@g
    tau=np.arange(N)
    def ctil(rho):  # rho(lag) callable -> expected circular correlation /M
        c=np.zeros(N); c[:M+1]=rho(tau[:M+1])
        t=tau[M+1:]; c[M+1:]=((N-t)*rho(t)+(t-M)*rho(t-N))/M
        return c
    C=np.zeros((N,M+1))
    rr=lambda la:xc(hS,hS,np.asarray(la))
    big=np.arange(-N-M,N+1); rrb=rr(big); pb=xc(hS,g,np.arange(-N,N+1))
    for j in range(M): C[:,j]=ctil(lambda la:rrb[np.asarray(la)-j+N+M])
    C[:,M]=ctil(lambda la:pb[np.asarray(la)+N])
    Sn=np.abs(np.fft.rfft(hS,N))**2; Lam=Sn+eps_rel*Sn.mean()
    Fn=minphase(Lam,N)
    Y=np.fft.rfft(C,axis=0)
    qq=np.fft.irfft(Y/np.conj(Fn)[:,None],N,axis=0); qq[M:]=0
    gg=np.fft.irfft(np.fft.rfft(qq,axis=0)/Fn[:,None],N,axis=0)[:M]
    v=-np.linalg.solve(gg[:,:M],gg[:,M])
    Phi=linalg.toeplitz(rr(np.arange(M)))
    ev=np.linalg.eigvals(gg[:,:M]).real
    # bin-normalised a=1 on same framing for reference
    g1=np.fft.irfft(Y/Lam[:,None],N,axis=0)[:M]; v1=-np.linalg.solve(g1[:,:M],g1[:,M])
    return dB(v@Phi@v/Jo),dB(v1@Phi@v1/Jo),ev.min(),ev.max()
print("M | Jo(M)/Jd | (Jo(M)-Jinf)/Jinf | SF ideal bias = [J(trunc_M w_inf)-Jo(M)]/Jo(M) | leading ||y_o||^2/Jo | SF impl-model (N=2M) | binnorm a=1 impl | eig(A_SF) min,max")
for M in [128,192,256,384,512,768,1024,1536,2048]:
    w,g=wiener(M); Jo=g@g
    Jt,_=Jof(winf[:M])
    p=xc(hS,g,np.arange(0,M+G//4))   # lags 0..
    # y_o(l)=sum_{m>=M} h(m-l)p(m)
    yo=np.array([hinv[M-l:M-l+G//4]@p[M:M+G//4] for l in range(M)])
    out=impl(M,2*M) if M<=1024 else (np.nan,)*4
    print("%5d | %.2f | %.2f dB | %.2f dB | %.2f dB | %.2f dB | %.2f dB | %.3f %.3f"%(M,dB(Jo/Jd),dB((Jo-Jinf)/Jinf),dB((Jt-Jo)/Jo),dB(yo@yo/Jo),out[0],out[1],out[2],out[3]))
for N in [2048,4096]:
    print("M=512, N=%d impl-model SF bias %.2f dB"%(N,impl(512,N)[0]))
