import numpy as np
from scipy import linalg
from base import *
dB=lambda x:10*np.log10(x+1e-300)
M,N,B,fs=512,1024,128,16000
z=np.load("/tmp/chk/fp_a145.npz")
kern=lambda spec:np.fft.irfft(spec,G)
w,g=wiener(M); Jo=g@g; Jd=hP@hP
Phi=linalg.toeplitz(xc(hS,hS,np.arange(M))); ch=np.linalg.cholesky(Phi)
Pb=Phi[0,0]
LL=4*N; lags=np.arange(-LL,LL+1); p=xc(hS,g,lags); Lm=np.arange(M)[:,None]-lags[None,:]
for a in ["0.0","0.25","0.5","0.75","1.0","ER"]:
    A=z["A_"+a]; v=z["v_"+a]
    ev=np.linalg.eigvals(A); re=np.sort(ev.real)
    asym=np.linalg.norm(A-A.T)/np.linalg.norm(A)
    msg="a=%s meas: mean eig %.4g min %.4g max %.4g asym %.3f | Jb(model Phi)/Jo %.2f dB"%(a,re.mean(),re[0],re[-1],asym,dB(v@Phi@v/Jo))
    if a!="ER":
        af=float(a); T=linalg.toeplitz(kern(s**(1-af))[:M]); et=np.linalg.eigvalsh(T)
        c=np.trace(A)/np.trace(T)
        msg+=" | ideal: scale c=%.3f, rel diff |A-cT|/|A| %.3f, lmin(T)*c %.4g ; bound c*smin^(1-a) %.4g"%(c,np.linalg.norm(A-c*T)/np.linalg.norm(A),c*et[0],c*s.min()**(1-af))
        vi=-np.linalg.solve(T,kern(s**(-af))[np.mod(Lm,G)]@p/Pb)
        msg+=" | v corr meas-vs-ideal %.3f"%(v@Phi@vi/np.sqrt((v@Phi@v)*(vi@Phi@vi)+1e-300))
    print(msg)
# tradeoff with measured A and v_inf, sample start v0=-w (model w)
Ts=[1,2,5,10,30,60]
for mu in [0.1]:
    mub=2*mu*B/M
    for a in ["0.0","0.25","0.5","0.75","1.0","ER"]:
        A=z["A_"+a]; vinf=z["v_"+a]
        lam,U=np.linalg.eig(A); c0=np.linalg.solve(U,(-w-vinf).astype(complex))
        row=[];rowa=[]
        for T in Ts:
            nT=int(T*fs/B); ns=np.unique(np.round(np.linspace(0,nT,300)).astype(int)); ex=[]
            for n in ns:
                v=(vinf+U@((1-mub*lam)**n*c0)).real; ex.append(np.sum((ch.T@v)**2)/Jo)
            row.append(dB(ex[-1])); rowa.append(dB((1+np.trapezoid(ex,ns)/nT)*Jo/Jd))
        print("meas A, mu=.1, a=%s end-excess dB"%a,np.round(row,1)," avgANR[0,T]",np.round(rowa,2))
