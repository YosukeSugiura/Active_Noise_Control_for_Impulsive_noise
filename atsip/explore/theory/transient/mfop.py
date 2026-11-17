"""Per-frame gain operators A_i (M x M): direction = A_i v for the coherent error of deviation v.
Compares B0 / SF / MF / MF with leak-matched cap on the largest-g frames: spectral radius, and Phi-norm of (I - eta A_i)."""
import numpy as np, sys
from simx import *
def lagwin(P):  # leak-smoothed power: periodogram convolved with |Dirichlet_M|^2  (lag window (1-|t|/M)+)
    c=np.fft.irfft(P,N); t=np.minimum(np.arange(N),N-np.arange(N)); c*=np.maximum(0,1-t/M); return np.maximum(np.fft.rfft(c).real,0)
def mfcap_dir(R,E,Lam,alpha=0.05):
    Lc=np.maximum(Lam,alpha*lagwin(np.abs(R)**2)*N/M)
    g=np.fft.irfft(np.conj(R)*E,N)[:M]; return np.fft.irfft(np.fft.rfft(g,N)/Lc,N)[:M]
if __name__=="__main__":
    ch=np.linalg.cholesky(Phi); chi=np.linalg.inv(ch)
    for tag in ["a120_s0","cms145_s0","G0"]:
        data=get(tag); Rall=ML.prepass(data)["Rall"]; r=data["r"]; g=IC.gain_seq(Rall); nb=1000
        shared=X.Shared(data["S2"]); st=[]
        for i in range(nb): s=shared.step(Rall[i],Rall[i]); st.append((s["Srr"].copy(),s["Pbar"],s["eps"]))
        cand=np.arange(300,nb); sel=cand[np.argsort(g[cand])[-8:]] if tag!="G0" else cand[::90][:8]
        rows=[]
        for i in sel:
            s0=i*B; Srr,Pbar,eps=st[i]
            n=np.arange(s0+B-M,s0+B)                      # error-frame samples
            Xf=np.stack([r[n-l] for l in range(M)],1)      # M x M, coherent error = Xf v
            Ef=np.fft.rfft(np.concatenate([np.zeros((N-M,M)),Xf]),axis=0)   # NB x M
            R=Rall[i]; out=[g[i]]
            for k in ["B0","SF","MF","MFcap"]:
                if k=="B0": A=np.fft.irfft(np.conj(R)[:,None]*Ef/(Srr+eps)[:,None],N,axis=0)[:M]
                elif k=="SF": A=np.stack([SFm.sf_dir(R,Ef[:,j],Srr,Pbar,eps,1e-2) for j in range(M)],1)
                elif k=="MF": A=np.stack([MFm.mf_dir(R,Ef[:,j],Srr+eps) for j in range(M)],1)
                else: A=np.stack([mfcap_dir(R,Ef[:,j],Srr+eps) for j in range(M)],1)
                ev=np.linalg.eigvals(A); 
                T=ch.T@(np.eye(M)-0.05*A)@chi.T   # Phi-norm of one update at mu=0.1 (eta=0.05)
                out+=[np.abs(ev).max(),np.linalg.norm(T,2)]
            rows.append(out)
        rows=np.array(rows)
        print("==",tag,"(cap c/alpha = 10)")
        print("   g_i | B0: specrad, ||I-0.05A||_Phi | SF: .. | MF: .. | MFcap: ..")
        for rw in rows: print("  %5.1f | %6.1f %6.2f | %6.1f %6.2f | %7.1f %7.2f | %6.1f %6.2f"%tuple(rw))
        print("  mean  | %6.1f %6.2f | %6.1f %6.2f | %7.1f %7.2f | %6.1f %6.2f"%tuple(rows.mean(0)[1:]))
