"""Frozen-probe measurement of the model coefficients per frame:
 rho1_i = <v, dir_c>_Phi/|v|^2_Phi, rho2_i = |dir_c|^2_Phi/|v|^2_Phi, n_i = |dir_o|^2_Phi/Jo
for B0 (a=1), W0.5, SF (eps 1e-2), MF.  dir_c: direction from the coherent error r*v, dir_o: from the optimum residual e_o."""
import numpy as np, sys
from simx import *
from multiprocessing import Pool
def dirs(kind,R,E,Srr,Pbar,eps):
    if kind=="B0": return np.fft.irfft(np.conj(R)*E/(Srr+eps),N)[:M]
    if kind=="W0.5": return np.fft.irfft(np.conj(R)*E/(np.sqrt(Srr*Pbar)+eps),N)[:M]
    if kind=="SF": return SFm.sf_dir(R,E,Srr,Pbar,eps,1e-2)
    if kind=="MF": return MFm.mf_dir(R,E,Srr+eps)
KINDS=["B0","W0.5","SF","MF"]
def unit(tag):
    data=get(tag); Rall=ML.prepass(data)["Rall"]; x,d,r,hS=data["x"],data["d"],data["r"],data["hS"]; L=len(d); nb=L//B
    eo=d+signal.lfilter(hS,[1.0],np.convolve(x,wM)[:L])
    rng=np.random.default_rng(1)
    probes={"init":-wM,"rand":rng.standard_normal(M)}
    yc={k:signal.lfilter(v,[1.0],r) for k,v in probes.items()}
    nv={k:v@Phi@v for k,v in probes.items()}
    shared=X.Shared(data["S2"]); out={(k,p,q):[] for k in KINDS for p in probes for q in ("r1","r2")}; no={k:[] for k in KINDS}; do={k:[] for k in KINDS}
    for i in range(nb):
        s0=i*B; Eo=ML.frame_E(eo,s0,M); s=shared.step(Rall[i],Eo); Srr,Pbar,eps=s["Srr"],s["Pbar"],s["eps"]
        for k in KINDS:
            g=dirs(k,Rall[i],Eo,Srr,Pbar,eps); do[k].append(g)
            for p,v in probes.items():
                gc=dirs(k,Rall[i],ML.frame_E(yc[p],s0,M),Srr,Pbar,eps); Pg=Phi@gc
                out[(k,p,"r1")].append(v@Pg/nv[p]); out[(k,p,"r2")].append(gc@Pg/nv[p])
    g=IC.gain_seq(Rall); res={"g":g}
    for k in KINDS:
        Dm=np.array(do[k][100:]); mean=Dm.mean(0); Dz=Dm-mean
        res[k+"|n1"]=np.einsum("ij,jk,ik->i",Dz,Phi,Dz)/Jo
        Lr=32; nseg=len(Dz)//Lr; S=Dz[:nseg*Lr].reshape(nseg,Lr,M).sum(1)
        res[k+"|nLR"]=np.einsum("ij,jk,ik->i",S,Phi,S)/Jo/Lr
        res[k+"|bias"]=np.array([mean@Phi@mean/Jo])
        for p in probes:
            res[k+"|"+p+"|r1"]=np.array(out[(k,p,"r1")]); res[k+"|"+p+"|r2"]=np.array(out[(k,p,"r2")])
    return tag,res
if __name__=="__main__":
    TAGS=["G0","cms145_s0","a120_s0","cms145_s1"]
    allr={}
    with Pool(2) as pl:
        for tag,res in pl.imap(unit,TAGS):
            allr.update({tag+"|"+k:v for k,v in res.items()})
            g=res["g"]; print("==",tag,"E[g] %.3f E[g2]/E[g] %.2f"%(g.mean(),(g**2).mean()/g.mean()))
            for k in KINDS:
                n1=res[k+"|n1"].mean(); nL=res[k+"|nLR"].mean()
                line="  %-5s noise n1 %.3f nLR %.3f (ratio %.2f)"%(k,n1,nL,nL/n1)
                for p in ("init","rand"):
                    r1=res[k+"|"+p+"|r1"][100:]; r2=res[k+"|"+p+"|r2"][100:]; gg=g[100:]
                    top=np.argsort(gg)[-len(gg)//100:]
                    line+=" | %s: r1 %.3f r2 %.3f r2/r1^2 %.2f mu_max=4r1/r2 %.2f corr(r1,g) %.2f top1%%g share of r2 %.2f"%(p,r1.mean(),r2.mean(),r2.mean()/r1.mean()**2,4*r1.mean()/r2.mean(),np.corrcoef(r1,gg)[0,1],r2[top].sum()/r2.sum())
                print(line,flush=True)
    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)),"probe.npz"),**allr)
