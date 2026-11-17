"""Mean-square transient model (mean: exact M x M matrices; fluctuation: per-mode Szego recursion).
 state: m (mean deviation), F(w) (fluctuation excess density / Jo)
 theta = ('W', a, om)  Lam~ = (1-om) s + om s^a      |  ('SF',)
   u = A m + b ;  m <- m - eta g1 u
   F <- (1 - 2 eta g1 lam + rov m2 g2 eta^2 lam^2) F + eta^2 [ko g2 lam^2 So/Jo + (rov m2 g2 - g1^2) |u|^2_Phi/Jo * shape]
   E = |m|^2_Phi/Jo + int F
 greedy: eta* = P/Q, P = g1 (<m,u>_Phi/Jo + int lam F), Q = rov m2 g2 (|u|^2_Phi/Jo + int lam^2 F) + ko g2 int lam^2 So/Jo ; theta* = argmax P^2/Q
 g1(n), g2(n): start-up profile of the bin gain (EMA lag), -> 1.   eta = mu/2 (= 2 mu B/M)."""
import numpy as np, sys
from simx import *
G_=BS.G; s=BS.s; kern=lambda spec:np.fft.irfft(spec,G_)
LL=4*N; lags=np.arange(-LL,LL+1); p=BS.xc(BS.hS,gM,lags); Lm=np.arange(M)[:,None]-lags[None,:]; Pb=Phi[0,0]
dec=slice(None,None,32); sd=s[dec]; So=(np.abs(np.fft.rfft(gM,G_))**2)[dec]; Sdd=(np.abs(BS.Pw)**2)[dec]
mean=lambda X:float(np.mean(X))      # mean over the (uniform) frequency grid ~ (1/2pi) int
C0=0.5
def sf_v(eps_rel=1e-2,Nn=N):   # mean fixed point of SF in the frame model (reviewer's impl(), returns v)
    tau=np.arange(Nn)
    def ctil(rho):
        c=np.zeros(Nn); c[:M+1]=rho(tau[:M+1]); t=tau[M+1:]; c[M+1:]=((Nn-t)*rho(t)+(t-M)*rho(t-Nn))/M; return c
    C=np.zeros((Nn,M+1)); rr=lambda la:BS.xc(BS.hS,BS.hS,np.asarray(la))
    big=np.arange(-Nn-M,Nn+1); rrb=rr(big); pb=BS.xc(BS.hS,gM,np.arange(-Nn,Nn+1))
    for j in range(M): C[:,j]=ctil(lambda la:rrb[np.asarray(la)-j+Nn+M])
    C[:,M]=ctil(lambda la:pb[np.asarray(la)+Nn])
    Sn=np.abs(np.fft.rfft(BS.hS,Nn))**2; Lam=Sn+eps_rel*Sn.mean(); Fn=BS.minphase(Lam,Nn)
    Y=np.fft.rfft(C,axis=0); qq=np.fft.irfft(Y/np.conj(Fn)[:,None],Nn,axis=0); qq[M:]=0
    gg=np.fft.irfft(np.fft.rfft(qq,axis=0)/Fn[:,None],Nn,axis=0)[:M]
    return -np.linalg.solve(gg[:,:M],gg[:,M])
_cache={}
def op(theta):
    if theta in _cache: return _cache[theta]
    if theta[0]=="SF":
        if "vsf" not in _cache: _cache["vsf"]=sf_v()
        v=_cache["vsf"]; A=C0*np.eye(M); b=-C0*v; lam=C0*sd/(sd+1e-2)
    else:
        _,a,om=theta; Lt=(1-om)*s+om*s**a
        A=C0*linalg.toeplitz(kern(s/Lt)[:M]); b=C0*kern(1/Lt)[np.mod(Lm,G_)]@p/Pb; lam=C0*(s/Lt)[dec]
    _cache[theta]=(A,b,lam); return _cache[theta]
def bias_db(theta):
    A,b,_=op(theta); v=-np.linalg.solve(A,b); return 10*np.log10(v@Phi@v/Jo+1e-30)
def startup(nb,on=True):   # Gaussian-measured profile of bin gain mean / second moment (EMA lag), relative to steady state
    n=np.arange(nb)
    if not on: return np.ones(nb),np.ones(nb)
    g1=1+5.5*np.exp(-n/9.0); g2=1+30.0*np.exp(-n/7.0)
    return g1,g2
def simulate(nb,par,mu=None,theta=None,greedy=None,m0=None,F0=None,su=True,mu_cap=None,gseq=None,kinc=0.9,know_g=True):
    """par: dict(m2, ko, rov). Fixed schedule: mu, theta per block.  greedy: list of candidate thetas.
    gseq: actual frame-gain sequence g_n (conditional model); None -> ensemble moments E[g]=1, E[g^2]=m2 (with start-up profile).
    per mode:  rho_n = 1 - 2 eta G1 lam + rov eta^2 (G2 + kinc G1) lam^2 ;  noise eta^2 ko G1 lam^2 So/Jo."""
    m=-wM.copy() if m0 is None else m0.copy(); F=np.zeros_like(sd) if F0 is None else F0.copy()
    s1,s2=startup(nb,su); m2,ko,rov=par["m2"],par["ko"],par["rov"]
    E=np.zeros(nb); mus=np.zeros(nb); ths=[None]*nb
    for n in range(nb):
        E[n]=m@Phi@m/Jo+mean(F)
        if gseq is None: G1,G2=s1[n],m2*s2[n]
        else: G1,G2=gseq[n],gseq[n]**2
        k2=rov*(G2+kinc*G1)
        if greedy is not None:
            # the scheduler may know g_n (reference only) or only its moments
            if know_g or gseq is None: H1,H2=G1,k2
            else: H1,H2=1.0,rov*(m2+kinc)
            best=None
            for th in greedy:
                A,b,lam=op(th); u=A@m+b; Pu=Phi@u
                P=H1*(m@Pu/Jo+mean(lam*F)); Q=H2*(u@Pu/Jo+mean(lam*lam*F))+ko*H1*mean(lam*lam*So)/Jo
                val=P*P/Q if P>0 else -1
                if best is None or val>best[0]: best=(val,th,P/Q if P>0 else 0.0)
            th=best[1]; eta=best[2]
            if mu_cap is not None: eta=min(eta,mu_cap/2)
        else:
            th=theta[n] if isinstance(theta,list) else theta; eta=(mu[n] if np.ndim(mu) else mu)/2
        A,b,lam=op(th); u=A@m+b; uu=u@Phi@u/Jo
        shape=lam*lam*So; shape=shape/mean(shape)
        F=np.maximum(1-2*eta*G1*lam+k2*eta**2*lam**2,0)*F+eta**2*(ko*G1*lam**2*So/Jo+max(k2-G1**2,0)*uu*shape)
        m=m-eta*G1*u
        mus[n]=2*eta; ths[n]=th
        if not np.isfinite(E[n]) or E[n]>1e12: E[n:]=np.inf; break
    return dict(E=E,mu=mus,th=ths,m=m,F=F)
def xdb(E,rec=1):   # 'excess-dB' functional: mean of 10log10(1+ex): whole / first 5 s / last 10 %
    y=10*np.log10(1+np.asarray(E)); n5=int(5*FS/B/rec); n10=len(y)//10
    return np.array([y.mean(),y[:n5].mean(),y[-n10:].mean()])
PAR={"G":dict(m2=1.9,ko=None,rov=None,nLR=0.47),"a145":dict(m2=6.0,ko=None,rov=None,nLR=0.415),"a120":dict(m2=8.5,ko=None,rov=None,nLR=0.39)}
def setpar(rov):
    A,b,lam=op(("W",0.5,1.0)); base=mean(lam*lam*So)/Jo
    for k in PAR: PAR[k]["rov"]=rov; PAR[k]["ko"]=PAR[k]["nLR"]/base
    return PAR
def gload(tag):
    p=np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),"probe.npz")); return np.maximum(p[tag+"|SF|init|r1"],0)/C0
CLS={"G0":"G","G1":"G","cms145_s0":"a145","cms145_s1":"a145","a120_s0":"a120","a120_s1":"a120"}
if __name__=="__main__":
    print("bias dB: B0 %.2f W0.5 %.2f SF(eps 1e-2) %.2f"%(bias_db(("W",1.0,1.0)),bias_db(("W",0.5,1.0)),bias_db(("SF",))))
    meas=np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),"meas.npz")); nb=3750
    wr,mm=ramp(nb,4,3)
    CF={"SF0.05":(0.05,("SF",)),"SF0.1":(0.1,("SF",)),"SF0.2":(0.2,("SF",)),"SF0.4":(0.4,("SF",)),"SF+ramp":(0.05*ramp(nb,2,3)[1],("SF",)),
        "B0_0.1":(0.1,("W",1.0,1.0)),"W0.5_0.1":(0.1,("W",0.5,1.0)),"W0.5_0.2":(0.2,("W",0.5,1.0)),
        "GS":(0.05*mm,[("W",0.5,float(np.round(x,2))) for x in wr])}
    pts=[16,64,250,1000,3600]
    for rov in ([2.0,3.0,4.0] if len(sys.argv)<2 else [float(sys.argv[1])]):
        setpar(rov); err=[]; errE=[]
        print("=== rov",rov,{k:round(v["ko"],2) for k,v in PAR.items()})
        for tag in ["G0","cms145_s0","a120_s0"]:
            gs=gload(tag)
            for nm,(mu,th) in CF.items():
                r=simulate(nb,PAR[CLS[tag]],mu=mu,theta=th,gseq=gs,su=False)
                re_=simulate(nb,PAR[CLS[tag]],mu=mu,theta=th)
                me=meas[tag+"|"+nm]
                mo=10*np.log10(r["E"][[q-1 for q in pts]]); ms=10*np.log10([me[q//2-1] if q<3600 else me[-150:].mean() for q in pts])
                xm=xdb(r["E"]); xe=xdb(re_["E"]); xs=xdb(me,2)
                if np.isfinite(xm).all() and nm!="SF0.4": err.append(np.abs(xm-xs)); errE.append(np.abs(xe-xs))
                if len(sys.argv)>1: print("  %-9s %-9s cond-model ex dB @16/64/250/1000/end %s | meas %s | xdB whole/first5/last10: cond %s ens %s meas %s"%(tag,nm,np.round(mo,1),np.round(ms,1),np.round(xm,2),np.round(xe,2),np.round(xs,2)))
        print("  mean abs err of xdB (whole, first5, last10): conditional",np.round(np.mean(err,0),3)," ensemble",np.round(np.mean(errE,0),3))
