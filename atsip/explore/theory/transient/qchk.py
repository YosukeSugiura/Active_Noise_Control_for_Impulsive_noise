import sys; sys.argv=["x"]
import numpy as np, time
from vss2 import *
t=time.time()
data=get("G0"); Rall=ML.prepass(data)["Rall"]
class Rec(Ctrl2):
    def __init__(s,**k): super().__init__(**k); s.us=[]; s.ws=[]
    def step(s,i,info): s.ws.append(info["w"].copy()); return super().step(i,info)
    def post(s,i,g,mu): s.us.append(g.copy()); return super().post(i,g,mu)
ct=Rec(); r=run(data,Rall,kind="SF",mu=0.0,ctrl=ct,rec=1)
U=np.array(ct.us); W=np.array(ct.ws); q=np.array(ct.qs); ex=r["ex"]; mu=r["mu"]
np.savez("qchk.npz",U=U[:, :].astype(np.float32),W=W.astype(np.float32),q=q,ex=ex,mu=mu)
print("time",time.time()-t)
for n in [10,20,40,80,160,320,640,1280,2500,3700]:
    sl=slice(n-5,n+5); print("n %4d q %.4f mu %.4f ex %.4f  |u|^2 %.3e"%(n,q[sl].mean(),mu[sl].mean(),ex[sl].mean(),(U[sl]**2).sum(1).mean()))
# steady-state noise: Euclid vs Phi, autocorrelation
Uz=U[2500:]-U[2500:].mean(0)
e0=(Uz**2).sum(1).mean(); p0=np.einsum("ij,jk,ik->i",Uz,Phi,Uz).mean()
print("steady dir: Euclid %.3e Phi/Phi00 %.3e ratio(gamma_o) %.2f"%(e0,p0/Phi[0,0],e0/(p0/Phi[0,0])))
for nm,Pm in [("Euclid",None),("Phi",Phi)]:
    rk=[]
    for k in range(0,13):
        A=Uz[:len(Uz)-k]; Bm=Uz[k:]
        rk.append(np.mean(np.sum(A*(Bm if Pm is None else Bm@Pm),1)))
    rk=np.array(rk)/rk[0]; print(nm,"rho_k",np.round(rk,3),"tau %.2f"%(1+2*rk[1:].sum()))
