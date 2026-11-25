"""Isolated update-law ablations of frozen P5-early-cap.
The historical ablate.py remains unchanged (final.py imports its atomic writer).
NoSF retains the original spectral floor and regularizer, replacing the
spectral factor only. NoRecon changes only update errors, not trial judgment.
BgCoh uses the exact existing CohVSS update, with reconstructed background
error, a cloned foreground estimator at trial start, and coefficient-only
transfer; foreground estimator state continues without an early phase.
"""
import argparse
import copy
import itertools
import json
import math
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import final as F
import final2
import coef as C
from paired_ci import paired

HERE=Path(__file__).resolve().parent
OUT=HERE/"out"
TUNE=HERE/"TUNE_ablate.json"
VERSION=1
P=F.PROPOSED
VARIANTS=("noSF","noRecon","bgCoh")
SCEN=("stat","RS-near","RS-far","EN0")
SWITCH=("RS-near","RS-far")
PILOT=tuple(F.PILOT_CLASSES)
CLASSES=tuple(F.FINAL_CLASSES)
COH="Freq_FxNLMS_CohVSS"

def read(path):
    return json.loads(path.read_text())

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    F.ab.atomic_write(path,lambda h:json.dump(obj,h,indent=2))

def configure():
    final2.configure()

def grid(v):
    if v=="noSF":
        return [dict(e=e,l=l) for e,l in itertools.product((.25,.5,1.,2.),(.5,1.,2.))]
    if v=="noRecon":
        return [dict(e=e,l=1.) for e in (.25,.5,1.)]
    if v=="bgCoh":
        return [dict(k=k) for k in (2,4,8,16)]
    raise ValueError(v)

def fixed(v):
    return dict(e=1.,l=1.) if v!="bgCoh" else None

def cfg(setting):
    c=F.q5.config(early_ok=(4,.25))
    if "e" in setting:
        c["mu_e"]*=setting["e"]
        c["mu_l"]=F.q.mu_late(c["delta"])*setting["l"]
    return c

def direction(R,E,Srr,Pbar,s_eps,eps_reg,trunc=F.p.SFm.TRUNC,ret_y=False):
    # Same regularized spectrum as SF, including its original numerical floor.
    # Project the binwise normalized gradient onto causal FIR support.
    lam=np.maximum(Srr+s_eps+eps_reg*Pbar,1e-300)
    x=np.fft.irfft(np.conj(R)*E/lam,F.N)
    x[trunc:]=0.
    dv=x[:F.M]
    if ret_y:
        return dv,x,np.ones_like(R)
    return dv

class MeasuredFrame(F.q4.Frame4):
    def __call__(self,i,w,e_all,ys):
        # Retain all Frame4 state for detector and reconstructed trial rho.
        super().__call__(i,w,e_all,ys)
        return F.p.ML.frame_E(e_all,i*F.B,F.M)

class NoRecon(F.P5Cap):
    def block(self,i,c):
        # Both foreground and background direction calls receive the same
        # measured foreground error. P5's trial_frame and rho are untouched.
        old=F.p.SFm.sf_dir
        def measured(R,E,*args,**kwargs):
            return old(R,c.E,*args,**kwargs)
        F.p.SFm.sf_dir=measured
        try:
            return super().block(i,c)
        finally:
            F.p.SFm.sf_dir=old

def coh_update(algo,w,xf,ef,mu):
    # Populate the same chronological ring buffers that run_baseline_block
    # fills sample by sample, then invoke its actual last-sample update.
    n,m=algo.N,algo.M
    assert len(xf)==n and len(ef)==m and algo.count==0
    algo.x2[:n]=xf
    algo.x2[n:]=xf
    algo.e2[:m]=ef
    algo.e2[m:]=ef
    algo.x_head=n-1
    algo.e_head=m-1
    algo.count=algo.BlockSize-1
    result,_=algo.update(w.copy(),float(ef[-1]),None,[float(xf[-1])],mu)
    assert algo.count==0
    return result

class BgCoh(F.P5Cap):
    def __init__(self,c,dd,nb,X4,setting):
        super().__init__(c,dd,nb,X4)
        base=read(HERE/"TUNE.json")["chosen"][COH]
        par=dict(base["params"],c_impulse=dd["hSm_seq"][0][1],order_control=F.M)
        self.fg_algo=F.make_algo(COH,par)
        self.mu=float(base["mu"])
        self.k=int(setting["k"])
        self.xq=dd["xq"]
        self.until=0
        self.phases=[]
        assert len(dd["hSm_seq"])==1
        assert self.fg_algo.M==F.M and self.fg_algo.N==F.N and self.fg_algo.BlockSize==F.B

    def in_early(self,i):
        return False

    def mu_formula(self,i):
        return self.mu

    def block(self,i,c):
        end=(i+1)*F.B
        xf=self.xq[end:end+F.N]
        measured=np.zeros(F.M)
        ev=self.frame.e_all[max(0,end-F.M):end]
        measured[-len(ev):]=ev
        busy=self.bg is not None
        transfer=False
        wb=None
        if busy:
            k=i-self.bg["block"]
            wb=self.bg["w"]
            trial=self.trials[-1]
            tb=self.frame.trial_frame(wb)
            decided=None
            if k>=2:
                rho=self._rho(tb,c.w)
                eo=self.c["early_ok"]
                if not np.isfinite(rho) or rho>self.c["abort"]:
                    decided="abort"
                elif eo is not None and k>=eo[0] and rho<=eo[1]:
                    decided="confirmed"
                elif k>=self.c["K"]:
                    decided="confirmed" if rho<=self.c["theta"] else "rollback"
            if decided is None:
                rebuilt=np.zeros(F.M)
                rebuilt[-len(tb):]=tb
                self.bg["w"]=coh_update(self.bg["algo"],wb,xf,rebuilt,self.mu*self.k)
            else:
                trial.update(decide=int(i),rho=float(rho),ok=decided=="confirmed",status=decided)
                if decided=="confirmed":
                    transfer=True
                    self.restarts.append(trial["block"])
                self.bg=None
        # Keep the foreground's coherence estimator even on transfer;
        # on the transfer block update its statistics, not its coefficients.
        fg=coh_update(self.fg_algo,c.w,xf,measured,0. if transfer else self.mu)
        c.w_override=wb.copy() if transfer else fg
        if self.det is not None:
            rw=float(self.rw[i])
            z=self.frame.energy/rw if rw>0 else np.nan
            if self.det.update(i,z) and not busy:
                self.bg=dict(block=i,w=c.w.copy(),algo=copy.deepcopy(self.fg_algo))
                self.trials.append(dict(block=int(i),decide=None,rho=None,ok=None,status="pending"))
        self.tr["mu_app"].append(0. if transfer else self.mu)
        self.tr["mu_state"].append(self.mu)
        self.tr["restart_apply_block"].append(i if transfer else -1)
        self.tr["early"].append(0.)
        self.tr["transfer"].append(float(transfer))
        return 0.,None

def simulate(v,setting,built,X4,hook=None,nb=None,disable_detector=False):
    dd,Rall,st=built
    if v==P:
        return F.run_rule(P,dd,Rall,st,X4,nb=nb,hook=hook)
    nbr=min(len(dd["d"])//F.B,len(Rall)) if nb is None else nb
    c=cfg(setting)
    if v=="noSF":
        rule=F.P5Cap(c,dd,nbr,X4)
    elif v=="noRecon":
        rule=NoRecon(c,dd,nbr,X4)
    elif v=="bgCoh":
        rule=BgCoh(c,dd,nbr,X4,setting)
    else:
        raise ValueError(v)
    if disable_detector:
        rule.det=None
    rule.frame=MeasuredFrame(dd,rule) if v in ("noRecon","bgCoh") else F.q4.Frame4(dd,rule)
    old=F.p.SFm.sf_dir
    oldcache=F._SF_CACHE[0]
    F._SF_CACHE[0]=st if hasattr(st,"Fsf") and not F.os.environ.get("ANC_NO_SFCACHE") else None
    if v=="noSF":
        F.p.SFm.sf_dir=direction
    t=time.perf_counter()
    try:
        sim=F.simulate(dd,Rall,st,rule,nb=nb,error_frame=rule.frame,hook=hook)
    finally:
        F.p.SFm.sf_dir=old
        F._SF_CACHE[0]=oldcache
    sim["seconds"]=time.perf_counter()-t
    return rule,sim

def prior(folder,case,method):
    p=OUT/folder/"{}_{}_s{}.json".format(*case)
    matches=[r for r in read(p)["rows"] if r["method"]==method and r.get("variant","main")=="main"]
    if len(matches)!=1:
        raise ValueError((p,method,len(matches)))
    return matches[0]

def X4():
    return F.p.parse(["c5","--preset","final"]).X4

def build(case,final):
    c,s,i=case
    return F.build(c,s,i,F.LS_FINAL,final=final)

def measure(v,setting,case,built,post,x4,with_coef=False):
    dd=built[0]
    ws={}
    def hook(i,c,eb,mu):
        if i-F.CB in C.MAIN_KS:
            ws[i-F.CB]=c.w.copy()
    rule,sim=simulate(v,setting,built,x4,hook=hook if with_coef else None)
    pm=None
    if post is not None:
        _,ps=simulate(v,setting,post,x4)
        pm=dict(e=ps["e"],div=ps["div"],d=post[0]["d"][:len(ps["e"])])
    cl,sc,sd=case
    r=dict(method=v,cls=cl,scen=sc,seed=sd,kind="rule",setting=setting,
           **F.measures(sim,dd,pm),
           **F.trial_stats(rule,dict(dd,scored_blocks=sim["n_done"])))
    r["post_div"]=int(pm["div"]) if pm is not None else None
    coef=None
    if with_coef:
        coef=dict(method=v,cls=cl,scen=sc,seed=sd,setting=setting,div=r["div"],
                  J=C.frozen_anr(dd,{k:ws.get(k) for k in C.MAIN_KS}))
    return r,coef

def verify(a,b,case):
    for k in ("div","rec_blocks","rec_cens","n_trial","n_confirm","false_confirm"):
        if a.get(k)!=b.get(k):
            raise ValueError(f"Frozen identity mismatch {case} {k}: {a.get(k)} != {b.get(k)}")
    x,y=a.get("L1_whole"),b.get("L1_whole")
    if (x is None)!=(y is None) or (x is not None and not (math.isfinite(x) and math.isfinite(y) and abs(x-y)<=1e-7+1e-6*abs(y))):
        raise ValueError(f"Frozen L1 mismatch {case} {x} != {y}")
    return dict(cls=case[0],scen=case[1],seed=case[2],div=a["div"],
                rec_blocks=a.get("rec_blocks"),L1_whole=x,reference_L1_whole=y,
                abs_diff=None if x is None else abs(x-y))

def check():
    x4=X4()
    cases=(("G","stat",100),("a145","RS-far",101),("machinegun","RS-near",102),
           ("valve","EN0",103),("a120","RS-far",104))
    checks=[]
    for case in cases:
        b=build(case,True)
        post=build((case[0],"post:"+case[1],case[2]),True) if case[1] in SWITCH else None
        row,_=measure(P,{},case,b,post,x4)
        checks.append(verify(row,prior("final2_run",case,P),case))
    # Direction replacement and measured frame smoke; exact Coh baseline
    # equivalence with the detector disabled on a short run.
    b=build(("G","stat",10),False)
    for v in VARIANTS:
        _,sim=simulate(v,grid(v)[0],b,x4,nb=32,disable_detector=True)
        assert not sim["div"] and np.isfinite(sim["e"]).all(),v
    base=read(HERE/"TUNE.json")["chosen"][COH]
    ref=F.run_baseline(COH,base["mu"],base["params"],b[0],32*F.B)
    _,ours=simulate("bgCoh",dict(k=2),b,x4,nb=32,disable_detector=True)
    np.testing.assert_allclose(ours["e"],ref["e"],rtol=1e-9,atol=1e-10)
    np.testing.assert_allclose(ours["w"],ref["w"],rtol=1e-9,atol=1e-10)
    dump(OUT/"ablate_gate.json",dict(version=VERSION,checks=checks,coh_smoke_max_error=float(np.max(np.abs(ours["e"]-ref["e"])))))
    print("Frozen identity gate passed",len(checks),"/",len(cases),flush=True)

def name(folder,v,case):
    return OUT/folder/("{}_{}_{}_s{}.json".format(v,*case))

def cache(folder,v,case,settings,complete=True):
    f=name(folder,v,case)
    if not f.exists():
        return []
    obj=read(f)
    if obj.get("version")!=VERSION or obj.get("settings")!=settings:
        raise ValueError(f"Stale cache: {f}")
    rows=obj["rows"]
    if len({r["label"] for r in rows})!=len(rows):
        raise ValueError(f"Duplicate rows: {f}")
    for r in rows:
        assert (r["cls"],r["scen"],r["seed"],r["method"])==(*case,v)
        assert r["setting"]==settings[r["label"]]
    if complete and set(r["label"] for r in rows)!=set(settings):
        return []
    return rows

def store(folder,v,case,settings,rows):
    dump(name(folder,v,case),dict(version=VERSION,settings=settings,rows=rows))

def tune_settings(v):
    return {str(i):s for i,s in enumerate(grid(v))}

def tune_cases():
    return [(c,s,i) for c in PILOT for s in F.TUNE_SCEN for i in F.TUNE_SEEDS]

def tune_job(job):
    v,case=job
    settings=tune_settings(v)
    rows=cache("ablate_tune",v,case,settings,False)
    done={r["label"] for r in rows}
    if done==set(settings):
        return
    b=build(case,False)
    post=build((case[0],"post:"+case[1],case[2]),False) if case[1] in SWITCH else None
    x4=X4()
    for label,s in settings.items():
        if label in done:
            continue
        r,_=measure(v,s,case,b,post,x4)
        rows.append(dict(r,label=label))
        store("ablate_tune",v,case,settings,rows)
    print("tune",v,case,len(rows),flush=True)

def mean(rs,k):
    values=list(F.seed_table([r for r in rs if not r["div"]],k).values())
    return float(np.mean(values)) if values else None

def km(rs):
    if not rs or rs[0]["scen"] not in SWITCH:
        return "—"
    a,b=F.recovery_summary(rs)
    n=sum(not r["div"] and r.get("rec_cens") is not None for r in rs)
    return (f"{int(a)*F.B/F.FS:.3f}" if str(a).isdigit() else str(a))+f" ({b}/{n})"

def km_tune(rs):
    a,b=F.recovery_summary(rs)
    return dict(median_seconds=int(a)*F.B/F.FS if str(a).isdigit() else None,censored=b,
                n=sum(not r["div"] and r.get("rec_cens") is not None for r in rs))

def aggregate():
    result={}
    for v in VARIANTS:
        settings=tune_settings(v)
        allrows=defaultdict(list)
        for case in tune_cases():
            rows=cache("ablate_tune",v,case,settings)
            if len(rows)!=len(settings):
                raise ValueError(("Incomplete tuning",v,case))
            for r in rows:
                allrows[r["label"]].append(r)
        scores=[]
        for label,s in settings.items():
            rs=allrows[label]
            score=F.class_mean([r for r in rs if not r["div"]],"L1_whole")
            scores.append(dict(label=label,setting=s,div=sum(r["div"] for r in rs),
                               L1_whole=score,RS_far=km_tune([r for r in rs if r["scen"]=="RS-far"]),n=len(rs)))
        best=min(scores,key=lambda r:(r["div"],math.inf if r["L1_whole"] is None else r["L1_whole"]))
        result[v]=dict(chosen=best,grid=scores,fixed=fixed(v))
    dump(TUNE,dict(version=VERSION,variants=result,seeds=list(F.TUNE_SEEDS),scenarios=list(F.TUNE_SCEN),classes=list(PILOT)))
    print("Selected",[(v,r["chosen"]) for v,r in result.items()],flush=True)

def eval_settings(v):
    obj=read(TUNE)["variants"][v]
    s=dict(selected=obj["chosen"]["setting"])
    if obj["fixed"] is not None:
        s["fixed"]=obj["fixed"]
    return s

def eval_cases():
    return [(c,s,i) for c in CLASSES for s in SCEN for i in F.FINAL_SEEDS]

def eval_job(job):
    v,case=job
    settings=eval_settings(v)
    switch=case[1] in SWITCH
    rows=cache("ablate_run",v,case,settings,False)
    co=cache("ablate_coef",v,case,settings,False) if switch else []
    done={r["label"] for r in rows}
    cd={r["label"] for r in co}
    if done==set(settings) and (not switch or cd==set(settings)):
        return
    b=build(case,True)
    post=build((case[0],"post:"+case[1],case[2]),True) if switch else None
    x4=X4()
    for label,s in settings.items():
        if label in done and (not switch or label in cd):
            continue
        # Reuse the identical selected/fixed run if the parameter sets coincide.
        existing=next((r for r in rows if r["setting"]==s),None)
        existing_coef=next((r for r in co if r["setting"]==s),None)
        if existing is not None and (not switch or existing_coef is not None):
            r=dict(existing,label=label)
            cf=dict(existing_coef,label=label) if switch else None
        else:
            r,cf=measure(v,s,case,b,post,x4,with_coef=switch)
            r["label"]=label
            if cf is not None:
                cf["label"]=label
        if label not in done:
            rows.append(r)
            store("ablate_run",v,case,settings,rows)
        if switch and label not in cd:
            co.append(cf)
            store("ablate_coef",v,case,settings,co)
    print("eval",v,case,len(rows),flush=True)

def fmt(x):
    return "—" if x is None else f"{x:.3f}"

def med(rs,k):
    values=[r["J"].get(str(k)) for r in rs if not r["div"] and r["J"].get(str(k)) is not None]
    return float(np.median(values)) if values else None

def diff(a,b,metric,scale=1.):
    v=paired(a,b,metric)
    if v is None:
        return "—"
    m,l,h,n=v
    return f"{m*scale:+.3f} [{l*scale:+.3f}, {h*scale:+.3f}] ({n})"

def report():
    gate=read(OUT/"ablate_gate.json")
    assert gate["version"]==VERSION and len(gate["checks"])==5
    cfg=read(TUNE)["variants"]
    rows=defaultdict(list)
    coef=defaultdict(list)
    for case in eval_cases():
        c,s,i=case
        rows[P,"main",s].append(prior("final2_run",case,P))
        if s in SWITCH:
            coef[P,"main",s].append(prior("coef2_run",case,P))
        for v in VARIANTS:
            st=eval_settings(v)
            rs=cache("ablate_run",v,case,st)
            assert len(rs)==len(st),(v,case,"eval")
            for r in rs:
                rows[v,r["label"],s].append(r)
            if s in SWITCH:
                cs=cache("ablate_coef",v,case,st)
                assert len(cs)==len(st),(v,case,"coef")
                for r in cs:
                    coef[v,r["label"],s].append(r)
    lines=["# 更新則の要素除去と背景CohVSS試行","",
           "凍結提案法との一致ゲート 5/5。離散量厳密、L1は絶対差 ≤ 1e-7 + 1e-6×|既存値|。",
           "noSFは元のスペクトルの床・正則化を維持し因果制約付きビン正規化に置換。noReconは更新のみ実測誤差に置換し判定の再構成を維持。",
           "bgCohは既存CohVSS更新・共通muを用い、背景は再構成誤差とκ倍のmu、開始時に前景の統計を複製し、移行は係数のみ。gの正規化はCohVSS自体にないため適用せず早期段なし。",
           "選定 1710 条件・設定の組、最終評価 4000 行（同一設定の再利用を含む）、固定係数 2000 行。",
           "L1は負値が良い。KMは非発散かつ回復目標が定義できる実行。括弧は打切り数/解析数。対応差は提案−比較、seed bootstrap 2000回。",
           "回復の対応差は両方で回復を観測したペアのみ。固定係数は既存のC.frozen_anrと同じ定義。","",
           "## 設定選定（90試行/候補）","","| 変種 | 設定 | 発散 | L1 whole | RS-far KM秒 | 打切 | 採用 |","|---|---|---:|---:|---:|---:|---|"]
    for v in VARIANTS:
        for r in cfg[v]["grid"]:
            k=r["RS_far"]
            lines.append(f"| {v} | {r['setting']} | {r['div']}/{r['n']} | {fmt(r['L1_whole'])} | {fmt(k['median_seconds'])} | {k['censored']}/{k['n']} | {'○' if r==cfg[v]['chosen'] else ''} |")
    lines+=["","## 共通評価","","| 条件 | 手法 | 設定 | 発散 | KM秒 (打切/解析数) | L1 whole | L1 直後2秒 | J(.128s) | J(.512s) | 試行総数 | 移行総数 | 誤移行総数 |","|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in SCEN:
        for v,l in [(P,"main")]+[(v,l) for v in VARIANTS for l in eval_settings(v)]:
            rs=rows[v,l,s]
            cs=coef[v,l,s]
            assert len(rs)==200
            a=[sum(r.get(k,0) for r in rs) for k in ("n_trial","n_confirm","false_confirm")]
            lines.append(f"| {s} | {v} | {l} | {sum(r['div'] for r in rs)}/200 | {km(rs)} | {fmt(mean(rs,'L1_whole'))} | {fmt(mean(rs,'L1_ev'))} | {fmt(med(cs,16))} | {fmt(med(cs,64))} | {a[0]} | {a[1]} | {a[2]} |")
    lines+=["","## 提案法との対応差（95%区間）","","| 条件 | 変種 | 設定 | 入力 | L1 whole | L1直後2秒 | 回復(s) | J(.128s) | J(.512s) |","|---|---|---|---|---|---|---|---|---|"]
    for s in SCEN:
        for v in VARIANTS:
            for l in eval_settings(v):
                for cl in (None,*CLASSES):
                    filt=lambda rs:[r for r in rs if cl is None or r["cls"]==cl]
                    a=filt(rows[P,"main",s]); b=filt(rows[v,l,s])
                    ac=filt(coef[P,"main",s]); bc=filt(coef[v,l,s])
                    d=[diff(a,b,lambda r:r.get(k),scale) for k,scale in (("L1_whole",1.),("L1_ev",1.),("rec_blocks",F.B/F.FS))]
                    d.extend(diff(ac,bc,lambda r:r["J"].get(str(k))) for k in C.MAIN_KS)
                    lines.append(f"| {s} | {v} | {l} | {cl or '全体'} | "+" | ".join(d)+" |")
    lines+=["","## 観測結果（数値のみ）"]
    for v in VARIANTS:
        s="RS-far"; a=rows[P,"main",s]; b=rows[v,"selected",s]
        lines.append(f"- {v}選定版、RS-far：発散 {sum(r['div'] for r in b)}/200、KM {km(b)}、提案法 {km(a)}、L1 wholeの対応差 {diff(a,b,lambda r:r.get('L1_whole'))} dB。")
    F.ab.atomic_write(HERE/"RESULTS_ablate.md",lambda h:h.write("\n".join(lines)+"\n"))
    print("Complete: 4000 evaluation rows and 2000 frozen-coefficient rows",flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd",choices=("check","tune","tune-merge","eval","report"))
    p.add_argument("--variant",choices=VARIANTS)
    p.add_argument("--noise",choices=PILOT)
    p.add_argument("--scen",choices=SCEN)
    p.add_argument("--procs",type=int,default=4)
    a=p.parse_args()
    configure()
    if a.cmd=="check":
        check(); return
    if a.cmd=="tune-merge":
        aggregate(); return
    if a.cmd=="report":
        report(); return
    vs=(a.variant,) if a.variant else VARIANTS
    if a.cmd=="tune":
        jobs=[(v,c) for v in vs for c in tune_cases() if not a.noise or c[0]==a.noise]
        fn=tune_job
    else:
        jobs=[(v,c) for v in vs for c in eval_cases() if not a.scen or c[1]==a.scen]
        fn=eval_job
    with Pool(a.procs,initializer=configure) as pool:
        list(pool.imap_unordered(fn,jobs))

if __name__=="__main__":
    main()
