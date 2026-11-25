"""Dual-threshold normalized FxLMS, based on Akhtar & Mitsuhashi (JSV 2009).
Online rolling quantiles and normalization are evaluation adaptations, not
verified original-paper settings. No existing implementation is modified.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from multiprocessing import Pool
from pathlib import Path
import time
import numpy as np
import final as F
import final2
import coef as C
from fastbase import njit
from paired_ci import paired

NAME = "Th2_FxNLMS"
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
TUNE = HERE / "TUNE_thx.json"
VERSION = 1
WINDOW = 8000
UPDATE = 128
QUANTILES = ((1,99), (5,95), (10,90))
FACTORS = (.25,.5,1.,2.,4.,8.)
SWEEP = (.5,1.,2.,4.,8.,16.)
SWITCH = ("RS-near","RS-far")
SCEN = ("stat","RS-near","RS-far","EN0","ENm10")
PILOT = tuple(F.PILOT_CLASSES)
CLASSES = tuple(F.FINAL_CLASSES)
PROPOSED = F.PROPOSED

@njit(cache=True)
def quantile(a, q):
    # Exact NumPy linear-interpolation order statistic; no full sort.
    pos = (a.size-1)*q/100.
    k = int(math.ceil(pos))
    v = np.partition(a, k)
    f = pos-math.floor(pos)
    if f == 0.:
        return v[k]
    lo = np.max(v[:k])
    return lo+(v[k]-lo)*f

@njit(cache=True)
def _loop(x,d,xr,hrevs,lengths,sw,hm,M,B,mu,eps,pref,div_gain,window,interval,ql,qh):
    n_tot = len(d)
    L = hrevs.shape[1]
    Lm = len(hm)
    w = np.zeros(M)
    e = np.zeros(n_tot)
    yr = np.zeros(2*L)
    xc = np.zeros(2*Lm)
    rr = np.zeros(2*M)
    ih = ix = ir = seg = 0
    xl = el = -np.inf
    xh = eh = np.inf
    nx = ne = used = 0
    div = False
    n_done = n_tot
    fail_n = -1
    for n in range(n_tot):
        if seg+1 < len(sw) and n == sw[seg+1]:
            seg += 1
        # Only completed past samples; first block uses no clipping.
        if n > 0 and n % interval == 0:
            j = max(0,n-window)
            xl = quantile(x[j:n],ql)
            xh = quantile(x[j:n],qh)
            el = quantile(e[j:n],ql)
            eh = quantile(e[j:n],qh)
        i = n_tot-1-n
        y = np.dot(w,xr[i:i+M])
        yr[ih] = yr[ih+L] = y
        ih = (ih+1)%L
        lh = lengths[seg]
        en = d[n]+np.dot(yr[ih+L-lh:ih+L],hrevs[seg,:lh])
        e[n] = en
        bad = not math.isfinite(en)
        if not bad:
            xv = x[n]
            xclip = min(max(xv,xl),xh)
            eclip = min(max(en,el),eh)
            nx += int(xclip != xv)
            ne += int(eclip != en)
            used += 1
            # Clip raw reference, then apply the secondary path model.
            ix = (ix+Lm-1)%Lm
            xc[ix] = xc[ix+Lm] = xclip
            r = np.dot(xc[ix:ix+Lm],hm)
            ir = (ir+M-1)%M
            rr[ir] = rr[ir+M] = r
            rv = rr[ir:ir+M]
            step = (mu/(np.dot(rv,rv)+eps))*eclip
            for k in range(M):
                w[k] -= step*rv[k]
        if not bad and (n+1)%B == 0:
            pw = 0.
            for k in range(n+1-B,n+1):
                pw += e[k]*e[k]
            bad = not math.isfinite(pw) or pw/B > div_gain*pref
            if not bad:
                for k in range(M):
                    if not math.isfinite(w[k]):
                        bad = True
                        break
        if bad:
            div = True
            n_done = (n//B)*B
            fail_n = n
            break
    return e,w,div,n_done,fail_n,nx,ne,used

def run_baseline(F,mu,params,dd,n_tot=None):
    if len(dd["hSm_seq"]) != 1:
        raise ValueError("Th2 only supports fixed secondary-path model in frozen scenarios")
    d,x = dd["d"],dd["x"]
    n = len(d) if n_tot is None else min(n_tot,len(d))
    d = np.ascontiguousarray(d[:n],dtype=np.float64)
    x = np.ascontiguousarray(x[:n],dtype=np.float64)
    xr = np.concatenate([x[::-1],np.zeros(F.M-1)])
    plant = dd["hS_seq"]
    lengths = np.array([len(h) for _,h in plant],dtype=np.int64)
    hrevs = np.zeros((len(plant),int(lengths.max())))
    for i,(_,h) in enumerate(plant):
        hrevs[i,:lengths[i]] = h[::-1]
    sw = np.array([b*F.B for b,_ in plant],dtype=np.int64)
    hm = np.ascontiguousarray(dd["hSm_seq"][0][1],dtype=np.float64)
    pref = float(dd.get("primary_power",np.mean(dd["d"]**2)))
    lo,hi = params["percentiles"]
    t = time.perf_counter()
    e,w,div,done,fail,nx,ne,used = _loop(x,d,xr,hrevs,lengths,sw,hm,F.M,F.B,
        float(mu),float(params.get("eps",1e-3)),pref,float(F.DIV_GAIN),
        int(params["window"]),int(params["interval"]),float(lo),float(hi))
    failure = None
    if div:
        failure = dict(F.error_peak(e,done),sample=int(fail))
        e[done:] = np.nan
    return dict(e=e,w=w,div=bool(div),n_done=int(done),failure=failure,
        seconds=time.perf_counter()-t,clip_x=int(nx),clip_e=int(ne),clip_n=int(used))

def install(F):
    if getattr(F.run_baseline,"_thx",False):
        return
    original = F.run_baseline
    def run(name,mu,params,dd,n_tot=None):
        return run_baseline(F,mu,params,dd,n_tot) if name == NAME else original(name,mu,params,dd,n_tot)
    run._thx = True
    F.run_baseline = run

def configure():
    final2.configure()
    install(F)

def grid():
    mu0 = read(HERE/"TUNE.json")["chosen"]["FxNLMS"]["mu"]
    return [dict(key=f"x{f:g}_q{a}_{b}",mu=mu0*f,params=dict(percentiles=[a,b],
             window=WINDOW,interval=UPDATE,eps=1e-3)) for f in FACTORS for a,b in QUANTILES]

def read(p):
    return json.loads(p.read_text())

def dump(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    F.ab.atomic_write(p,lambda h: json.dump(obj,h,indent=2))

def path(folder,case):
    return OUT/folder/"{}_{}_s{}.json".format(*case)

def cached(folder,case,settings,nrows):
    p = path(folder,case)
    if not p.exists():
        return None
    v = read(p)
    if v.get("version") != VERSION or v.get("settings") != settings or len(v.get("rows",[])) != nrows:
        return None
    if any((r["cls"],r["scen"],r["seed"]) != case for r in v["rows"]):
        raise ValueError(f"Wrong case identity: {p}")
    return v["rows"]

def require(folder,case,settings,nrows):
    rows = cached(folder,case,settings,nrows)
    if rows is None:
        raise ValueError(f"Missing/stale {folder} {case}")
    return rows

def store(folder,case,settings,rows):
    dump(path(folder,case),dict(version=VERSION,settings=settings,rows=rows))

def tag(case,**kw):
    c,s,i = case
    return dict(method=NAME,cls=c,scen=s,seed=i,**kw)

def simulate(setting,dd,n_tot=None):
    return F.run_baseline(NAME,setting["mu"],setting["params"],dd,n_tot)

def tune_job(case):
    settings = grid()
    if cached("thx_tune",case,settings,len(settings)) is not None:
        return
    dd,_,_ = F.build(*case,F.LS_FINAL)
    rows = [dict(**tag(case,key=s["key"]),**F.measures(simulate(s,dd),dd)) for s in settings]
    store("thx_tune",case,settings,rows)
    print("tune",case,len(rows),flush=True)

def tune_cases(classes=PILOT):
    return [(c,s,i) for c in classes for s in F.TUNE_SCEN for i in F.TUNE_SEEDS]

def choose(cases):
    settings = grid()
    rows = [require("thx_tune",case,settings,len(settings)) for case in cases]
    scores = []
    for i,s in enumerate(settings):
        rs = [r[i] for r in rows]
        assert all(r["key"] == s["key"] for r in rs)
        ok = [r for r in rs if not r["div"]]
        scores.append(dict(**s,div=sum(r["div"] for r in rs),
                           L1_whole=F.class_mean(ok,"L1_whole")))
    return min(scores,key=lambda r:(r["div"],float("inf") if r["L1_whole"] is None else r["L1_whole"])),scores

def aggregate():
    best,scores = choose(tune_cases())
    per = {c:choose(tune_cases((c,)))[0] for c in PILOT}
    dump(TUNE,dict(version=VERSION,chosen={NAME:best},per_class={NAME:per},grid=scores,
         classes=PILOT,seeds=F.TUNE_SEEDS,scen=F.TUNE_SCEN,
         note="Normalized online variant; 8000 past samples, 128 update; exact linear quantiles. Original window/update unverified."))
    print("Selected",best,per,flush=True)

def selected(c=None):
    cfg = read(TUNE)
    s = cfg["per_class"][NAME].get(c,cfg["chosen"][NAME]) if c else cfg["chosen"][NAME]
    return {k:s[k] for k in ("key","mu","params")}

def prior(folder,case,method,variant="main"):
    rs = read(path(folder,case))["rows"]
    matches = [r for r in rs if r["method"] == method and r.get("variant","main")==variant]
    if len(matches)!=1:
        raise ValueError(f"Invalid reference {folder} {case} {method} {variant}")
    return matches[0]

def measure(case,setting,dd,post,variant,factor=None):
    sim = simulate(setting,dd)
    ps = simulate(setting,post) if post is not None and not sim["div"] else None
    r = dict(**tag(case,kind="base",variant=variant,key=setting["key"],mu=setting["mu"]),
             **F.measures(sim,dd,ps))
    r.update(clip_x=sim["clip_x"],clip_e=sim["clip_e"],clip_n=sim["clip_n"],
             post_div=int(ps["div"]) if ps is not None else None)
    if post is not None:
        r["sweep_rec_limit"] = max(0,len(dd["d"])//F.B-dd["changes"][0]-F.BPS+1)
        if not sim["div"]:
            anr = F.p.O.anr_curve(sim["e"],dd["d"][:len(sim["e"])])
            ts = (63+64*np.arange(len(anr)))/F.FS
            t = dd["changes"][0]/F.BPS
            r["L1_pre5"] = float(anr[(ts>=t-5)&(ts<t)].mean())
    if factor is not None:
        r["factor"]=factor
    return r,sim

def build(case,need_post=True):
    c,s,i=case
    dd,_,_=F.build(c,s,i,F.LS_FINAL,final=True)
    post=F.build(c,"post:"+s,i,F.LS_FINAL,final=True)[0] if need_post and s in SWITCH else None
    return dd,post

def eval_job(case):
    setting=selected()
    rs=cached("thx_run",case,setting,1)
    cs=cached("thx_coef",case,setting,1) if case[1] in SWITCH else None
    if rs is not None and (case[1] not in SWITCH or cs is not None):
        return
    dd,post=build(case,rs is None)
    if rs is None:
        r,sim=measure(case,setting,dd,post,"main")
        # Compare archived metrics and rerun FxNLMS for same-environment waveform check.
        old=prior("final2_run",case,"FxNLMS")
        base=read(HERE/"TUNE.json")["chosen"]["FxNLMS"]
        b=F.run_baseline("FxNLMS",base["mu"],base["params"],dd)
        r["same_as_fx_metrics"]=int(not r["div"] and not old["div"] and
            all(r.get(k)==old.get(k) for k in ("L1_whole","rec_blocks","L1_last10","L1_ev")))
        r["same_as_fx_waveform"]=int(not sim["div"] and not b["div"] and np.array_equal(sim["e"],b["e"]))
        r["same_as_fx_waveform_tol"]=int(not sim["div"] and not b["div"] and
            np.allclose(sim["e"],b["e"],rtol=1e-10,atol=1e-10))
        store("thx_run",case,setting,[r])
        rs=[r]
    if case[1] in SWITCH and cs is None:
        ws={}
        for k in C.MAIN_KS:
            prefix=simulate(setting,dd,(C.CB+k)*F.B)
            ws[k]=None if prefix["div"] else prefix["w"]
        row=dict(**tag(case,variant="main",kind="base"),div=rs[0]["div"],J=C.frozen_anr(dd,ws))
        store("thx_coef",case,setting,[row])
    print("eval",case,flush=True)

def class_job(case):
    setting=selected(case[0])
    if cached("thx_class_run",case,setting,1) is not None:
        return
    dd,post=build(case)
    r,_=measure(case,setting,dd,post,"class")
    store("thx_class_run",case,setting,[r])
    print("class",case,flush=True)

def sweep_settings():
    s=selected()
    return [dict(s,mu=s["mu"]*factor,factor=factor) for factor in SWEEP]

def sweep_job(case):
    settings=sweep_settings()
    p=path("thx_sweep_run",case)
    rows=[]
    if p.exists():
        obj=read(p)
        if obj.get("version")==VERSION and obj.get("settings")==settings:
            rows=obj["rows"]
    done={r["factor"] for r in rows}
    if done==set(SWEEP):
        return
    dd,post=build(case)
    for s in settings:
        if s["factor"] in done:
            continue
        r,_=measure(case,s,dd,post,"sweep",s["factor"])
        rows.append(r)
        store("thx_sweep_run",case,settings,rows)
    print("sweep",case,flush=True)

def cases(classes=CLASSES,scenarios=SCEN):
    return [(c,s,i) for c in classes for s in scenarios for i in F.FINAL_SEEDS]

def km(rows,include):
    counts=Counter()
    missing=0
    lim=max(0,30*F.FS//F.B-F.CB-F.BPS+1)
    for r in rows:
        if r["div"]:
            if include:
                counts[(r.get("sweep_rec_limit",lim),1)]+=1
        elif r.get("rec_cens") is None:
            missing+=1
        else:
            c=int(r["rec_cens"])
            counts[(r["rec_limit"] if c else r["rec_blocks"],c)]+=1
    risk=n=sum(counts.values())
    surv=1.
    med=None
    for t in sorted({t for t,c in counts}):
        events,cens=counts[t,0],counts[t,1]
        surv *= (risk-events)/risk
        if surv<=.5+1e-12:
            med=t*F.B/F.FS
            break
        risk-=events+cens
    return med,n,sum(v for (t,c),v in counts.items() if c),missing

def fmt(x,nd=3):
    return "—" if x is None else f"{x:.{nd}f}"

def kmtext(rs,inc=False):
    v,n,c,m=km(rs,inc)
    return f"{fmt(v) if v is not None else ('打切' if n else '—')} ({c}/{n}; 目標なし {m})"

def mean(rs,key):
    vals=list(F.seed_table([r for r in rs if not r["div"]],key).values())
    return float(np.mean(vals)) if vals else None

def diff(left,right,metric,scale=1.):
    v=paired(left,right,metric)
    if v is None:
        return "—"
    a,l,h,n=v
    return f"{a*scale:+.3f} [{l*scale:+.3f}, {h*scale:+.3f}] ({n})"

def check_one(a,b):
    for k in ("div","rec_blocks","rec_cens","rec_limit","clip_x","clip_e","clip_n","post_div","sweep_rec_limit","transition_count","trial_count"):
        if a.get(k)!=b.get(k):
            raise ValueError(f"1x mismatch {a['cls']} {a['scen']} {a['seed']} {k}")
    for k in ("L1_whole","L1_last10","L1_ev","L1_pre","L1_pre5","L2_last5","rec_target"):
        x,y=a.get(k),b.get(k)
        if (x is None)!=(y is None) or (x is not None and not (math.isfinite(x) and math.isfinite(y) and abs(x-y) <= 1e-7 + 1e-6*abs(x))):
            raise ValueError(f"1x mismatch {a['cls']} {a['scen']} {a['seed']} {k}: {x} != {y}")

def report():
    allcases=cases()
    switchcases=cases(CLASSES,SWITCH)
    classcases=cases(CLASSES,("stat","RS-far"))
    er={case:require("thx_run",case,selected(),1)[0] for case in allcases}
    cr={case:require("thx_class_run",case,selected(case[0]),1)[0] for case in classcases}
    sr={case:require("thx_sweep_run",case,sweep_settings(),len(SWEEP)) for case in switchcases}
    co={case:require("thx_coef",case,selected(),1)[0] for case in switchcases}
    for case,rs in sr.items():
        assert sorted(r["factor"] for r in rs)==sorted(SWEEP)
        check_one(er[case],next(r for r in rs if r["factor"]==1.))
    prop={case:prior("final2_run",case,PROPOSED) for case in allcases}
    pc={case:prior("coef2_run",case,PROPOSED) for case in switchcases}
    cfg=read(TUNE)
    lines=["# Th2-FxNLMS 追加評価","",
      "Akhtar & Mitsuhashi (2009) に基づく参照・誤差両信号のクリップを正規化 FxLMS に適用。",
      "原論文の窓長・更新間隔は未確認。過去8000標本、128標本更新、初回128標本は無制限。",
      "線形補間パーセンタイルを部分選択で厳密計算（近似なし）。制御出力は未クリップ参照。",
      "a120 の入力別設定は共通設定を使用。L1は負値が良い。回復目標なしはKMから除外。",
      "発散含むKMでは発散試行を共通観測終了時点で右打切り。発散を独立打切りと解釈しない。",
      f"選定 {len(tune_cases())*len(grid())}、eval {len(er)}、coef {len(co)} 条件、class {len(cr)}、sweep {sum(map(len,sr.values()))}。1倍一致 {len(sr)}/{len(sr)}（eval基準の絶対差 ≤ 1e-7 + 1e-6×|eval|、回復・発散・クリップ標本数は厳密。係数固定はsweepに未保存、移行・試行回数は未記録）。",
      "", "## 選定", "", "| 設定対象 | mu | パーセンタイル | 発散 | L1 whole |","|---|---:|---|---:|---:|"]
    for c,s in [("共通",cfg["chosen"][NAME]),*cfg["per_class"][NAME].items()]:
        lines.append(f"| {c} | {s['mu']:g} | {s['params']['percentiles']} | {s['div']} | {fmt(s['L1_whole'])} |")
    lines+=["","## 共通・入力別評価","","| 条件 | 手法 | 発散 | KM除く (s) | L1 whole | L1 last10 | L1 直後2秒 | 提案−比較 L1 whole [95%] |","|---|---|---:|---|---:|---:|---:|---|"]
    for s in SCEN:
        pp=[r for case,r in prop.items() if case[1]==s]
        for label,source in [(PROPOSED,prop),(NAME,er),("Th2 class",cr)]:
            rs=[r for case,r in source.items() if case[1]==s]
            if not rs:
                continue
            lines.append(f"| {s} | {label} | {sum(r['div'] for r in rs)}/{len(rs)} | {kmtext(rs) if s in SWITCH else '—'} | {fmt(mean(rs,'L1_whole'))} | {fmt(mean(rs,'L1_last10'))} | {fmt(mean(rs,'L1_ev'))} | {diff(pp,rs,lambda r:r.get('L1_whole')) if label!=PROPOSED else '—'} |")
    lines+=["","## クリップ割合","","各試行の全標本（30秒、480000標本）を分母とする。平均は試行間の算術平均、最大は試行間最大。初回ブロックを含む。",
             "| 条件 | 参照平均 (%) | 参照最大 (%) | 誤差平均 (%) | 誤差最大 (%) |",
             "|---|---:|---:|---:|---:|"]
    for s in SCEN:
        rs=[r for case,r in er.items() if case[1]==s]
        n=30*F.FS
        assert all(0 <= r[k] <= r["clip_n"] <= n for r in rs for k in ("clip_x","clip_e"))
        vals=[]
        for k in ("clip_x","clip_e"):
            rates=[100*r[k]/n for r in rs]
            vals.extend((float(np.mean(rates)),max(rates)))
        lines.append(f"| {s} | "+" | ".join(fmt(v) for v in vals)+" |")
    lines+=["","## FxNLMSとのL1 whole一致数","","両手法のL1 wholeが有限で、絶対差が1e-6 dB以内の試行数。発散・欠測は一致に含めない。",
             "| 条件 | L1 whole 一致 | 比較可能 | 既存評価指標厳密一致 | 波形厳密一致 | 波形許容誤差1e-10 |",
             "|---|---:|---:|---:|---:|---:|"]
    for s in SCEN:
        rs=[(case,r) for case,r in er.items() if case[1]==s]
        count=valid=0
        for case,r in rs:
            old=prior("final2_run",case,"FxNLMS")
            x,y=r.get("L1_whole"),old.get("L1_whole")
            if not r["div"] and not old["div"] and x is not None and y is not None and math.isfinite(x) and math.isfinite(y):
                valid+=1
                count+=int(abs(x-y)<=1e-6)
        vals=[sum(r[k] for _,r in rs) for k in ("same_as_fx_metrics","same_as_fx_waveform","same_as_fx_waveform_tol")]
        lines.append(f"| {s} | {count}/{len(rs)} | {valid}/{len(rs)} | {vals[0]}/{len(rs)} | {vals[1]}/{len(rs)} | {vals[2]}/{len(rs)} |")
    lines+=["","## 係数固定および対応差（seed bootstrap 2000回）","","提案−比較。回復は両方で回復を観測した対のみ。固定係数は C.frozen_anr と同定義。","| 条件 | 入力 | 手法 | 指標 | 提案 | 比較 | 差 [95%] |","|---|---|---|---|---:|---:|---|"]
    for s in SCEN:
        for c in (None,*CLASSES):
            for label,source in [(NAME,er),("Th2 class",cr)]:
                rs=[r for case,r in source.items() if case[1]==s and (c is None or case[0]==c)]
                pp=[r for case,r in prop.items() if case[1]==s and (c is None or case[0]==c)]
                if not rs:
                    continue
                keys=("L1_whole","L1_last10")+ (("L1_ev",) if s!="stat" else ())
                for k in keys:
                    lines.append(f"| {s} | {c or '全体'} | {label} | {k} | {fmt(mean(pp,k))} | {fmt(mean(rs,k))} | {diff(pp,rs,lambda r:r.get(k))} |")
                if s in SWITCH:
                    lines.append(f"| {s} | {c or '全体'} | {label} | 回復(s) | {kmtext(pp)} | {kmtext(rs)} | {diff(pp,rs,lambda r:r.get('rec_blocks'),F.B/F.FS)} |")
            if s in SWITCH:
                pp=[r for case,r in pc.items() if case[1]==s and (c is None or case[0]==c)]
                rs=[r for case,r in co.items() if case[1]==s and (c is None or case[0]==c)]
                for k in C.MAIN_KS:
                    metric=lambda r:r["J"].get(str(k))
                    def median(rows):
                        v=[metric(r) for r in rows if not r["div"] and metric(r) is not None]
                        return fmt(float(np.median(v)) if v else None)
                    lines.append(f"| {s} | {c or '全体'} | {NAME} | 固定係数 {k*F.B/F.FS:.3f}s | {median(pp)} | {median(rs)} | {diff(pp,rs,metric)} |")
    lines+=["","## ステップ掃引","","| 条件 | 手法 | 倍率 | 発散 | KM含む (s) | KM除く (s) | L1 whole | L1 pre5 | 提案−比較 L1 [95%] | 提案−比較回復(s) [95%] |","|---|---|---:|---:|---|---|---:|---:|---|---|"]
    groups=[]
    for s in SWITCH:
        pp=[r for case,r in prop.items() if case[1]==s]
        for factor in (None,*SWEEP):
            rs=pp if factor is None else [r for case,v in sr.items() if case[1]==s for r in v if r["factor"]==factor]
            label=PROPOSED if factor is None else NAME
            groups.append((s,label,factor,rs))
            lines.append(f"| {s} | {label} | {factor if factor is not None else '—'} | {sum(r['div'] for r in rs)}/{len(rs)} | {kmtext(rs,True)} | {kmtext(rs)} | {fmt(mean(rs,'L1_whole'))} | {fmt(mean(rs,'L1_pre5'))} | {diff(pp,rs,lambda r:r.get('L1_whole')) if factor is not None else '—'} | {diff(pp,rs,lambda r:r.get('rec_blocks'),F.B/F.FS) if factor is not None else '—'} |")
    lines+=["","## 騒音別発散数","","| 条件 | 手法 | 倍率 | "+" | ".join(CLASSES)+" |","|---|---|---:|"+"---:|"*len(CLASSES)]
    for s,label,factor,rs in groups:
        ct=Counter(r["cls"] for r in rs if r["div"])
        lines.append(f"| {s} | {label} | {factor if factor is not None else '—'} | "+" | ".join(str(ct[c]) for c in CLASSES)+" |")
    F.ab.atomic_write(HERE/"RESULTS_thx.md",lambda h:h.write("\n".join(lines)+"\n"))
    print(f"Report complete; 1x checked {len(sr)} cases",flush=True)

def selfcheck():
    rng=np.random.default_rng(7)
    for a in (rng.normal(size=8000),np.zeros(128),np.arange(128,dtype=float)):
        for q in (1,5,10,90,95,99):
            assert np.isclose(quantile(a,q),np.percentile(a,q),rtol=1e-12,atol=1e-12)
    # No clipping during first block, identity secondary path, simple finite loop.
    x=np.ascontiguousarray(rng.normal(size=256))
    d=x.copy()
    M=8; B=128
    r=_loop(x,d,np.concatenate([x[::-1],np.zeros(M-1)]),np.ones((1,1)),np.ones(1,dtype=np.int64),
          np.zeros(1,dtype=np.int64),np.ones(1),M,B,.01,1e-3,1.,1e4,8000,128,1.,99.)
    assert not r[2] and r[3]==len(x) and r[-1]==len(x)
    print("Quantile and loop smoke checks passed",flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("cmd",choices=("tune","eval","class","sweep","report","check"))
    p.add_argument("--noise",choices=CLASSES)
    p.add_argument("--group",type=int,choices=(0,1))
    p.add_argument("--scen",choices=SCEN)
    p.add_argument("--aggregate",action="store_true")
    p.add_argument("--procs",type=int,default=4)
    a=p.parse_args()
    configure()
    if a.cmd=="check":
        selfcheck()
        return
    if a.cmd=="report":
        report()
        return
    if a.cmd=="tune" and a.aggregate:
        aggregate()
        return
    if a.cmd=="tune":
        selfcheck()
        js=tune_cases((a.noise,) if a.noise else PILOT)
        fn=tune_job
    else:
        assert (HERE/"EVAL_FROZEN.json").is_file()
        classes=CLASSES if a.group is None else CLASSES[5*a.group:5*(a.group+1)]
        if a.noise:
            classes=tuple(c for c in classes if c==a.noise)
        scenarios=SCEN if a.cmd=="eval" else ("stat","RS-far") if a.cmd=="class" else SWITCH
        if a.scen:
            scenarios=tuple(s for s in scenarios if s==a.scen)
        js=cases(classes,scenarios)
        fn={"eval":eval_job,"class":class_job,"sweep":sweep_job}[a.cmd]
    with Pool(a.procs,initializer=configure) as pool:
        list(pool.imap_unordered(fn,js))
    if a.cmd=="tune" and not a.noise:
        aggregate()

if __name__=="__main__":
    main()
