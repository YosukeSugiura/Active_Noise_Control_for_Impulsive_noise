"""Read-only replay of frozen P5 for Proposition 3 (eta).
Original judge.py, judge_extra.py, result files and controller remain unchanged.
rho_true_v is the true ratio of Proposition 3: both errors include v.
rho_true (without v) is separately checked against the old scatter.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import judge as J
import judge_extra as E
import final as F

HERE=Path(__file__).resolve().parent
OUT=HERE/"out"/"judge_eta_run"
OLD=HERE/"out"/"judge_scatter.npz"
DEST=HERE/"out"/"judge_scatter_eta.npz"
VERSION=1
FIELDS=("rho_rec","rho_true","rho_true_v","kappa","bound_ok")

def finite(x):
    return x is not None and math.isfinite(float(x))

def number(x):
    return float(x) if finite(x) else None

def full(k,eta,r,t):
    valid=all(finite(x) and float(x)>=0 for x in (k,eta,r,t))
    if not valid or eta>=1:
        return None,-1,False
    q=math.sqrt(r)
    rhs=(k+eta)*(1+q)/(1-eta)
    if not math.isfinite(rhs):
        return None,-1,False
    lhs=abs(math.sqrt(t)-q)
    return rhs,int(lhs<=rhs),bool(k*(1+q)+2*eta<1-q)

def suff0(k,r):
    return bool(finite(k) and finite(r) and k>=0 and r>=0 and k*(1+math.sqrt(r))<1-math.sqrt(r))

class EtaObserver(E.ExtraObserver):
    def capture(self,rule,tb,wf):
        rho=super().capture(rule,tb,wf)
        _,row,wf,wb=self.pending
        start,end=row["window_start"],row["window_end"]
        rec=rule.frame.trial_frame(wf)[-F.B:]
        true=self.true_residual(wf,start,end,with_v=True)
        den=float(np.linalg.norm(rec))
        num=float(np.linalg.norm(rec-true))
        eta=num/den if den>0 and math.isfinite(num) and math.isfinite(den) else None
        eta=number(eta)
        ex=self.extra[-1]
        k=ex["kappa"]
        r,t,tv=(float(row[n]) for n in ("rho_rec","rho_true","rho_true_v"))
        # Recompute the legacy eta=0 test using the exact original code.
        _,_,old=E.bound(k,r,tv)
        rhs,ok,suff=full(k,eta,r,tv)
        ex.update(eta=eta,eta_numerator=number(num),eta_denominator=number(den),
                  full_rhs=rhs,full_bound_ok=ok,suff_full=suff,
                  rho_rec=J.encode(r),rho_true=J.encode(t),rho_true_v=J.encode(tv),
                  bound_ok=-1 if old is None else int(old),suff_zero=suff0(k,r),
                  decision=row["decision"])
        return rho

def setup():
    J.setup()

def path_for(job):
    return OUT/J.path_for(job).relative_to(J.OUT)

def atomic_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    F.ab.atomic_write(path,lambda h:json.dump(obj,h,allow_nan=False,separators=(",",":")))

def compute(job):
    level,c,s,seed=job
    data=J.build(c,s,seed,level)
    observer=EtaObserver(data[0])
    x4=F.p.parse(["c5","--preset","final"]).X4
    rule,sim=observer.run(data,x4)
    rows=[]
    assert len(observer.extra)==len(observer.records)
    for ex,r in zip(observer.extra,observer.records):
        assert (ex["block"],ex["trial_block"])==(r["block"],r["trial_block"])
        out={k:ex[k] for k in (*FIELDS,"block","trial_block","decision","eta","eta_numerator",
                "eta_denominator","full_rhs","full_bound_ok","suff_full","suff_zero","threshold")}
        out["phase"]=r.get("phase")
        rows.append(out)
    return dict(version=VERSION,job=list(job),decisions=rows,div=int(sim["div"]),direct_checks=observer.direct_checks)

def load(path):
    return json.loads(path.read_text())

def validate(obj,job):
    if obj.get("version")!=VERSION or obj.get("job")!=list(job):
        raise ValueError(f"Invalid checkpoint {job}")
    keys={(r["trial_block"],r["block"]) for r in obj["decisions"]}
    if len(keys)!=len(obj["decisions"]):
        raise ValueError(f"Duplicate decisions {job}")
    for r in obj["decisions"]:
        if not set((*FIELDS,"eta","full_rhs","full_bound_ok","suff_full"))<=r.keys():
            raise ValueError(f"Missing fields {job}")
    return obj

def run_case(job):
    p=path_for(job)
    if p.exists():
        obj=validate(load(p),job)
        return job,len(obj["decisions"]),True
    obj=validate(compute(job),job)
    atomic_json(p,obj)
    return job,len(obj["decisions"]),False

def gate():
    setup()
    # Test zero, exact equality, eta>=1, missing denominators and sufficient condition.
    assert full(0.,0.,.25,.25)==(0.,1,True)
    assert full(0.,1.,.25,.25)==(None,-1,False)
    assert full(None,0.,.25,.25)==(None,-1,False)
    assert full(0.,0.,.25,1.)==(0.,0,True)
    jobs=[("nominal","G","RS-far",100),("nominal","a145","EN0",100),
          ("-30dB","jackhammer","RS-far",114),("-20dB","jackhammer","stat",101)]
    rows=[]
    for job in jobs:
        r=compute(job)
        rows.append(dict(job=list(job),decisions=len(r["decisions"]),div=r["div"]))
        print("Gate",job,len(r["decisions"]),flush=True)
    atomic_json(HERE/"out"/"judge_eta_gate.json",dict(version=VERSION,checks=rows))

def run(level,scen,group,procs):
    jobs=J.cases(level,scen,group)
    assert len(jobs)==100
    with Pool(procs,initializer=setup) as pool:
        for i,(job,n,reused) in enumerate(pool.imap_unordered(run_case,jobs),1):
            print(f"{i}/100 {job}: {n} decisions; {'cached' if reused else 'replayed'}",flush=True)

def key_old(a,i):
    return (str(a["model_error"][i]),str(a["noise"][i]),str(a["condition"][i]),
            int(a["seed"][i]),int(a["trial_block"][i]),int(a["block"][i]))

def same(a,b):
    a=float("nan") if a is None else float(a)
    b=float("nan") if b is None else float(b)
    return a==b or (math.isnan(a) and math.isnan(b))

def fmt(x):
    if x is None:return "—"
    if math.isnan(float(x)):return "NaN"
    return f"{float(x):.9g}"

def rowline(vals):
    return "| "+" | ".join(map(str,vals))+" |"

def dist(rs):
    vals=[float(r["eta"]) for r in rs if finite(r["eta"])]
    q=[fmt(v) for v in np.quantile(vals,[.5,.95,.99])] if vals else ["—"]*3
    return [len(rs),len(vals),len(rs)-len(vals),*q,fmt(max(vals)) if vals else "—",sum(v>=1 for v in vals)]

def kind(key,phase=None):
    if key[2]=="EN0":
        return "外乱あり条件"
    if key[2]=="stat":
        return "変化なし"
    # Match original judge's phase definition using end of decision window.
    return "変化後" if (key[-1]+1)>=F.CB else "変化前"

def reverse(r):
    return r["decision"]=="transfer" and float(r["rho_true_v"])>=1

def report():
    jobs=[job for level in J.LEVELS for job in J.cases(level)]
    assert len(jobs)==2000
    lookup={}
    groups=defaultdict(list)
    div=0
    for job in jobs:
        obj=validate(load(path_for(job)),job)
        div+=obj["div"]
        for r in obj["decisions"]:
            key=(*job,r["trial_block"],r["block"])
            if key in lookup:raise ValueError(f"Duplicate replay key {key}")
            lookup[key]=r
            groups[job[0],job[2]].append((key,r))
    with np.load(OLD,allow_pickle=False) as f:
        old={k:f[k] for k in f.files}
    n=len(old["rho_rec"])
    old_keys=[key_old(old,i) for i in range(n)]
    if len(set(old_keys))!=n:raise ValueError("Duplicate original keys")
    index={k:i for i,k in enumerate(old_keys)}
    newkeys=set(lookup)
    missing=set(index)-newkeys
    added=newkeys-set(index)
    common=set(index)&newkeys
    mismatch=Counter()
    per=defaultdict(Counter)
    maxdiff=defaultdict(float)
    old_fail=[]
    for key in sorted(common):
        i=index[key];r=lookup[key]
        for field in FIELDS:
            if not same(old[field][i],r[field]):
                mismatch[field]+=1
                per[key[0],key[2]][field]+=1
                if finite(old[field][i]) and finite(r[field]):
                    maxdiff[field]=max(maxdiff[field],abs(float(old[field][i])-float(r[field])))
        if str(old["decision"][i])!=r["decision"]:
            mismatch["decision"]+=1
            per[key[0],key[2]]["decision"]+=1
    for i in np.flatnonzero(old["bound_ok"]==0):
        key=old_keys[int(i)]
        old_fail.append((key,lookup.get(key)))
    # Preserve original arrays exactly, adding new arrays in old key order.
    # Additional/missing keys are explicitly reported; replay-only keys remain in JSON.
    arrays=dict(old)
    for field in ("eta","eta_numerator","eta_denominator","full_rhs","rho_rec","rho_true","rho_true_v","kappa"):
        outname=("replay_"+field) if field in FIELDS else field
        if outname in arrays:raise ValueError(f"Existing array {outname}")
        arrays[outname]=np.array([float(lookup[k][field]) if k in lookup and lookup[k][field] is not None else np.nan
                                  for k in old_keys],dtype=np.float64)
    for field in ("full_bound_ok","suff_full","suff_zero","bound_ok"):
        outname="replay_bound_ok" if field=="bound_ok" else field
        if outname in arrays:raise ValueError(f"Existing array {outname}")
        arrays[outname]=np.array([int(lookup[k][field]) if k in lookup else (-1 if field in ("bound_ok","full_bound_ok") else 0)
                                  for k in old_keys],dtype=np.int8 if field in ("bound_ok","full_bound_ok") else np.bool_)
    arrays["eta_replay_present"]=np.array([k in lookup for k in old_keys],dtype=np.bool_)
    tmp=DEST.with_name(".judge_scatter_eta.tmp.npz")
    np.savez_compressed(tmp,**arrays)
    with np.load(tmp,allow_pickle=False) as f:
        assert set(f.files)==set(arrays)
        for k,a in old.items():
            b=f[k]
            if a.dtype!=b.dtype or not (np.array_equal(a,b,equal_nan=True) if a.dtype.kind in "fc" else np.array_equal(a,b)):
                raise ValueError(f"Old array changed {k}")
    tmp.replace(DEST)
    keys=[(l,s) for l in J.LEVELS for s in (J.SCENARIOS if l=="nominal" else J.SCENARIOS[:3])]
    lines=["# 命題3：ηを含む判定上界の検証","",
           f"実行 {len(jobs)}/2000、再計算判定 {len(lookup)}（既存 {n}）、全実行発散 {div}。",
           "判定窓128標本。η = norm(reconstructed foreground − true foreground with v)/norm(reconstructed foreground)。",
           "命題3の真の比は外乱を含むrho_true_v。rho_true（外乱なし）も再計算し既存値と照合。",
           "full_bound_ok は厳密な不等式（許容誤差なし）、1=成立、0=不成立、−1=評価不能。旧bound_okは旧コードの許容誤差を保持。",
           "κとηの分母ゼロ・非有限値、η≥1、非有限比は評価不能。十分条件は厳密な不等号。NaN同士は一致とする。",
           "既存judge_scatter.npzは変更せず、judge_scatter_eta.npzに既存配列をそのまま保持して追加。新旧キー不一致時は再計算値を明示し、既存配列を上書きしない。","",
           "## 一致確認","",
           f"既存のみ {len(missing)}、再計算のみ {len(added)}、共通キー {len(common)}。",
           "| 項目 | 完全一致 | 不一致 | 有限最大絶対差 |","|---|---:|---:|---:|"]
    for f in (*FIELDS,"decision"):
        lines.append(rowline([f,len(common)-mismatch[f],mismatch[f],fmt(maxdiff[f]) if f!="decision" else "—"]))
    lines+=["","| モデル | 条件 | "+ " | ".join((*FIELDS,"decision"))+" |","|---|---|"+"---:|"*6]
    for l,s in keys:
        lines.append(rowline([l,s,*[per[l,s][f] for f in (*FIELDS,"decision")]]))
    if missing or added:
        lines+=["","キー差異（全件）:"]+[f"- 既存のみ: {k}" for k in sorted(missing)]+[f"- 再計算のみ: {k}" for k in sorted(added)]
    lines+=["","## a. η分布","","| モデル | 条件 | 対象 | 判定数 | 有効 | 不明 | 中央値 | P95 | P99 | 最大 | η≥1 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for l,s in keys:
        rs=[r for _,r in groups[l,s]]
        for label,sub in (("全判定",rs),("移行のみ",[r for r in rs if r["decision"]=="transfer"])):
            lines.append(rowline([l,s,label,*dist(sub)]))
    lines+=["","## b. 完全上界の成立数","","| モデル | 条件 | 判定 | 成立 | 不成立 | 評価不能 | 旧η=0不成立（再計算） |","|---|---|---:|---:|---:|---:|---:|"]
    for l,s in [*keys,("全体","全条件")]:
        rs=list(lookup.values()) if l=="全体" else [r for _,r in groups[l,s]]
        c=Counter(r["full_bound_ok"] for r in rs)
        lines.append(rowline([l,s,len(rs),c[1],c[0],c[-1],sum(r["bound_ok"]==0 for r in rs)]))
    lines+=["",f"既存η=0の不成立 {len(old_fail)}件の追跡：","","| モデル | 条件 | 騒音 | seed | trial block | 判定block | κ | η | ρ | ρ★ | 完全RHS | 完全上界 |","|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k,r in old_fail:
        lines.append(rowline([*k,*([fmt(r[x]) for x in ("kappa","eta","rho_rec","rho_true_v","full_rhs")]+[r["full_bound_ok"]] if r is not None else ["—"]*6)]))
    lines+=["","## c. 移行判定の十分条件","",
            "旧条件は保存済みκとρからκ(1+sqrtρ)<1−sqrtρを再集計。参考として閾値ベースのκ<0.33/0.17も掲載。変化前を別掲し、EN0は外乱あり条件として別集計。",
            "| モデル | 条件/時期 | 移行数 | 旧条件（既存） | 旧条件（再計算） | 閾値ベース（既存） | 完全条件 | 完全条件成立かつ反転 | 全反転 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    def suffrow(l,label,items,oldmask):
        rs=[r for k,r in items if r["decision"]=="transfer"]
        ids=[i for i,k in enumerate(old_keys) if oldmask(k) and str(old["decision"][i])=="transfer"]
        oldn=sum(suff0(old["kappa"][i],old["rho_rec"][i]) for i in ids)
        conservative=sum(finite(old["kappa"][i]) and finite(old["threshold"][i]) and
                         old["kappa"][i] < (.33 if old["threshold"][i]==.25 else .17) for i in ids)
        return rowline([l,label,len(rs),f"{oldn}/{len(ids)}",sum(r["suff_zero"] for r in rs),
                        f"{conservative}/{len(ids)}",sum(r["suff_full"] for r in rs),
                        sum(r["suff_full"] and reverse(r) for r in rs),sum(reverse(r) for r in rs)])
    for l,s in keys:
        for phase in (("変化前","変化後") if s.startswith("RS-") else ("外乱あり条件",) if s=="EN0" else ("変化なし",)):
            mask=lambda k,l=l,s=s,phase=phase:k[0]==l and k[2]==s and kind(k)==phase
            lines.append(suffrow(l,s+"/"+phase,[(k,r) for k,r in groups[l,s] if mask(k)],mask))
    for l in J.LEVELS:
        for phase in ("変化後","変化前","変化なし","外乱あり条件"):
            mask=lambda k,l=l,phase=phase:k[0]==l and kind(k)==phase
            items=[(k,r) for k,r in lookup.items() if mask(k)]
            if items:
                lines.append(suffrow(l,"合計/"+phase,items,mask))
    lines+=["","## d. 反転した移行の全件","","| モデル | 騒音 | 条件 | seed | trial block | 判定block | 時期 | κ | η | ρ | ρ★ | 旧条件 | 完全条件 | 完全上界 |",
            "|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|---|---:|"]
    reversals=[(k,r) for k,r in lookup.items() if reverse(r)]
    for k,r in sorted(reversals):
        lines.append(rowline([*k,kind(k),*[fmt(r[x]) for x in ("kappa","eta","rho_rec","rho_true_v")],
                              r["suff_zero"],r["suff_full"],r["full_bound_ok"]]))
    lines+=["",f"反転総数 {len(reversals)}。完全十分条件成立かつ反転 {sum(r['suff_full'] for _,r in reversals)}。",""]
    F.ab.atomic_write(HERE/"RESULTS_judge_eta.md",lambda h:h.write("\n".join(lines)))
    print(f"Report complete: {len(jobs)} runs, {len(lookup)} decisions, exact mismatch {dict(mismatch)}",flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("gate","run","report"))
    p.add_argument("--level",choices=J.LEVELS)
    p.add_argument("--scen",choices=J.SCENARIOS)
    p.add_argument("--group",type=int,choices=(0,1))
    p.add_argument("--procs",type=int,default=4)
    a=p.parse_args()
    if a.command=="gate":gate()
    elif a.command=="report":report()
    else:
        if a.level is None or a.scen is None or a.group is None or a.procs<1:
            p.error("run requires level, scen, group and positive procs")
        run(a.level,a.scen,a.group,a.procs)

if __name__=="__main__":
    main()
