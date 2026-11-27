"""Bank, design-grid, and evaluation jobs for the single-channel APFM."""
import argparse
from dataclasses import asdict
import json
from multiprocessing import Pool
from pathlib import Path

import apfm
import coef as C
import final as F

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
FACTORS = F.TUNE_FACTORS
SCEN = ("stat", "RS-near", "RS-far", "EN0", "ENm10")
METHODS = ("APFM", "VSS-NFBFxLMS")
PARAMETERS = HERE / "APFM_PARAMETERS.json"
SELECTED = OUT / "apfm_selected.json"


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    F.ab.atomic_write(path, lambda f: json.dump(data, f, allow_nan=False))


def settings(index):
    source = read(PARAMETERS)
    if not source["source"].strip():
        raise ValueError("parameter source is required")
    values = dict(source["values"])
    for key in ("mu_tilde", "mu_max"):
        values[key] *= FACTORS[index]
    return apfm.Settings(**values)


def target(folder, case):
    return OUT/folder/"{}_{}_s{}.json".format(*case)


def bank(index, pair_id):
    params = settings(index)
    path = OUT/"apfm_bank"/f"g{index}_pair{pair_id:02}.json"
    if path.exists():
        saved = read(path)
        if saved["settings"] == asdict(params) and saved["pair"] == pair_id:
            return
    seed = 900+pair_id
    x = apfm.np.random.default_rng(seed).standard_normal(30*F.FS)
    pair = F.load_pair(pair_id)
    s, primary = pair["S"], pair["P"]
    d = F.p.signal.lfilter(primary,[1.],x)
    dd = dict(x=x,d=d,hS_seq=[(0,s)],hSm_seq=[(0,s[:F.LS_FINAL])])
    sim = apfm.simulate(dd,params)
    record = dict(pair=pair_id,seed=seed,settings=asdict(params),div=int(sim["div"]),
                  w=None if sim["div"] else sim["w"].tolist())
    write(path,record)


def make_bank(task):
    bank(*task)


def get_bank(index, seed):
    pair_id = seed%8+1
    path = OUT/"apfm_bank"/f"g{index}_pair{pair_id:02}.json"
    payload = read(path)
    if payload["settings"] != asdict(settings(index)) or payload["pair"] != pair_id:
        raise ValueError(f"bank does not match current design: {path}")
    return None if payload["div"] else apfm.np.asarray(payload["w"])


def run_case(task):
    mode,method,index,case = task
    params = settings(index)
    path = target(f"apfm_{mode}/{method}/g{index}",case)
    if path.exists() and read(path)["settings"] == asdict(params):
        return
    cls,scen,seed = case
    trained = get_bank(index,seed) if method == "APFM" else None
    if mode == "tune" and method == "APFM" and trained is None:
        write(path,dict(cls=cls,scen=scen,seed=seed,method=method,settings=asdict(params),
                        div=1,bank_failed=True))
        return
    if method == "APFM" and trained is None:
        raise ValueError("selected bank must be trained successfully before evaluation")
    final = mode == "eval"
    dd,_,_ = F.build(*case,F.LS_FINAL,final=final)
    change = scen.startswith("RS-")
    capture = [F.CB+k for k in C.MAIN_KS] if change else []
    sim = apfm.simulate(dd,params,bank=trained,checkpoints=capture)
    reference = None
    if final and change and not sim["div"]:
        post,_,_ = F.build(cls,"post:"+scen,seed,F.LS_FINAL,final=True)
        ps = apfm.simulate(post,params,bank=trained)
        reference = dict(e=ps["e"],d=post["d"],div=ps["div"])
    row = dict(cls=cls,scen=scen,seed=seed,method=method,settings=asdict(params),
               **F.measures(sim,dd,reference))
    if final and change:
        row["J"] = C.frozen_anr(dd,{k:None if sim["div"] else sim["checkpoints"][F.CB+k] for k in C.MAIN_KS})
        start = F.CB*F.B
        row["power_db"] = {str(s):None if sim["div"] else F.ratio_db(sim["e"],dd["d"],start,start+int(s*F.FS))
                           for s in (.5,1,2)}
    if final and method == "APFM":
        choices = apfm.np.asarray(sim["choices"],dtype=int)
        row["choice_counts"] = apfm.np.bincount(choices,minlength=3).tolist()
        row["switches"] = int(apfm.np.count_nonzero(apfm.np.diff(choices)))
        row["noncurrent_first_2s"] = (int(apfm.np.count_nonzero(
            choices[F.CB:F.CB+2*F.BPS] != apfm.CURRENT)) if change else None)
    write(path,row)


def cases(final):
    classes = F.FINAL_CLASSES if final else F.PILOT_CLASSES
    scenarios = SCEN if final else F.TUNE_SCEN
    seeds = F.FINAL_SEEDS if final else F.TUNE_SEEDS
    return [(c,s,seed) for seed in seeds for s in scenarios for c in classes]


def select():
    table = []
    for index,factor in enumerate(FACTORS):
        rows = [read(target(f"apfm_tune/APFM/g{index}",c)) for c in cases(False)]
        if any(r["settings"] != asdict(settings(index)) for r in rows):
            raise ValueError("tuning cache uses different settings")
        complete = [r for r in rows if not r["div"]]
        table.append(dict(index=index,factor=factor,settings=asdict(settings(index)),n=len(rows),
                          diverged=sum(r["div"] for r in rows),
                          bank_failures=sum(bool(r.get("bank_failed")) for r in rows),
                          ANR=F.class_mean(complete,"L1_whole")))
    chosen = min(table,key=lambda r:(r["diverged"],float("inf") if r["ANR"] is None else r["ANR"],r["index"]))
    if chosen["bank_failures"] or any(get_bank(chosen["index"], seed) is None for seed in F.FINAL_SEEDS):
        raise ValueError("selected grid point has a divergent trained filter")
    write(SELECTED,dict(index=chosen["index"],candidates=table,source=read(PARAMETERS)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("bank","tune","select","eval"))
    parser.add_argument("--method",choices=METHODS)
    parser.add_argument("--shard",type=F.parse_shard,default=(0,1))
    parser.add_argument("--procs",type=int,default=1)
    args = parser.parse_args()
    if args.mode == "select":
        select()
        return
    if args.mode == "bank":
        tasks = [(g,pair) for g in range(len(FACTORS)) for pair in range(1,9)]
        fn = make_bank
    elif args.mode == "tune":
        tasks = [("tune","APFM",g,c) for g in range(len(FACTORS)) for c in cases(False)]
        fn = run_case
    else:
        if not args.method:
            parser.error("eval needs --method")
        selection = read(SELECTED)
        if selection["source"] != read(PARAMETERS):
            raise ValueError("selected parameters do not match the immutable grid")
        tasks = [("eval",args.method,selection["index"],c) for c in cases(True)]
        fn = run_case
    with Pool(args.procs) as pool:
        for _ in pool.imap_unordered(fn,F.shard_jobs(tasks,args.shard)):
            pass


if __name__ == "__main__":
    main()
