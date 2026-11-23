"""One-at-a-time sensitivity of the frozen final2 proposed method.

Run with `python sens.py run --procs N`, or rebuild the tables with
`python sens.py report`. Results are saved atomically, one case per JSON,
and completed results are reused when the run is resumed.
"""
import argparse
from collections import defaultdict
import json
from multiprocessing import Pool
from pathlib import Path
import tarfile
from unittest.mock import patch

import final as F
import final2
from paired_ci import paired

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "sens_run"
REFERENCE = HERE / "out" / "final2_run"
SCENARIOS = ("stat", "RS-near", "RS-far", "EN0")
DEFAULT = dict(gamma=4, rho1=0.25, rho2=0.5, sigma_e=1, n2=16)
SETTINGS = {
    "default": DEFAULT,
    **{f"gamma-{v}": dict(DEFAULT, gamma=v) for v in (2, 3, 6, 8)},
    **{f"rho-{a:g}-{b:g}": dict(DEFAULT, rho1=a, rho2=b) for a,b in ((.18,.35),(.35,.7))},
    **{f"sigma-{v:g}": dict(DEFAULT, sigma_e=v) for v in (.5,1.5)},
    **{f"n2-{v}": dict(DEFAULT, n2=v) for v in (8,32)},
}
JOBS = [(c,s,sd) for c in F.FINAL_CLASSES for s in SCENARIOS for sd in F.FINAL_SEEDS]


def initialize():
    final2.configure()


def path_for(name, job):
    c,s,sd = job
    return OUT / name / f"{c}_{s}_s{sd}.json"


def score_case(job, names):
    cls, scen, seed = job
    pending = []
    for name in names:
        path = path_for(name,job)
        if path.exists():
            old = json.loads(path.read_text())
            if old.get("version") == F.FINAL_VERSION and old.get("settings") == SETTINGS[name]:
                continue
        pending.append(name)
    if not pending:
        return
    dd, Rall, st = F.build(cls,scen,seed,F.LS_FINAL,final=True)
    post = F.build(cls,"post:"+scen,seed,F.LS_FINAL,final=True) if F.has_post(scen) else None
    raw_x4 = F.p.parse(["c5","--preset","final"]).X4
    config = F.q5.config

    for name in pending:
        values = SETTINGS[name]
        x4 = dict(raw_x4,z_thr=values["gamma"])

        def changed_config(*args, **kwargs):
            cfg = config(*args, **kwargs)
            cfg.update(K=values["n2"], theta=values["rho2"],
                       early_ok=(cfg["early_ok"][0],values["rho1"]), mu_e=values["sigma_e"])
            return cfg

        if name == "default":
            assert raw_x4["z_thr"] == values["gamma"]
            cfg = config(early_ok=(4,.25))
            assert (cfg["K"],cfg["theta"],cfg["mu_e"],cfg["early_ok"][1]) == (16,.5,1,.25)
            rule, sim = F.run_rule(F.PROPOSED,dd,Rall,st,raw_x4)
            post_sim = F.run_rule(F.PROPOSED,*post,raw_x4)[1] if post is not None else None
        else:
            with patch.object(F.q5,"config",changed_config):
                rule, sim = F.run_rule(F.PROPOSED,dd,Rall,st,x4)
                post_sim = F.run_rule(F.PROPOSED,*post,x4)[1] if post is not None else None
        row = dict(cls=cls,scen=scen,seed=seed,method=F.PROPOSED,kind="rule",variant="main",
                   **F.measures(sim,dd,post_sim),
                   **F.trial_stats(rule,dict(dd,scored_blocks=sim["n_done"])))
        if post_sim is not None:
            row["post_div"] = int(post_sim["div"])
        path = path_for(name,job)
        path.parent.mkdir(parents=True,exist_ok=True)
        record = dict(version=F.FINAL_VERSION,setting=name,settings=values,rows=[row])
        F.ab.atomic_write(path,lambda handle: json.dump(record,handle,allow_nan=False))


def baseline_job(job):
    score_case(job,("default",))
    actual, = json.loads(path_for("default",job).read_text())["rows"]
    saved = json.loads((REFERENCE / path_for("default",job).name).read_text())["rows"]
    expected, = [r for r in saved if r["method"] == F.PROPOSED and r["variant"] == "main"]
    key = "L1_whole"
    if key not in actual or key not in expected or actual[key] != expected[key]:
        raise ValueError(f"{job}: computed {actual.get(key)!r}, archived {expected.get(key)!r}")
    return job


def variant_job(job):
    score_case(job,tuple(name for name in SETTINGS if name != "default"))
    return job


def run(procs):
    if not all((REFERENCE / path_for("default",job).name).is_file() for job in JOBS):
        with tarfile.open(HERE/"results"/"final2_run.tar.gz") as archive:
            archive.extractall(REFERENCE.parent,filter="data")
    if not (HERE/"EVAL_FROZEN.json").is_file():
        raise SystemExit("EVAL_FROZEN.json is required")
    assert len(JOBS) == 800 and tuple(F.FINAL_SEEDS) == tuple(range(100,120))
    with Pool(procs,initializer=initialize) as pool:
        for i,_ in enumerate(pool.imap_unordered(baseline_job,JOBS),1):
            if i % 50 == 0:
                print(f"Default verified {i}/{len(JOBS)}",flush=True)
        print("Default L1_whole: all 800 exactly match the archive.",flush=True)
        for i,_ in enumerate(pool.imap_unordered(variant_job,JOBS),1):
            if i % 25 == 0:
                print(f"Variants completed for {i}/{len(JOBS)} cases",flush=True)
    report()


def report():
    groups = defaultdict(list)
    parameters = {}
    seen = set()
    for path in sorted(OUT.rglob("*.json")):
        data = json.loads(path.read_text())
        name, values = data["setting"], data["settings"]
        if name in parameters and parameters[name] != values:
            raise ValueError(f"Inconsistent parameters for {name}")
        parameters[name] = values
        row, = data["rows"]
        key = name,row["cls"],row["scen"],row["seed"]
        if key in seen:
            raise ValueError(f"Duplicate result {key}")
        seen.add(key)
        groups[name,row["scen"]].append(row)
    if not groups:
        raise SystemExit(f"No results in {OUT}")

    lines = ["# 提案法のパラメータ感度", "",
             f"集計対象は {len(seen)} 実行。騒音 10 種、seed 100–119。1 設定につき各条件 200 実行。",
             f"変化後の試行開始は切替ブロックから {F.p.RS_TRUE_WIN} ブロック以内で判定する。",
             "移行は実行中の確定の総数。stat・EN0 は経路変化なしでの移行を数える。",
             "発散欄は評価本体。回復中央値は Kaplan–Meier 法。括弧は右打ち切り件数。",
             "L1 whole は完走分を seed 内でクラス等重み平均し、seed 間で平均した値。",
             "対応差は変更設定 − 既定値。seed を初期値 20261121 で 2,000 回再標本化した 95% 区間。", "",
             "| 設定 | γ | (ρ1, ρ2) | σ_E | n2 |", "|---|---:|---:|---:|---:|"]
    names = sorted(parameters,key=lambda n: (n != "default",n))
    for name in names:
        v = parameters[name]
        lines.append(f"| {name} | {v['gamma']:g} | ({v['rho1']:g}, {v['rho2']:g}) | {v['sigma_e']:g} | {v['n2']} |")
    lines.extend(("", "| 設定 | 条件 | 実行 | 発散 | 回復中央値 s（打切） | 移行 | 変化後に試行なし | L1 whole dB | 既定値との差 [95%]（対数） |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---|"))
    for name in names:
        for scen in SCENARIOS:
            rows = groups[name,scen]
            if not rows:
                continue
            if F.has_post(scen):
                med,cens = F.recovery_summary(rows)
                seconds = f"{int(med)*F.B/F.FS:.3f}" if med.isdigit() else med
                recovery = f"{seconds}（{cens}）"
                no_start = sum(not r["detection_blocks"] for r in rows)
            else:
                recovery = no_start = "—"
            vals = list(F.seed_table([r for r in rows if not r["div"]],"L1_whole").values())
            avg = F.fmt(float(F.np.mean(vals)) if vals else None,3)
            v = paired(rows,groups["default",scen],lambda r:r.get("L1_whole"))
            diff = "—" if v is None else f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]（{v[3]}）"
            lines.append(f"| {name} | {scen} | {len(rows)} | {sum(r['div'] for r in rows)} | "
                         f"{recovery} | {sum(r['n_confirm'] for r in rows)} | {no_start} | {avg} | {diff} |")
    (HERE/"RESULTS_sens.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("run","report"))
    parser.add_argument("--procs",type=int,default=2)
    args = parser.parse_args()
    if args.command == "run":
        initialize()
        run(args.procs)
    else:
        report()


if __name__ == "__main__":
    main()
