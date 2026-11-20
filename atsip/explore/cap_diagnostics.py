"""S1c diagnostic traces, E12 comparison, and report observations."""
import argparse
import json
from itertools import product
from multiprocessing import Pool
from statistics import mean, median

import final as f

np = f.np
E12_DIR = f.CAP_DIR / "E12"
REPORT = f.HERE / "RESULTS_final_cap.md"
E12_GROUPS = [(None, f.q5.CLASSES), *((m, f.q5.SHAT_CLASSES) for m in f.q5.SHATS)]
E12_JOBS = [(f"{cls}_s{seed}", model) for model, classes in E12_GROUPS
            for cls, seed in product(classes, f.CAP_SEEDS)]
PATH_CLASSES = ("pc", "pcb", "EVpc")


def e12_job(job):
    tag, model = job
    dest = E12_DIR / f"{tag}_{model or 'S'}.json"
    if dest.exists():
        cached = json.loads(dest.read_text())
        if cached["version"] == f.CAP_VERSION:
            return cached["rows"]
    dd, R, st = f.q.prep(tag, shat=model)
    nb = len(dd["d"]) // f.B
    X4 = f.p.parse(["c5", "--preset", "final"]).X4
    cfg = f.q5.config(early_ok=(4, 0.25))
    cls = tag.rsplit("_s", 1)[0]
    rows = []
    for rule_cls, label in ((f.q5.P5, "P5-early"), (f.P5Cap, "P5-early-cap")):
        rule = rule_cls(cfg, dd, nb, X4)
        rule.frame = f.q4.Frame4(dd, rule)
        sim = f.T.simulate(dd, R, st, rule, nb=nb, error_frame=rule.frame)
        row = f.q.score(tag, label, rule, sim, dd, False)
        stats = f.trial_stats(rule, dict(dd, changes=f.p.change_blocks(dd),
                                        scored_blocks=len(sim["tr"]["mu_app"])))
        row.update(f.cap_steps(sim["tr"]), decide_blocks=stats["decide_blocks"], shat=model or "S")
        if not row["div_num"]:
            row["phi_whole"] = f.p.dbm(sim["ex"])
        rows.append(row)
    dest.parent.mkdir(parents=True, exist_ok=True)
    f.ab.atomic_write(dest, lambda stream: json.dump(dict(version=f.CAP_VERSION, rows=rows), stream))
    print(f"E12 {tag} {model or 'S'}: div {[r['div_num'] for r in rows]}", flush=True)
    return rows


def e12(procs):
    with Pool(procs) as pool:
        return list(pool.imap_unordered(e12_job, E12_JOBS))


def table(lines, headers, rows):
    lines.extend(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"])
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    lines.append("")


def average(values, ndigits=3):
    return f.fmt(mean(values), ndigits) if values else "—"


def distribution(values):
    return f"{min(values)} / {median(values):g} / {max(values)}" if values else "—"


def valve(lines):
    lines.extend(["### valve の開始40ブロック", "",
                  "stat、seed 10、Ŝ256。パワーは128サンプルの平均二乗。係数ノルムは誤差を生成した更新前の値。"
                  "発散判定ブロックを記録し、その後は停止のため欠測。g の最小値と a の超過は停止ブロックも含む。", ""])
    dd, R, st = f.build("valve", "stat", 10, 256)
    X4 = f.p.parse(["c5", "--preset", "final"]).X4
    for label in ("nors", "P5-early", "nors-cap"):
        rows = []
        def capture(i, c, eb, mu):
            lo, hi = i * f.B, (i+1) * f.B
            power = lambda x: float(np.mean(x*x))
            rows.append((i, power(dd["x"][lo:hi]), c.g, mu*f.H_STEP, float(np.linalg.norm(c.w)),
                         power(eb), power(dd["d"][lo:hi])))
        _, sim = f.run_rule(label, dd, R, st, X4, nb=40, hook=capture)
        stop = f"発散ブロック {sim['n_done']}（{sim['failure']['sample']/f.FS:.4f} 秒）。" if sim["div"] else "先頭40ブロックで発散0件。"
        peak = max(row[3] for row in rows)
        exceed = [r[0] for r in rows if r[3] > 0.55]
        lines.extend([f"#### {label}", "",
                      f"{stop} 最小 g は {min(r[2] for r in rows):.6f}、最大 a は {peak:.6f}。"
                      f"a > 0.55 のブロックは {', '.join(map(str,exceed)) if exceed else 'なし'}。", ""])
        rows = [(i, *(f"{v:.6g}" for v in values)) for i, *values in rows]
        rows.extend((i, *("—",)*6) for i in range(len(rows),40))
        table(lines, ("ブロック（0始まり）","入力パワー","g","a","‖w‖","誤差パワー","一次騒音パワー"), rows)


def cap_findings(lines):
    jobs = list(product(f.PILOT_CLASSES, f.PILOT_SCEN, f.CAP_SEEDS))
    rows = [r for c,s,seed in jobs for r in json.loads((f.CAP_DIR/f"{c}_{s}_s{seed}.json").read_text())["rows"]]
    assert len(rows) == 1080
    idx = {(r["cls"],r["scen"],r["seed"],r["method"]):r for r in rows}
    assert set(idx) == {(*key,label) for key in jobs for label in f.CAP_RULES}
    assert all(r["a_max_early"]<=f.H_STEP/f.GE_CAP+1e-15 for r in rows if r["method"].endswith("-cap"))
    assert all(r.get("a_max_bg") is None or r["a_max_bg"]<=f.H_STEP/f.GE_CAP+1e-15 for r in rows)
    lines.extend(["### 実測経路の発散、判定、基準との一致", "",
                  "基準は対応する再始動なし規則。両規則が完走した場合に、スピーカー出力とマイク誤差を完全一致で比較した。"
                  "誤確定と見逃しは観測された判定を集計し、発散で検出窓を完走しなかった変化は見逃しに含めない。", ""])
    out=[]
    for scen,label in product((*f.PILOT_SCEN,"全条件"),f.CAP_RULES):
        rs=[r for r in rows if r["method"]==label and (scen==r["scen"] or scen=="全条件")]
        match=[r["same_as_ref"] for r in rs if "same_as_ref" in r]
        out.append((scen,label,sum(r["div"] for r in rs),sum(r["false_confirm"] for r in rs),
                    sum(r["missed"] or 0 for r in rs) if label.startswith("P5") else "—",
                    f"{sum(match)}/{len(match)}" if match else "—"))
    table(lines,("条件","手法","発散","誤確定","見逃し","基準と一致"),out)
    lines.extend(["### クラス別のステップ上限", "",
                  "前半は前景の適応更新を集計し、転送・復元を除いた。裏試行の更新は cap 版で別に記録した。", ""])
    out=[]
    for cls,label in product(f.PILOT_CLASSES,f.CAP_RULES):
        rs=[r for r in rows if r["cls"]==cls and r["method"]==label]
        vals=[r["a_max_early"] for r in rs]
        bg=[r["a_max_bg"] for r in rs if r["a_max_bg"] is not None]
        out.append((cls,label,f.fmt(max(vals),6),sum(a>0.55 for a in vals),f.fmt(max(bg) if bg else None,6)))
    table(lines,("クラス","手法","前半 a 最大","a>0.55 の件数","裏 a 最大"),out)
    lines.extend(["### 条件別の対応差と悪化した対", "",
                  "cap−従来。正の差は L1 の上昇または回復の遅れ。回復差は両方が実測回復した対のみ。"
                  "回復の打切は別列に数え、参照発散などで回復を定義できない実行はその比較から外した。", ""])
    summary=[]
    worst=[]
    for scen,label in product(f.PILOT_SCEN,("nors","P5-early")):
        pairs=[(idx[*k,label+"-cap"],idx[*k,label]) for k in jobs if k[1]==scen]
        dif=lambda name:[a[name]-b[name] for a,b in pairs if not a["div"] and not b["div"]
                         and a.get(name) is not None and b.get(name) is not None]
        whole,event,rec=(dif(k) for k in ("L1_whole","L1_ev","rec_blocks"))
        cens=[(a["rec_cens"],b["rec_cens"]) for a,b in pairs if "rec_cens" in a and "rec_cens" in b]
        for field in ("L1_whole", "L1_ev", "rec_blocks"):
            candidates=[(a[field]-b[field],a) for a,b in pairs if not a["div"] and not b["div"]
                        and a.get(field) is not None and b.get(field) is not None and a[field]>b[field]]
            if candidates:
                delta, row = max(candidates, key=lambda item:item[0])
                worst.append((scen,label,field,row["cls"],row["seed"],f.fmt(delta,6)))
        summary.append((scen,label,average(whole),len(whole),average(event),len(event),average(rec,1),len(rec),
                        sum(a>b for a,b in cens),sum(a<b for a,b in cens),
                        sum(a["div"]>b["div"] for a,b in pairs),sum(a["div"]<b["div"] for a,b in pairs),
                        max([x for x in whole if x>0]+[0]).__format__(".3f"),
                        max([x for x in event if x>0]+[0]).__format__(".3f"),
                        max([x for x in rec if x>0]+[0]).__format__(".1f")))
    table(lines,("条件","手法","L1 whole Δ","対数","直後2秒 Δ","対数","回復 Δ","対数",
                 "新規打切","解消打切","新規発散","解消発散","最大 L1 悪化","最大直後悪化","最大回復遅れ"),summary)

    if worst:
        lines.extend(["各条件・指標で最大の悪化を記録した対。", ""])
        table(lines,("条件","手法","指標","クラス","seed","差"),worst)


def e12_findings(lines):
    groups=[json.loads((E12_DIR/f"{tag}_{model or 'S'}.json").read_text())["rows"] for tag,model in E12_JOBS]
    assert len(groups)==140 and all(len(group)==2 for group in groups)
    assert all(a["a_max_early"]<=f.H_STEP/f.GE_CAP+1e-15 for _,a in groups)
    lines.extend(["### 既定経路の E12 入力", "",
                  "E12 と同じ30秒・seed 10–14、Ŝ=S の13クラスと g0.7、g1.4、n30 の各5クラス。"
                  "既定経路と E12 の Φ 超過の定義を使った。Φ whole は全ブロックの線形 Φ 超過を平均して dB 化。"
                  "L1 と Φ の差は cap−従来で、両方が完走した対の平均。", ""])
    counts=[]
    for model,_ in E12_GROUPS:
        sel=[p for p in groups if p[0]["shat"]==(model or "S")]
        for which,label in enumerate(("P5-early","P5-early-cap")):
            rs=[p[which] for p in sel]
            delays=[d for r in rs for d in r["decide_blocks"]]
            counts.append((model or "S",label,len(rs),*(sum(r[k] or 0 for r in rs) for k in ("false_rs","missed_rs","div_num")),
                           distribution(delays),f.fmt(max(r["a_max_early"] for r in rs),6)))
    table(lines,("モデル","手法","実行","誤確定","見逃し","発散","移行ブロック 最小/中央/最大","前半 a 最大"),counts)
    differences=[]
    for model,classes in E12_GROUPS:
        for cls in (*classes,"全クラス"):
            sel=[(b,a) for b,a in groups if b["shat"]==(model or "S") and (cls=="全クラス" or b["class"]==cls)
                 and not a["div_num"] and not b["div_num"]]
            delta=lambda k:average([a[k]-b[k] for b,a in sel if a.get(k) is not None and b.get(k) is not None])
            metric=lambda which:average([p[which]["L1_ev"] for p in sel if p[which].get("L1_ev") is not None])
            differences.append((model or "S",cls,len(sel),delta("L1_whole"),delta("phi_whole"),delta("ex_ev"),
                                metric(0) if cls in PATH_CLASSES else "—",
                                metric(1) if cls in PATH_CLASSES else "—",
                                delta("L1_ev") if cls in PATH_CLASSES else "—"))
    table(lines,("モデル","クラス","完走対","L1 whole Δ","Φ whole Δ","Φ 直後2秒 Δ",
                 "経路変化の直後2秒 L1 従来","同 cap","同 Δ"),differences)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=("e12","report"))
    parser.add_argument("--procs", type=int, default=2)
    args = parser.parse_args()
    if args.cmd == "e12":
        e12(args.procs)
    else:
        lines = ["", "## 所見", ""]
        valve(lines)
        cap_findings(lines)
        e12_findings(lines)
        initial = REPORT.read_text().split("\n## 所見\n",1)[0].rstrip()
        REPORT.write_text(initial + "\n" + "\n".join(lines) + "\n")
        print(f"Appended S1c findings to {REPORT.name}")


if __name__ == "__main__":
    main()
