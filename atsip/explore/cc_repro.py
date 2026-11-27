"""Diagnose the C_FXLMS mismatch against stored cc_run values (volvo, RS-near).
For each case: stored measures, a rerun, a second rerun (determinism), and reruns with the primary noise d
perturbed by a relative 1e-12 and 1e-9 (sensitivity to rounding-size changes)."""
import json, sys
import numpy as np
import judge_extra as X
J, F, cc = X.J, X.F, X.cc

def main():
    X.setup()
    st = X.settings(cc.NAME)
    out = []
    for cls, scen, seed in (("volvo", "RS-near", 100), ("volvo", "RS-near", 101), ("volvo", "RS-far", 100),
                            ("a145", "RS-near", 100), ("jackhammer", "RS-near", 100)):
        job = ("nominal", cls, scen, seed)
        dd = J.build(cls, scen, seed, "nominal")[0]
        ref = X.original_reference(job, cc.NAME)
        row = dict(case=[cls, scen, seed], stored={k: ref.get(k) for k in ("L1_whole", "L1_pre", "L1_ev", "L1_last10")},
                   stored_lam=ref.get("lam"))
        def run(scale=None, tag=""):
            d2 = dict(dd)
            if scale:
                rng = np.random.default_rng(1)
                d2["d"] = dd["d"] * (1.0 + scale * rng.standard_normal(len(dd["d"])))
            sim = cc.run_baseline(F, st["mu"], st["params"], d2)
            m = F.measures(sim, d2)
            return {k: m.get(k) for k in ("div", "L1_whole", "L1_pre", "L1_ev", "L1_last10")} | {"lam": sim.get("lam") if not isinstance(sim.get("lam"), np.ndarray) else None}
        row["rerun"] = run(); row["rerun2"] = run()
        row["perturb_1e-12"] = run(1e-12); row["perturb_1e-9"] = run(1e-9)
        out.append(row)
        print(json.dumps(row, default=str), flush=True)
    json.dump(out, open("out/cc_repro.json", "w"), indent=1, default=str)

if __name__ == "__main__":
    main()
