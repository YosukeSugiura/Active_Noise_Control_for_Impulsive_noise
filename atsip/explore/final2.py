"""Final evaluation with the revised detector (baseline frozen from the trigger block to the end of the guard).

Everything else is final.py / coef.py unchanged: the detector class is replaced before the rules are built, the
conditions are the five of the paper (no change, source switch near/far, error-side disturbance 0 / -10 dB), and the
outputs go to separate folders and files.

  python final2.py final [--dry] [--procs K] [--shard k/n]     -> out/final2_run (or final2_dry)
  python final2.py final-report [--dry]                        -> RESULTS_final2.md (or _dry)
  python final2.py coef [--dry] [--procs K] [--shard k/n]      -> out/coef2_run (or coef2_dry)
  python final2.py coef-report [--dry]                         -> RESULTS_coef2.md (or _dry)
"""
import argparse, os, shutil
from multiprocessing import Pool
from pathlib import Path
import final as F
import coef as C
import det2

HERE = Path(__file__).resolve().parent
F.p.RSNDetector = det2.FreezeDetector          # every rule with a detector (proposed, 1/g-free, immediate) uses it
F.FINAL_SCEN = ("stat", "RS-near", "RS-far", "EN0", "ENm10")
F.final_dir = lambda dry: HERE / "out" / ("final2_dry" if dry else "final2_run")
C.SCEN = ("RS-near", "RS-far")
C.out_dir = lambda dry: HERE / "out" / ("coef2_dry" if dry else "coef2_run")


def report_to(fn, dry, old, new):
    """Run a report function of final.py / coef.py that writes `old`, and store its output as `new` instead."""
    old, new, keep = HERE / old, HERE / new, HERE / (old + ".keep")
    had = old.exists()
    if had:
        shutil.copy2(old, keep)
    try:
        fn(dry)
        shutil.move(old, new)
    finally:
        if had:
            shutil.move(keep, old)
    print("stored as", new.name)


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("cmd", choices=("final", "final-report", "coef", "coef-report"))
    a.add_argument("--dry", action="store_true"); a.add_argument("--procs", type=int, default=os.cpu_count() or 1); a.add_argument("--shard", default="0/1")
    o = a.parse_args(); sfx = "_dry" if o.dry else ""
    k, n = (int(v) for v in o.shard.split("/"))
    if o.cmd == "final":
        F.final_run(o.procs, o.dry, (k, n))
    elif o.cmd == "final-report":
        report_to(F.final_report, o.dry, f"RESULTS_final{sfx}.md", f"RESULTS_final2{sfx}.md")
    elif o.cmd == "coef":
        jobs = [j for i, j in enumerate(C.jobs(o.dry)) if i % n == k]
        with Pool(o.procs) as pool:
            list(pool.imap_unordered(C.run_job, jobs))
    else:
        report_to(C.report, o.dry, f"RESULTS_coef{sfx}.md", f"RESULTS_coef2{sfx}.md")
