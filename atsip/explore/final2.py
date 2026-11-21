"""Final evaluation with the revised detector (baseline frozen from the trigger block to the end of the guard).

Everything else is final.py / coef.py unchanged: the detector class is replaced before the rules are built, the
conditions are the five of the paper (no change, source switch near/far, error-side disturbance 0 / -10 dB), and the
outputs go to separate folders and files.

  python final2.py final [--dry] [--procs K] [--shard k/n]     -> out/final2_run (or final2_dry)
  python final2.py final-report [--dry]                        -> RESULTS_final2.md (or _dry)
  python final2.py coef [--dry] [--procs K] [--shard k/n]      -> out/coef2_run (or coef2_dry)
  python final2.py coef-report [--dry]                         -> RESULTS_coef2.md (or _dry)
"""
import argparse, os
from tempfile import TemporaryDirectory
from multiprocessing import Pool
from pathlib import Path
import final as F
import coef as C
import det2

HERE = Path(__file__).resolve().parent

def configure():
    """Install the evaluator settings in the parent and in every worker."""
    F.p.RSNDetector = det2.FreezeDetector
    F.FINAL_SCEN = ("stat", "RS-near", "RS-far", "EN0", "ENm10")
    F.final_dir = lambda dry: HERE / "out" / ("final2_dry" if dry else "final2_run")
    C.SCEN = ("RS-near", "RS-far")
    C.out_dir = lambda dry: HERE / "out" / ("coef2_dry" if dry else "coef2_run")


def report_to(module, fn, dry, old, new):
    """Use the original report writer in a scratch directory."""
    root = module.HERE
    with TemporaryDirectory(dir=HERE, prefix=".final2-report-") as scratch:
        module.HERE = Path(scratch)
        try:
            fn(dry)
            (module.HERE / old).replace(HERE / new)
        finally:
            module.HERE = root
    print("stored as", new)


if __name__ == "__main__":
    configure()
    a = argparse.ArgumentParser(); a.add_argument("cmd", choices=("final", "final-report", "coef", "coef-report"))
    a.add_argument("--dry", action="store_true"); a.add_argument("--procs", type=int, default=os.cpu_count() or 1); a.add_argument("--shard", default="0/1", type=F.parse_shard)
    o = a.parse_args(); sfx = "_dry" if o.dry else ""
    if o.cmd in ("final", "coef"):
        if not o.dry and not (HERE / "EVAL_FROZEN.json").is_file():
            a.error("EVAL_FROZEN.json is required for the final run")
        jobs, worker = ((F.final_jobs(o.dry), F.run_final_job) if o.cmd == "final"
                        else (C.jobs(o.dry), C.run_job))
        with Pool(o.procs, initializer=configure) as pool:
            list(pool.imap_unordered(worker, F.shard_jobs(jobs, o.shard)))
    elif o.cmd == "final-report":
        report_to(F, F.final_report, o.dry, f"RESULTS_final{sfx}.md", f"RESULTS_final2{sfx}.md")
    else:
        report_to(C, C.report, o.dry, f"RESULTS_coef{sfx}.md", f"RESULTS_coef2{sfx}.md")
