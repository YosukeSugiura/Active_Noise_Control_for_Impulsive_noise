# Theory scripts

Scratch scripts behind THEORY_MEMO.md and DERIVATION_vss.md, moved from /tmp into the repository. Numbers only; no claims beyond the memos.

- closedform/: closed-form checks of Prop. 1-3 (base.py = shared path, Wiener and spectra; main2.py, trade.py, sf.py, hank.py). cmp.py needs a prototype output (/tmp/chk/fp_a145.npz) that is not in the repository.
- transient/: transient model and step-size rules.
  - simx.py: generic delayless FD-FxNLMS simulator with per-block step/normalizer schedules or a controller.
  - model.py: mean-square transient model (conditional on the frame gains g_n). probe.npz, meas.npz: measured coefficients and learning curves it reads.
  - probe.py, meas.py: produce probe.npz, meas.npz.
  - optsched.py, horizon.py, a025.py: model-optimal schedules.
  - vss.py, vss2.py, vss3.py: earlier proportional step controllers (SF-V2, GS-V2).
  - qchk.py, cchk.py, dof.py, gam.py: checks quoted in DERIVATION_vss.md sections 10-20.
- ../vssI.py, ../vssI2.py: integral step rules (SF-I, GS-T2), results in ../RESULTS_vssI*.md.

Run from the directory of the script (python3 model.py etc.).
