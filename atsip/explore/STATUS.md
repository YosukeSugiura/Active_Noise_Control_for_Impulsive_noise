# STATUS

- テーマ — ATSIP の P4 再始動試行と早期取り消し（E11）
- 実行者 — Codex
- 更新日時 — 2026-11-19 07:49:51 UTC
- 状態 — `atsip/dev` でコードレビュー、8チェック、smoke 91試行、本実験455試行と集計を完了。実行コードは `75e6a68`。13クラス、seed 10–14、各30秒、2並列、BLAS・OMP・MKL は各1スレッド。
- 結果 — K=16・窓B・abort 4、K=16・窓B・abort 2、K=16・窓M・abort 4 は全65入力で誤作動・見逃し・発散がすべて0。窓Bの早期取り消しは SC 全5件で検出の2–3ブロック後に働いた。K=12・窓B・abort 4 は pcb で1件見逃した。
- 未解決 — EN では abort 2・4 とも早期取り消しがなく、K=16 のイベント直後2秒の Φ 超過は再始動なしに対し4.47–5.29 dB悪化した。最終評価用 seed 100–119 は未使用。
- 次の一手 — 今回の3条件を満たした規則を比較し、最終評価に使う設定を決める。
- 成果物の場所 — [RESULTS_dev4.md](RESULTS_dev4.md)、[RESULTS_dev4.csv](RESULTS_dev4.csv)、[RESULTS_dev4_trials.csv](RESULTS_dev4_trials.csv)、[REVIEW_prop4.md](REVIEW_prop4.md)、[prop4.py](prop4.py)、[vssT_lib.py](vssT_lib.py)。実行ログは `out/dev4/E11/`。
