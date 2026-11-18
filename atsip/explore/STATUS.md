# STATUS

- テーマ — ATSIP の P4 再始動試行とロールバック（E10）
- 実行者 — ChatGPT（Codex）
- 更新日時 — 2026-11-18 22:14:23 UTC
- 状態 — `atsip/dev` でレビュー、P4・P2・P3 のチェック、スモーク 117 試行、本実験 585 試行と集計を完了。実行コードは `f74bb94`。seed 10–14、各入力 30 秒、2 並列、BLAS 等は各 1 スレッド。
- 結果 — 誤作動・見逃し・発散を全て 0 にした規則はなかった。K = 16・M は判定済みの試行で分離幅 11.30 dB、経路変化 20/20 件を確定し、SC は 5/5 件で判定前に発散。K = 4・M は SC の発散を 0/5 件にしたが経路変化を 4 件見逃した。
- 未解決 — SC が発散する前の取り消しと、pcb の確定に必要な試行時間の両立。最終評価用 seed 100–119 は未使用。
- 次の一手 — SC を早期に取り消す条件を検討し、同じ条件で pcb の見逃しが増えないかを確かめる。
- 成果物の場所 — [RESULTS_dev3.md](RESULTS_dev3.md)、[RESULTS_dev3.csv](RESULTS_dev3.csv)、[RESULTS_dev3_trials.csv](RESULTS_dev3_trials.csv)、[REVIEW_prop4.md](REVIEW_prop4.md)、[prop4.py](prop4.py)。試行別 JSON は `out/dev3/E10/`。
