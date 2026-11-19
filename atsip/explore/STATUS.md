# STATUS

- テーマ — ATSIP の凍結 P4 と裏試行 P5 の比較（E12）
- 実行者 — Codex
- 更新日時 — 2026-11-19 08:32:28 UTC
- 状態 — `atsip/dev` でコードレビュー、P2–P5 のチェック、smoke 140試行、本実験700試行と集計を完了。実行コードは `7a07ad6`。モデル一致65入力とモデル誤差あり75入力、seed 10–14、各30秒、2並列。
- 結果 — P4・P5・P5-early は全モデル条件で誤確定・見逃し・発散が0件。モデル一致の経路変化では P4 の直後2秒の L1 が P5 より0.45–0.95 dB良い。EN・SC では P5 が20入力すべてで再始動なしのマイク誤差と Φ 超過を完全に再現した。
- 未解決 — 回復指標はモデル一致の経路変化15入力のうち各規則8入力で250ブロックの上限に達した。最終評価用 seed 100–119 は未使用。
- 次の一手 — 経路変化の直後性能と外乱時のピークを比較し、最終評価に用いる規則と回復時間の観測窓を決める。
- 成果物の場所 — [RESULTS_dev5.md](RESULTS_dev5.md)、[RESULTS_dev5.csv](RESULTS_dev5.csv)、[RESULTS_dev5_trials.csv](RESULTS_dev5_trials.csv)、[REVIEW_prop5.md](REVIEW_prop5.md)、[prop5.py](prop5.py)。実行ログは `out/dev5/E12/`。
