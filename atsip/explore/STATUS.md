# STATUS

- テーマ — ATSIP 論文化に向けた P2（SF の利得補正、ND 初期化、固定後半ステップ、rsN2）の検証
- 実行者 — ChatGPT（Codex）
- 更新日時 — 2026-11-18 17:18:49 JST
- 状態 — `atsip/dev` でコードレビュー、回帰チェック、スモーク318試行、本実験 E1–E7 の1,590試行と集計を完了。実行コードは `6699a56`。Python 3.12.14、NumPy 2.3.5、SciPy 1.17.0、9並列、BLAS等は各1スレッド。
- 結果 — μ_e は SF 0.5、ND 1.25、前半は fixed64。E4 では最終則と L1 whole に差がなく、ND の初秒の Φ 超過が 3.48 dB 低かった。数値発散は E5 で5件、E6 で36件、E7 で25件。E6 の1サンプル遅れでは ND・SF とも全15試行が発散した。
- 未解決 — E5 のバースト外乱と入力の色変化での再始動誤検出、E6 の遅れ・タップ誤差に対する安定性。最終評価用 seed 100–119 は未使用。
- 次の一手 — rsN2 が経路変化と誤差マイク外乱・入力の色変化を識別できる統計を検討し、同じ E5 条件で確認する。二次経路の遅れ誤差は E6 の条件で安定性を確認する。
- 成果物の場所 — [RESULTS_dev1.md](RESULTS_dev1.md)、[RESULTS_dev1.csv](RESULTS_dev1.csv)、[RESULTS_dev1_curves.csv](RESULTS_dev1_curves.csv)、[RESULTS_dev1_selection.json](RESULTS_dev1_selection.json)、[REVIEW_prop2.md](REVIEW_prop2.md)。試行別 JSON は `out/dev1/`。
