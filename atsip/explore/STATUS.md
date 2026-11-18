# STATUS

- テーマ — ATSIP 版 ANC の可変ステップ則 protoC の仕様固定と修正
- 実行者 — ChatGPT（Codex）
- 更新日時 — 2026-11-18 12:39:01 JST
- 状態 — `atsip/fix0` のコード修正、回帰テスト、実験、比較を完了
- 結果 — `check --preset final` が合格。8 入力 × 3 seed × 5 則の 120 ケースで数値発散 0 件、増幅 0 件。基準コミットの再実行と、比較対象の dB 指標が丸め前まで一致。提案則は経路変化 9 件すべてを検出し、誤作動・見逃し・打ち切りは各 0 件。
- 未解決 — この修正範囲の未解決項目なし。別 seed での最終評価は未実施。
- 次の一手 — 最終評価では `--preset final` と別 seed を使う。
- 成果物の場所 — [RESULTS_fix0.md](RESULTS_fix0.md)、[RESULTS_fix0.csv](RESULTS_fix0.csv)、[実行手順](RUN_protoC.md)。各タグの JSON と NPZ は実行環境の `out/fix0/`。
