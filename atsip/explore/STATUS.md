# STATUS

- テーマ — ATSIP S3（最終評価、seed 100–119、1600 条件）
- 実行者 — Codex（2 シャード並列）、集計は Claude
- 更新日時 — 2026-11-21 JST
- 状態 — 1600/1600 条件完了。EVAL_FROZEN.json のハッシュはすべて一致。
- 結果 — 回復の中央値は経路変化 4 条件すべてで外部の最良より短い。定常は同等。主条件の発散 0、誤確定 1/1400。RM-10s の発散は再始動なしより 2 件多い。詳細は THEORY_frozen.md §10 の 8。
- 未解決 — 論文の執筆。採択見込みの再評価（Claude、ChatGPT）。
- 成果物 — [最終結果](RESULTS_final.md)、results/final_run_shard{0,1}.tar.gz（各条件の JSON）
