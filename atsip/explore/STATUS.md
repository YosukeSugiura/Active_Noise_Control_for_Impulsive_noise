# STATUS

- テーマ — ATSIP 最終則の前半ステップ、1/g、推定器、再始動の寄与の切り分け
- 実行者 — ChatGPT（Codex）
- 更新日時 — 2026-11-18 14:05:55 JST
- 状態 — ローカルの `atsip/fix0`（基点 `b7fadef`）で短い試行240ケース、本試行1,200ケース、集計および回帰確認を完了。
- 結果 — A3 の最良 μ は 0.02。A6 との L1 whole は8クラス中7クラスで差なし（0.1 dB未満）。pc は A6 が全体で0.121 dB、15–17秒で0.847 dB改善。A7 の数値発散4件は a145・a120・pc・pcb の seed 12。A6 の数値発散は0件、全1,200ケースの増幅も0件。A6 の再始動は3つの経路変化窓でそれぞれ5/5 seedを改善し、全15変化を検出した。
- 未解決 — `RESULTS_nodelay.csv` はローカルに未配置。最終評価用 seed 100–119 は未使用。
- 次の一手 — `RESULTS_nodelay.csv` を `atsip/explore/` に置き、このディレクトリで `python ablate.py report` を再実行する。別ブランチが後日リモートに用意された場合は、リポジトリの最上位で `git fetch origin atsip/nodelay`、続いて `git checkout origin/atsip/nodelay -- atsip/explore/RESULTS_nodelay.csv` でCSVだけを取得できる。
- 成果物の場所 — [RESULTS_ablate.md](RESULTS_ablate.md)、[RESULTS_ablate.csv](RESULTS_ablate.csv)、[ablate.py](ablate.py)。試行別JSONは `out/ablate/`。Git bundle はリポジトリ最上位の `fix0.bundle`。
