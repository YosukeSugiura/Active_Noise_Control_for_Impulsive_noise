# STATUS

- テーマ — ATSIP 最終評価 S0 のデータ準備
- 更新日時 — 2026-11-19 11:11:31 UTC
- 状態 — データ取得、経路と実音の生成、記述統計を完了。評価実行と最終評価用 seed の使用はなし。
- 結果 — MeshRIR から重複のない 8 組を選択し、各組の5・10・20・40 cmの移動後の経路を整備。NOISEX-92 は40秒4本。MIMII valve と UrbanSound8K は短い元クリップと統計を取得。
- 未解決 — valve・jackhammer・engine_idling の40秒連続素材。NOISEX-92 の元音声の再配布許諾。MeshRIR の共通遅れの扱い。
- 次の一手 — S1 で経路の組、因果性の余裕、共通遅れ、実音の採用範囲を決める。
- 成果物の場所 — [データ報告](DATA_REPORT.md)、[取得・再生成手順](data/README.md)、[実音台帳](data/noise.csv)、[経路一覧](data/paths/paths.json)。取得物は `data/raw/`、生成 WAV は `data/prepared/`。
