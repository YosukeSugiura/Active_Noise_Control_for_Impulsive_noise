# STATUS

- テーマ — ATSIP 最終評価 S0b の経路と実音の追加
- 更新日時 — 2026-11-19 12:07:23 UTC
- 状態 — 8組の paths16、白色入力の M=512 上限、実音素材と台帳を整備。評価の実行と seed 100–119 の使用はなし。
- 結果 — 全8組で P_near・P_far を作成。M=512 の上限は P が −13.35〜−12.31 dB、near が −14.10〜−11.91 dB、far が −16.68〜−14.30 dB。MIMII valve と SONYC-UST jackhammer を各40秒に整えた。
- 未解決 — 上限は白色入力の計算値。接続済み実音では継ぎ目の扱いを後段で決める。
- 次の一手 — S1 で paths16 と実音台帳の読み込みを実装する。
- 成果物の場所 — [報告](DATA_REPORT.md)、[経路台帳](data/paths16/paths16.json)、[上限](data/feasibility16.out)、[実音台帳](data/noise.csv)、[再生成手順](data/README.md)。
