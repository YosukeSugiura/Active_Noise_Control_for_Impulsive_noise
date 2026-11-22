# 論文の構成（第3版、2026-11-22）

投稿先は ATSIP か Applied Acoustics（未定）。どちらでも使える構成にしておき、決まったら字数と体裁を合わせる。
主張と数字は THEORY_frozen.md（第8版）の「論文の目的と主張」に従う。調整の経緯は書かない。

## 題目（案）

- Fast and output-preserving reconvergence of frequency-domain filtered-x ANC under impulsive noise and primary path changes
- Background-trial restart for frequency-domain active noise control of impulsive noise

## 節立て（第3版：7 節、図 4、表 3 に絞る）

| 節 | 内容 |
|---|---|
| 1. Introduction | 背景と課題、関連研究（インパルス性向けの更新則、SF 型の更新、誤差の作り直し、二重経路法）を 1 段落ずつ、本稿との違い、貢献 3 つ |
| 2. Proposed method | 問題設定と記号、SF 方向と有限長での性質（定理 A、補題 B）、1/g（定理 C）、始動・再始動の直後の作り直し（補題 F）、裏の試行と検出器（命題 H）、擬似コード。証明は付録 |
| 3. Evaluation framework | 経路、入力、比較手法とステップの決め方、指標と統計を短く（Table 1） |
| 4. Reconvergence after a primary path change | 主結果。問い → 結果 → 読み取れること（見逃した入力の扱いも）。Fig. 3、Fig. 4、Table 2 |
| 5. Cost of the restart mechanism | 変化がないときに損をしないか：定常、誤差側の外乱、出力の同一性、各要素の寄与、演算量をまとめて（Table 3） |
| 6. Conclusion | 主張と範囲の限定 |
| Appendix | 証明、後半のステップの設計式 |

## 図表

| 番号 | 中身 | 元データ |
|---|---|---|
| Fig. 1 | 系の構成図 | 作図 |
| Fig. 2 | 提案法の流れ（前景、検出器、裏の試行、移行の判定） | 作図 |
| Fig. 3 | 騒音源の切り替え後の消音量の時間変化（提案、APSIPA 2026 版、再始動なし、FxNLMS。near と far） | out/coef2_run |
| Fig. 4 | 入力別の対応差（10 入力 × 2 条件 × 2 時点のヒートマップ） | RESULTS_coef2.md |
| Table 1 | 設定値と比較手法（選ばれたステップを含む） | THEORY_frozen.md §1、TUNE.json |
| Table 2 | 主結果：回復時間、0.13・0.51 秒後の消音量、対応差と 95% 区間 | RESULTS_final2.md、RESULTS_coef2.md |
| Table 3 | 再始動の仕組みの代償：定常・外乱の消音量、移行、出力の同一性、内部の比較、演算量 | RESULTS_final2.md、RESULTS_fast.md |

実測経路の例と実音の波形の図は、必要なら付録に回す。

## 書く前に確かめること

- 関連研究の各手法について、原論文の評価条件（一次経路の変化を扱っているか）を確かめる。
- Akhtar & Nishihara（2015）など、未確認の書誌。
- 英文は、杉浦の文体ガイド（Notion）に従う。
