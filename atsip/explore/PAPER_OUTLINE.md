# 論文の構成（第1版、2026-11-22）

投稿先は ATSIP か Applied Acoustics（未定）。どちらでも使える構成にしておき、決まったら字数と体裁を合わせる。
主張と数字は THEORY_frozen.md（第8版）の「論文の目的と主張」に従う。調整の経緯は書かない。

## 題目（案）

- Fast and output-preserving reconvergence of frequency-domain filtered-x ANC under impulsive noise and primary path changes
- Background-trial restart for frequency-domain active noise control of impulsive noise

## 節立て

| 節 | 内容 | 根拠（ファイル） |
|---|---|---|
| 1. Introduction | インパルス性騒音の ANC と、一次経路が急に変わる場面（騒音源の切り替え）。既存の更新則は定常の頑健さを扱い、変化の後の再収束は扱わない。貢献 3 つと土台 | THEORY_frozen.md「論文の目的と主張」「貢献」 |
| 2. Related work | (a) インパルス性騒音向けの更新則（クリップ、対数、M 推定、FxNFLOC、EVSS など）。(b) 周波数領域・スペクトル因子型の更新（Elliott & Rafaely）。(c) 現在の係数で誤差を作り直す更新（modified FxLMS）。(d) 二重経路法（エコーキャンセラ）と可変ステップ。各項で本稿との違いを 1 文 | 引用は原論文で評価条件を確認してから |
| 3. Problem formulation | 遅延なしブロック周波数領域 FxNLMS（M、N、B）、記号、一次経路の変化のモデル。対象は一次経路の変化（二次経路の変化と Ŝ の再同定は対象外） | THEORY_frozen.md §0 |
| 4. Proposed method | 4.1 SF 方向と有限長での性質（定理 A、補題 B、後半のステップの設計式）。4.2 尺度で条件づけた更新と 1/g（定理 C）。4.3 始動・再始動の直後だけ誤差フレームを作り直す（補題 F、g の下限 0.55）。4.4 裏の試行による再始動と検出器（命題 H、移行の判定、保護区間の間は平均を止める）。4.5 アルゴリズムのまとめ（擬似コード）と演算量 | THEORY_frozen.md §1–§8、RESULTS_fast.md |
| 5. Experimental setup | 経路（MeshRIR、8 組、16 kHz、100 Hz 高域通過、Ŝ 512 タップ）。入力（合成 4 種、実音 6 種）。条件（変化なし、騒音源の切り替え near・far、誤差側の外乱 2 種）。比較手法（外部 8、ステップは格子探索で全条件共通に固定）。指標（回復時間、変化から 0.13・0.51 秒後の消音量、30 秒全体と末尾の消音量、移行・見逃し・発散）。seed と統計（対応差、seed の再標本化による 95% 区間） | EVAL_PLAN.md、TUNE.json、data/README.md |
| 6. Results | 6.1 騒音源の切り替え後の再収束。6.2 入力別の結果。6.3 定常の性能。6.4 誤差側の外乱と出力の同一性。6.5 各要素の寄与（内部の比較）。6.6 演算量 | RESULTS_final2.md、RESULTS_coef2.md、RESULTS_fast.md |
| 7. Discussion | 見逃しが残る入力（machinegun、volvo の near）と、その場合は再始動なしと同じ動きになること。強い外乱の後の移行。回復時間の基準が手法ごとであること。一般化の範囲（1 室 8 組、6 録音）。実機、参照経路、音響フィードバック | THEORY_frozen.md「書かない主張」 |
| 8. Conclusion | 主張 3 つを数字で 1 段落 | — |
| Appendix | 定理 A・補題 B・定理 C・補題 F・命題 H の証明、後半のステップの設計式の導出 | THEORY_frozen.md §2–§8、theory/ |

## 図表

| 番号 | 中身 | 節 | 元データ |
|---|---|---|---|
| Fig. 1 | 系の構成図（一次経路、二次経路、参照・誤差マイク、制御フィルタ） | 3 | 作図 |
| Fig. 2 | 提案法の流れ（前景、検出器、裏の試行、移行の判定） | 4.4 | 作図 |
| Fig. 3 | 実測経路の例（P、S、切り替え後の P）と、実音の波形の例 | 5 | data/paths16、data/prepared |
| Table 1 | 設定値の一覧（M、N、B、ステップ、しきい値、保護区間など） | 5 | THEORY_frozen.md §1 |
| Table 2 | 比較手法と選ばれたステップ | 5 | TUNE.json |
| Fig. 4 | 騒音源の切り替え後の消音量の時間変化（提案、APSIPA 2026 版、再始動なし、FxNLMS。near と far） | 6.1 | out/coef2_run（中央値と四分位） |
| Table 3 | 主結果：回復時間の中央値、0.13・0.51 秒後の消音量、対応差と 95% 区間（near、far） | 6.1 | RESULTS_final2.md、RESULTS_coef2.md |
| Fig. 5 | 入力別の対応差（10 入力 × 2 条件 × 2 時点のヒートマップ、区間が 0 をまたぐものを区別） | 6.2 | RESULTS_coef2.md |
| Table 4 | 定常と外乱：30 秒全体・末尾・直後 2 秒の消音量、移行、出力の同一性の件数 | 6.3–6.4 | RESULTS_final2.md |
| Table 5 | 内部の比較：再始動なし、1/g なし、すぐ鳴らす版、常時試す版、常に作り直す版、SF 固定 | 6.5 | RESULTS_final2.md |
| Table 6 | 演算量（通常・前半・試行中の 1 ブロックあたりの乗算、FxNLMS との比） | 6.6 | RESULTS_fast.md |

## 書く前に確かめること

- 関連研究の各手法について、原論文の評価条件（一次経路の変化を扱っているか）を確かめる。
- Akhtar & Nishihara（2015）など、未確認の書誌。
- 英文は、杉浦の文体ガイド（Notion）に従う。
