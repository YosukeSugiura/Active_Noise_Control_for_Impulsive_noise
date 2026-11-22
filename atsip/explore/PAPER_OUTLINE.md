# 論文の構成（第2版、2026-11-22）

投稿先は ATSIP か Applied Acoustics（未定）。どちらでも使える構成にしておき、決まったら字数と体裁を合わせる。
主張と数字は THEORY_frozen.md（第8版）の「論文の目的と主張」に従う。調整の経緯は書かない。

## 題目（案）

- Fast and output-preserving reconvergence of frequency-domain filtered-x ANC under impulsive noise and primary path changes
- Background-trial restart for frequency-domain active noise control of impulsive noise

## 節立て（第2版：条件・結果・考察を分けず、問いごとに 1 節にまとめる）

実験の節は、それぞれ「問い → その節に固有の条件 → 結果 → 読み取れること・限界」の順で閉じる。
共通の条件（経路、入力、比較手法、指標）は 5 節に短くまとめ、各節では差分だけを書く。

| 節 | 内容 | 根拠（ファイル） |
|---|---|---|
| 1. Introduction | インパルス性騒音の ANC と、一次経路が急に変わる場面（騒音源の切り替え）。既存の更新則は定常の頑健さを扱い、変化の後の再収束は扱わない。貢献 3 つと土台 | THEORY_frozen.md「論文の目的と主張」「貢献」 |
| 2. Related work | インパルス性騒音向けの更新則、スペクトル因子型の更新、誤差の作り直し、二重経路法と可変ステップ。各項で本稿との違いを 1 文 | 引用は原論文で評価条件を確認してから |
| 3. Problem formulation | 遅延なしブロック周波数領域 FxNLMS、記号、一次経路の変化のモデル。対象は一次経路の変化 | THEORY_frozen.md §0 |
| 4. Proposed method | 4.1 SF 方向と有限長での性質（定理 A、補題 B）。4.2 尺度で条件づけた更新と 1/g（定理 C）。4.3 始動・再始動の直後の作り直し（補題 F）。4.4 裏の試行による再始動と検出器（命題 H）。4.5 擬似コード | THEORY_frozen.md §1–§8 |
| 5. Evaluation framework | 共通の条件だけ：実測経路（MeshRIR 8 組、Ŝ 512、100 Hz 高域通過）、入力 10 種、外部 8 手法とステップの決め方、指標の定義、統計（対応差、seed の再標本化）。Table 1・2、Fig. 3 | EVAL_PLAN.md、TUNE.json |
| 6. Reconvergence after a primary path change | 問い：騒音源が切り替わった後、どれだけ速く戻るか。条件：near・far。結果：Fig. 4、Table 3。入力別：Fig. 5。読み取れること：見逃した入力（machinegun、volvo の near）では再始動なしと同じ動きになり、常時適応する APSIPA 2026 版に遅れる | RESULTS_final2.md、RESULTS_coef2.md |
| 7. Steady state and error-side disturbances | 問い：再始動の仕組みは、変化がないときに損をしないか。条件：変化なし、外乱 2 種。結果：Table 4（消音量、移行、出力の同一性）。読み取れること：破棄した試行は出力を変えない。強い外乱の後に移行することがあり、消音量は悪化しない。外乱直後は APSIPA 2026 版が良い | RESULTS_final2.md |
| 8. Role of each component | 問い：各要素は要るか。条件：内部の比較 6 種。結果：Table 5。読み取れること：検出器なしで常時試すと移行が増え、1/g なしは発散する | RESULTS_final2.md |
| 9. Computational cost | 問い：実時間で使える負荷か。Table 6。通常時は FxNLMS の約 0.6 倍、試行中は約 1.2 倍 | RESULTS_fast.md |
| 10. Conclusion | 主張を数字で 1 段落。範囲の限定（1 室 8 組、6 録音、模擬、二次経路の変化は対象外）もここで 2〜3 文 | THEORY_frozen.md「書かない主張」 |
| Appendix | 証明と、後半のステップの設計式の導出 | THEORY_frozen.md §2–§8、theory/ |

## 図表

| 番号 | 中身 | 節 | 元データ |
|---|---|---|---|
| Fig. 1 | 系の構成図（一次経路、二次経路、参照・誤差マイク、制御フィルタ） | 3 | 作図 |
| Fig. 2 | 提案法の流れ（前景、検出器、裏の試行、移行の判定） | 4.4 | 作図 |
| Fig. 3 | 実測経路の例（P、S、切り替え後の P）と、実音の波形の例 | 5 | data/paths16、data/prepared |
| Table 1 | 設定値の一覧（M、N、B、ステップ、しきい値、保護区間など） | 5 | THEORY_frozen.md §1 |
| Table 2 | 比較手法と選ばれたステップ | 5 | TUNE.json |
| Fig. 4 | 騒音源の切り替え後の消音量の時間変化（提案、APSIPA 2026 版、再始動なし、FxNLMS。near と far） | 6 | out/coef2_run（中央値と四分位） |
| Table 3 | 主結果：回復時間の中央値、0.13・0.51 秒後の消音量、対応差と 95% 区間（near、far） | 6 | RESULTS_final2.md、RESULTS_coef2.md |
| Fig. 5 | 入力別の対応差（10 入力 × 2 条件 × 2 時点のヒートマップ、区間が 0 をまたぐものを区別） | 6 | RESULTS_coef2.md |
| Table 4 | 定常と外乱：30 秒全体・末尾・直後 2 秒の消音量、移行、出力の同一性の件数 | 7 | RESULTS_final2.md |
| Table 5 | 内部の比較：再始動なし、1/g なし、すぐ鳴らす版、常時試す版、常に作り直す版、SF 固定 | 8 | RESULTS_final2.md |
| Table 6 | 演算量（通常・前半・試行中の 1 ブロックあたりの乗算、FxNLMS との比） | 9 | RESULTS_fast.md |

## 書く前に確かめること

- 関連研究の各手法について、原論文の評価条件（一次経路の変化を扱っているか）を確かめる。
- Akhtar & Nishihara（2015）など、未確認の書誌。
- 英文は、杉浦の文体ガイド（Notion）に従う。
