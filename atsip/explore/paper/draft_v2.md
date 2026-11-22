# Draft v1 (abstract and introduction)

[..] は著者用のメモ。投稿前に消す。

## 段落ごとの論理（日本語）

| 段落 | 主張 | 根拠 | 前とのつながり |
|---|---|---|---|
| 1 | インパルス性の標本は参照にも誤差にも入り、FxLMS の更新を大きくして発散させうる | 更新が「フィルタ済み参照 × 誤差」に比例すること | 背景 |
| 2 | 既存の頑健な FxLMS は、参照と誤差の大きな標本の影響を抑えて安定を保つ | 文献（1 研究 1 文、著者を主語） | 1 の問題への既存の対処 |
| 3 | 一次経路の急変でも誤差が大きくなり、同じ抑制が更新を小さくして再収束が遅れる | 抑制の仕組み＋本評価で既存の最良が約 1 秒 | 2 の対処の限界 |
| 4 | ステップを大きくすれば速いが、インパルスでも誤差は増え、誤差の大きさを見る検出器は両方に反応する。そこで検出は試行の開始だけに使い、誤差が下がった場合だけ結果を採る | 1 と 3 | 3 への方針 |
| 5 | 前景と背景の仕組み。破棄した試行は出力を変えない。誤って確定した移行は出力を変えうるので、その件数を報告する | 構成（命題 H） | 4 の方針の具体化 |
| 6 | 背景を速く収束させる 2 要素と、その解析。二重フィルタ自体は既存 | 定理 C、補題 F、Ochiai 1977 | 5 が成り立つ条件 |
| 7 | 評価の範囲と結果（例外を含む）、構成 | 最終評価 | まとめ |

## Abstract

Adaptive algorithms for active noise control limit the influence of impulsive reference and error samples on the update. This limitation keeps the control filter stable under impulsive noise. However, a sudden primary path change also raises the error. The error stays large until the control filter reconverges. Limiting updates during this period slows reconvergence. We propose a frequency-domain filtered-x algorithm that tests a larger step size in a separate filter. The filter that generates the control signal, the foreground filter, uses its normal step size during this trial. The trial starts when the ratio of error power to filtered-reference power exceeds a threshold proportional to its running average on consecutive blocks. The algorithm uses the foreground coefficients to initialize the separate filter, called the background filter. The algorithm transfers the background coefficients to the foreground filter only if they yield sufficiently lower error power on the same data. We show that discarding a trial gives the same control signal as disabling it, given the same initial state and input. The background update takes its direction from a spectral factor of the filtered-reference power spectrum. It normalizes the step size by relative reference-frame power. The algorithm recomputes error frames with the current coefficients during start-up and after each transfer. We evaluated the algorithm in simulations with measured room impulse responses and ten noise types. After a noise-source switch, we observed median recovery times 63% shorter for a nearby new source and 68% shorter for a distant new source than for the fastest of eight existing algorithms. However, the fastest existing algorithm had a shorter median recovery time for machinegun noise in both conditions and for car interior noise in the near condition. With a fixed primary path and no added disturbance, the two algorithms differed by less than 0.1 dB in mean noise reduction.

[303 語（空白区切り）。63% = 1 − 0.36/0.98、68% = 1 − 0.40/1.26。0.1 dB 未満：30 秒全体 0.05 dB、末尾 10 秒 0.07 dB。]

## 1. Introduction

Impulsive noise contains short bursts with amplitudes well above the background level. Its amplitude distribution has heavier tails than a Gaussian distribution and is often modeled by a symmetric α-stable distribution [Shao and Nikias 1993]. In feedforward active noise control (ANC), the filtered-x least-mean-square (FxLMS) algorithm updates the control filter in proportion to the product of the filtered reference signal and the error signal [Kuo and Morgan 1996]. An impulsive sample in either signal can therefore produce a large update. Such an update can cause the control filter to diverge.

Robust variants of the FxLMS algorithm address this problem by limiting the influence of large samples in the reference and error signals. Sun et al. [Sun 2006] ignored samples above a threshold. Akhtar and Mitsuhashi [Akhtar 2009] clipped the reference and error samples at their respective thresholds. Wu et al. [Wu 2011] minimized the squared logarithm of the error. This cost grows slowly for large errors. Wu and Qiu [Wu and Qiu 2013] used an M-estimator in place of the squared error. Sun et al. [Sun et al. 2015] also used an M-estimator. Feng et al. [Feng et al. 2024] used a fractional lower-order statistic in place of the squared error. These variants keep the control filter stable under impulsive noise. Section 2 reviews these algorithms.

These variants limit the update for large samples regardless of their cause. However, a sudden primary path change also increases the error. A switch to another noise source can cause such a change. The error then stays large until the control filter converges to the new optimal coefficients. By limiting the update while the error remains large, these variants slow reconvergence. In simulations with measured room impulse responses, we observed a median recovery time of about 1 s after a noise-source switch for the fastest of eight existing algorithms (Section 6).

A larger step size can speed up reconvergence. An increase in error power can indicate a primary path change. However, an impulse also raises error power. A larger step size during an impulse can degrade the control filter or cause it to diverge. The proposed algorithm therefore uses an increase in error power as a trigger to test a larger step size. It tests that step size in a separate filter and adopts the resulting coefficients only if they reduce the error sufficiently. We call this test a trial.

The proposed algorithm uses two filters with the same structure. Only one filter generates the control signal. We call it the foreground filter. It uses the normal step size during a trial. The second filter adapts during the trial and is called the background filter. The trial starts when the ratio of error power to filtered-reference power exceeds a threshold proportional to its running average on consecutive blocks. The algorithm copies the foreground coefficients into the background filter. It then adapts those coefficients with the larger step size. The algorithm transfers the background coefficients to the foreground filter only if the background filter yields sufficiently lower error power on the same samples. For this comparison, the algorithm reconstructs both error signals from the measured reference and error signals using the secondary-path model. The algorithm discards the trial if the error comparison fails to justify a transfer within a fixed number of blocks. We show that a discarded trial yields the same control signal as a run with the trial disabled, given the same initial state and input. However, the transfer criterion can also be met while the primary path remains fixed. A transfer in this case replaces the foreground coefficients and can affect the control signal. We report the frequency of these transfers.

To meet the transfer criterion within the trial window, the background filter must reduce the error quickly. We analyze two elements of the update for finite block lengths. First, the update direction uses a spectral factor of the filtered-reference power spectrum [Elliott and Rafaely 2000]. The algorithm measures the power of each reference frame relative to its running average. It divides the step size by this ratio, with a lower bound on the denominator. For the isolated-impulse analysis, we hold the coefficients, spectral factor, and running power estimate fixed. We assume that the error signal contains no independent disturbance. Under these assumptions, we show that the normalized update is independent of the reference impulse amplitude when the relative frame power exceeds its lower bound. The algorithm therefore retains impulsive samples in the update without clipping them. Second, the algorithm recomputes error frames using the current coefficients during start-up and immediately after a transfer, as in the modified FxLMS algorithm [ref]. We derive the resulting update. We determine the range of stable step sizes numerically. Ochiai and colleagues [Ochiai et al. 1977] proposed the foreground and background filter structure for echo cancellation. We apply this structure to reconvergence in ANC under impulsive noise. We analyze and evaluate the resulting algorithm.

We evaluated the algorithm in closed-loop simulations using impulse responses measured in one room [MeshRIR]. The inputs comprised ten noise types, including four synthetic signals and six recordings. After a switch to a source near the original one, we observed a median recovery time of 0.36 s, compared with 0.98 s for the existing algorithm with the shortest median among the eight tested. For a distant new source, the respective medians were 0.40 s and 1.26 s. However, the fastest existing algorithm had a shorter median recovery time for machinegun noise in both conditions and for car interior noise in the near condition. With a fixed primary path and no added disturbance, the two algorithms differed by less than 0.1 dB in mean noise reduction over the full 30 s and over the final 10 s.

Section 2 reviews related work. Section 3 formulates the problem. Section 4 describes the proposed algorithm and its analysis. Section 5 describes the evaluation framework. Section 6 reports recovery after a primary path change. Section 7 evaluates the cost of the trial mechanism with the primary path held fixed. Section 8 concludes the paper.

## 文献（確認状況）

| 本文の表記 | 書誌 | 確認 |
|---|---|---|
| Shao and Nikias 1993 | M. Shao, C. L. Nikias, "Signal processing with fractional lower order moments: stable processes and their applications," Proc. IEEE, 81(7), 1993 | 題名を確認。巻号・頁は未照合 |
| Kuo and Morgan 1996 | S. M. Kuo, D. R. Morgan, Active Noise Control Systems: Algorithms and DSP Implementations, Wiley, 1996 | 未照合 |
| Sun 2006 | X. Sun, S. M. Kuo, G. Meng, "Adaptive algorithm for active control of impulsive noise," J. Sound Vib., 291(1–2), 516–522, 2006, doi:10.1016/j.jsv.2005.06.011 | 確認。しきい値を超えた標本を無視する方式（Mirza et al. 2016 の記述）。巻号は要照合 |
| Akhtar 2009 | M. T. Akhtar, W. Mitsuhashi, "Improving performance of FxLMS algorithm for active noise control of impulsive noise," J. Sound Vib., 327(3–5), 647–656, 2009, doi:10.1016/j.jsv.2009.07.023 | 確認 |
| Wu 2011 | L. Wu, H. He, X. Qiu, "An active impulsive noise control algorithm with logarithmic transformation," IEEE Trans. Audio Speech Lang. Process., 19(4), 1041–1044, 2011, doi:10.1109/TASL.2010.2061227 | 確認 |
| Wu and Qiu 2013 | L. Wu, X. Qiu, "An M-estimator based algorithm for active impulse-like noise control," Appl. Acoust., 74, 407–412, 2013 | リポジトリの README のみ |
| Sun et al. 2015 | G. Sun, M. Li, T. C. Lim, "Enhanced filtered-x least mean M-estimate algorithm for active impulsive noise control," Appl. Acoust., 90, 31–41, 2015 | リポジトリの README のみ |
| Feng et al. 2024 | P. Feng, Z.-Y. Wang, H. C. So, "Projection FxLMS framework of active noise control against impulsive noise environments," Signal Process., 225, 109624, 2024 | 確認 |
| Elliott and Rafaely 2000 | S. J. Elliott, B. Rafaely, "Frequency-domain adaptation of causal digital filters," IEEE Trans. Signal Process., 48(5), 1354–1364, 2000, doi:10.1109/78.839982 | 確認 |
| modified FxLMS | 未定（Bjarnason 1992 は確認できず） | 要調査 |
| Ochiai et al. 1977 | K. Ochiai, T. Araseki, T. Ogihara, "Echo canceler with two echo path models," IEEE Trans. Commun., 25(6), 589–595, 1977, doi:10.1109/TCOM.1977.1093869 | 確認 |
| MeshRIR | 要記入 | — |

[関連研究（2 章）で扱う候補：凸結合の手法（Ferrer et al. 2013 TASLP 21(1)、Cheng et al. 2022 Appl. Acoust. 186、Yu et al. 2020 JLFNVAC 39）。2 つのフィルタの出力を混ぜて鳴らすので、試行が出力に影響する。この点で提案法と区別できる。]

## 変更表

| 元の文 | 直した文 | 理由 |
|---|---|---|
| Adaptive algorithms for active control of impulsive noise limit the influence of large samples in the reference and error signals. | Adaptive algorithms for active noise control limit the influence of impulsive reference and error samples on the update. | 大きな標本の影響が現れる先を明示。 |
| However, a sudden change in the primary path also increases the error, and the error stays large until the control filter reconverges. | However, a sudden primary path change also raises the error. The error stays large until the control filter reconverges. | 経路変化の影響と誤差が続く期間を別の文で説明。 |
| The same limitation reduces the coefficient update during this period, and the reconvergence becomes slow. | Limiting updates during this period slows reconvergence. | 更新の制限から再収束の遅れまでを直接述べる。 |
| This paper proposes a frequency-domain filtered-x algorithm that adapts a background filter with a larger step size and adopts its coefficients only when they reduce the error. | We propose a frequency-domain filtered-x algorithm that tests a larger step size in a separate filter. | 試行の仕組みを先に述べ、制御信号と採用条件を後続の文で説明。 |
| The foreground filter generates the control signal and is updated with the normal step size. | The filter that generates the control signal, the foreground filter, uses its normal step size during this trial. | 前景フィルタを役割で定義し、通常のステップサイズを使う期間を試行中に限定。 |
| When the ratio of the error power to the filtered-reference power exceeds its running average, the background filter is initialized with the foreground coefficients. | The trial starts when the ratio of error power to filtered-reference power exceeds a threshold proportional to its running average on consecutive blocks. The algorithm uses the foreground coefficients to initialize the separate filter, called the background filter. | 凍結仕様に合わせてしきい値と連続判定を明確にし、検出と初期化を分ける。 |
| Its coefficients are transferred to the foreground filter only when they give a sufficiently smaller error on the same data. | The algorithm transfers the background coefficients to the foreground filter only if they yield sufficiently lower error power on the same data. | 移行を能動態で書き、比較対象の指標を明示。 |
| Otherwise, the background coefficients are discarded, and the control signal is identical to that obtained without the background filter under the same initial state and input. | We show that discarding a trial gives the same control signal as disabling it, given the same initial state and input. | 命題 H の結果に合わせ、同じ初期状態と入力から開始する試行単位の比較と明示。 |
| To make the background filter converge quickly, the update uses a spectral-factor direction and a step size normalized by the input power of each block. | The background update takes its direction from a spectral factor of the filtered-reference power spectrum. It normalizes the step size by relative reference-frame power. | 更新方向とステップサイズの尺度補正を分け、スペクトル因子の対象と尺度の単位を凍結仕様に合わせる。 |
| In addition, the error frames are recomputed with the current coefficients after start-up and after each transfer. | The algorithm recomputes error frames with the current coefficients during start-up and after each transfer. | 再計算を能動態にし、適用時期を明確にする。 |
| Compared with the best of eight existing algorithms, the proposed algorithm reduced the median recovery time after a switch of the noise source by 63% for a nearby new source and by 68% for a distant new source. | After a noise-source switch, we observed median recovery times 63% shorter for a nearby new source and 68% shorter for a distant new source than for the fastest of eight existing algorithms. | 評価で観察した結果と明示し、音源切り替えの条件を比較に結び付ける。 |
| For machinegun noise in both conditions and for car interior noise with a nearby new source, the existing algorithm recovered faster. | However, the fastest existing algorithm had a shorter median recovery time for machinegun noise in both conditions and for car interior noise in the near condition. | 直前の集計結果につなぎ、例外でも回復時間の中央値を指標として明示。 |
| Without a path change, the noise reduction of the two algorithms differed by less than 0.1 dB. | With a fixed primary path and no added disturbance, the two algorithms differed by less than 0.1 dB in mean noise reduction. | 0.1 dB 未満という記述を、平均値を比較した変化なしの条件に限定。 |
| 約 265 語。 | 303 語（空白区切り）。 | 改稿後の要旨を実測して著者用メモを更新。 |
| Impulsive noise contains short bursts whose amplitudes are much larger than the background level. | Impulsive noise contains short bursts with amplitudes well above the background level. | 関係詞節を省き、バーストの特徴を簡潔に述べる。 |
| An impulsive sample in either signal therefore produces a large update, and such an update can make the control filter diverge. | An impulsive sample in either signal can therefore produce a large update. Such an update can cause the control filter to diverge. | 積の一方が大きい場合の更新と発散の可能性を別々に述べる。 |
| To prevent this divergence, robust variants of the FxLMS algorithm limit the influence of large samples in the reference and error signals. | Robust variants of the FxLMS algorithm address this problem by limiting the influence of large samples in the reference and error signals. | 前段の発散リスクを受けて、既存手法の対処を説明。 |
| Sun et al. [Sun 2006] ignored the samples that exceeded a threshold. | Sun et al. [Sun 2006] ignored samples above a threshold. | しきい値条件を短く表現。 |
| Akhtar and Mitsuhashi [Akhtar 2009] replaced such samples in the reference and error signals with the threshold values. | Akhtar and Mitsuhashi [Akhtar 2009] clipped the reference and error samples at their respective thresholds. | サンプルをしきい値に置き換える操作を、クリッピングとして具体的に示す。 |
| Other studies replaced the squared error with an M-estimator [Wu and Qiu 2013, Sun et al. 2015] or with a fractional lower-order statistic [Feng et al. 2024]. | Wu and Qiu [Wu and Qiu 2013] used an M-estimator in place of the squared error. Sun et al. [Sun et al. 2015] also used an M-estimator. Feng et al. [Feng et al. 2024] used a fractional lower-order statistic in place of the squared error. | 各研究を著者が主語の一文に分け、各引用を対応する手法に置く。 |
| Section 2 reviews them in more detail. | Section 2 reviews these algorithms. | 代名詞の参照先を明確にする。 |
| These variants reduce the update for large samples regardless of their cause. | These variants limit the update for large samples regardless of their cause. | 前段と同じ動詞を使い、既存手法の作用を述べる。 |
| A sudden change in the primary path, such as a switch to another noise source, also increases the error. | However, a sudden primary path change also increases the error. A switch to another noise source can cause such a change. | 既存手法の制約とのつながりを示し、経路変化とその例を分ける。 |
| The robust variants therefore reduce the update when a large update is needed, and the reconvergence is slow. | By limiting the update while the error remains large, these variants slow reconvergence. | 更新の抑制が再収束を遅らせる関係を一文で示す。 |
| In our evaluation with measured room impulse responses, the best of eight existing algorithms needed about 1 s in median to recover after a switch of the noise source (Section 6). | In simulations with measured room impulse responses, we observed a median recovery time of about 1 s after a noise-source switch for the fastest of eight existing algorithms (Section 6). | 実測応答を使ったシミュレーション結果と明示し、中央値の表現を整える。 |
| A larger step size would speed up the reconvergence. | A larger step size can speed up reconvergence. | 速度向上の可能性を簡潔に表す。 |
| A change in the primary path can be detected from an increase in the error power. | An increase in error power can indicate a primary path change. | 誤差電力の上昇を変化の兆候として述べる。 |
| However, an impulse also increases the error power, and a larger step size during an impulse can degrade the control filter or make it diverge. | However, an impulse also raises error power. A larger step size during an impulse can degrade the control filter or cause it to diverge. | 検出器が反応する理由と、大きいステップサイズを直接使うリスクを分ける。 |
| An increase in the error power therefore shows only that the primary path may have changed. | The proposed algorithm therefore uses an increase in error power as a trigger to test a larger step size. | 兆候の用途を肯定形で示し、試行の説明につなぐ。 |
| The proposed algorithm first applies the larger step size to a separate filter and adopts the result only when it reduces the error. | It tests that step size in a separate filter and adopts the resulting coefficients only if they reduce the error sufficiently. | 何を移行するかを具体化し、十分な誤差低下が必要と明示。 |
| We refer to this adaptation as a trial. | We call this test a trial. | 直前に説明した適応の名称を短く定義。 |
| The foreground filter generates the control signal and is updated with the normal step size at all times. | Only one filter generates the control signal. We call it the foreground filter. It uses the normal step size during a trial. | 名称の前に役割を示し、通常ステップの適用範囲を凍結仕様に合わせる。 |
| The background filter is used for the trial. | The second filter adapts during the trial and is called the background filter. | 役割を示してから背景フィルタと定義。 |
| When the ratio of the error power to the filtered-reference power exceeds its running average, the background filter is initialized with the foreground coefficients and is updated with the larger step size. | The trial starts when the ratio of error power to filtered-reference power exceeds a threshold proportional to its running average on consecutive blocks. The algorithm copies the foreground coefficients into the background filter. It then adapts those coefficients with the larger step size. | 凍結仕様に合わせて連続したしきい値判定を明示し、検出と適応を分ける。 |
| Its coefficients are transferred to the foreground filter only after the error computed with the background coefficients becomes sufficiently smaller than the error computed with the foreground coefficients. | The algorithm transfers the background coefficients to the foreground filter only if the background filter yields sufficiently lower error power on the same samples. | 係数の移行条件を、同じ標本上の誤差電力の比較として示す。 |
| Both errors are computed from the measured reference and error signals by using the model of the secondary path. | For this comparison, the algorithm reconstructs both error signals from the measured reference and error signals using the secondary-path model. | 再構成の目的と、計算に使う信号とモデルを一文でつなぐ。 |
| The trial is discarded if this condition is not met within a fixed number of blocks. | The algorithm discards the trial if the error comparison fails to justify a transfer within a fixed number of blocks. | 時間制限と終了条件を能動態で示す。 |
| Because the background filter does not contribute to the control signal, a discarded trial leaves the control signal identical to that obtained without the trial, given the same initial state and input. | We show that a discarded trial yields the same control signal as a run with the trial disabled, given the same initial state and input. | 前景だけが制御信号を生成する構成を受けて、試行単位で出力の同一性が成り立つ条件を明示。 |
| In contrast, a transfer can also be confirmed when the primary path stays the same. | However, the transfer criterion can also be met while the primary path remains fixed. | 一次経路が固定でも移行条件を満たしうることを、廃棄時の結果に続けて述べる。 |
| Such a transfer replaces the foreground coefficients and thus affects the control signal. | A transfer in this case replaces the foreground coefficients and can affect the control signal. | 移行は係数を置き換え、出力には影響しうるという区別を示す。 |
| We report how often such transfers occurred. | We report the frequency of these transfers. | 集計対象を直前の文に結び付け、論文で報告する内容として表現。 |
| The trial is useful only if the background filter converges quickly. | To meet the transfer criterion within the trial window, the background filter must reduce the error quickly. | 背景フィルタに速い適応が必要な理由を移行条件と試行期間で示す。 |
| For this purpose, the proposed algorithm combines two elements, and we analyze them for finite block lengths. | We analyze two elements of the update for finite block lengths. | 解析の対象と条件を先に示す。 |
| First, the update direction is based on the spectral factor of the reference power spectrum [Elliott and Rafaely 2000], and the step size is divided by the input power of each block. | First, the update direction uses a spectral factor of the filtered-reference power spectrum [Elliott and Rafaely 2000]. The algorithm measures the power of each reference frame relative to its running average. It divides the step size by this ratio, with a lower bound on the denominator. | 更新方向と尺度補正を分け、分母の定義、下限、スペクトル因子の対象を凍結仕様に合わせる。 |
| We show that, after this division, the update caused by an isolated impulse does not depend on the amplitude of the impulse. | For the isolated-impulse analysis, we hold the coefficients, spectral factor, and running power estimate fixed. We assume that the error signal contains no independent disturbance. Under these assumptions, we show that the normalized update is independent of the reference impulse amplitude when the relative frame power exceeds its lower bound. | 定理 C の仮定と下限制限が作動しない場合を明記し、振幅不変性を解析の範囲に合わせる。 |
| The algorithm therefore uses impulsive samples without clipping them. | The algorithm therefore retains impulsive samples in the update without clipping them. | 振幅を切り詰めない対象と操作を明確にする。 |
| Second, after start-up and after each transfer, the error frame is recomputed with the current coefficients, as in the modified FxLMS algorithm [ref]. | Second, the algorithm recomputes error frames using the current coefficients during start-up and immediately after a transfer, as in the modified FxLMS algorithm [ref]. | 再計算する期間と使用する係数を能動態で述べ、仮引用を保持。 |
| We derive the form of this update and determine the range of stable step sizes numerically. | We derive the resulting update. We determine the range of stable step sizes numerically. | 解析による導出と数値評価を別の文にする。安定範囲の主張は著者確認事項に記載。 |
| The structure with foreground and background filters was proposed for echo cancellation [Ochiai et al. 1977]. | Ochiai and colleagues [Ochiai et al. 1977] proposed the foreground and background filter structure for echo cancellation. | 先行研究の著者を主語にし、引用キーを保持。 |
| The contribution of this paper lies in its use for the reconvergence of ANC under impulsive noise and in the analysis and evaluation of the combined algorithm. | We apply this structure to reconvergence in ANC under impulsive noise. We analyze and evaluate the resulting algorithm. | 用途と組み合わせた手法の解析・評価を分け、貢献を直接述べる。 |
| We evaluated the proposed algorithm in closed-loop simulations with room impulse responses measured in one room [MeshRIR] and with ten noise types (four synthetic signals and six recordings). | We evaluated the algorithm in closed-loop simulations using impulse responses measured in one room [MeshRIR]. The inputs comprised ten noise types, including four synthetic signals and six recordings. | シミュレーションの経路と入力の内訳を別の文で示す。 |
| Compared with the best of eight existing algorithms, the proposed algorithm reduced the median recovery time after a switch of the noise source from 0.98 s to 0.36 s for a nearby new source and from 1.26 s to 0.40 s for a distant new source. | After a switch to a source near the original one, we observed a median recovery time of 0.36 s, compared with 0.98 s for the existing algorithm with the shortest median among the eight tested. For a distant new source, the respective medians were 0.40 s and 1.26 s. | 実験で観察した回復時間と近・遠の基準を示し、比較対象の選定基準を明確にする。 |
| For machinegun noise in both conditions and for car interior noise with a nearby new source, the existing algorithm recovered faster. | However, the fastest existing algorithm had a shorter median recovery time for machinegun noise in both conditions and for car interior noise in the near condition. | 全体の集計に続けて例外条件と比較指標を示す。 |
| Without a path change, the noise reduction of the two algorithms differed by less than 0.1 dB. | With a fixed primary path and no added disturbance, the two algorithms differed by less than 0.1 dB in mean noise reduction over the full 30 s and over the final 10 s. | 0.1 dB 未満の対象を変化なしの条件に限定し、二つの集計窓を明示。 |
| The remainder of this paper is organized as follows. | —（削除） | 節の案内に直接入り、内容を重複させない。 |
| Section 2 reviews related work, and Section 3 formulates the problem. | Section 2 reviews related work. Section 3 formulates the problem. | 各節の内容を一文ずつ示す。 |
| Section 6 reports the reconvergence after a primary path change, and Section 7 reports the cost of the trial mechanism when the path does not change. | Section 6 reports recovery after a primary path change. Section 7 evaluates the cost of the trial mechanism with the primary path held fixed. | 再収束の評価と経路固定時の試行の評価を別の文で示す。 |

## 著者への確認事項

1. **インパルス性騒音の振幅分布（序論）**

   「Its amplitude distribution has heavier tails than a Gaussian distribution」は、インパルス性騒音一般の分布に対する主張にも読めます。対象を測定信号一般まで広げるか、SαS モデルを用いる場合に限るかを確認してください。

   限定案　“Symmetric α-stable distributions can model impulsive noise with heavy tails [Shao and Nikias 1993].”

2. **既存手法の安定性（要旨・序論）**

   「keeps the control filter stable」と「keep the control filter stable」は、各手法の安定性が入力やステップサイズによらず保証されるようにも読めます。安定性の保証を述べる根拠と条件を確認してください。

   限定案　“Limiting the influence of impulsive samples can reduce the risk of divergence.”

3. **再収束中の誤差と最適係数（要旨・序論）**

   「The error stays large until the control filter reconverges」と「The error then stays large until the control filter converges to the new optimal coefficients」は、高い誤差が必ず持続し、制御フィルタが最適係数に到達するという読み方ができます。評価の回復時間は誤差電力を基準にしており、係数の一致を測っていません。

   限定案　“The error can remain elevated while the control filter adapts to the new primary path.”

4. **安定なステップサイズの範囲（序論）**

   「determine the range of stable step sizes numerically」は、有限時間の数値観察から一般的な安定範囲を決めたという主張にも読めます。補題 F に合わせて、数値評価の範囲を確認してください。

   限定案　“We derive the resulting update. In numerical experiments, we identify the step sizes that yield decay over the observation window.”
