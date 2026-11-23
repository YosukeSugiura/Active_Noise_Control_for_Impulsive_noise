# Draft v4 (abstract, Sections 1–2)

v3 を査読者の目で見直して直した版。[..] は著者用のメモ。

## v3 から直した点

| # | v3 の問題 | v4 での扱い |
|---|---|---|
| 1 | 「大きな標本の抑制が再収束を遅くする」という因果は、自分のデータと合わない。抑制のない FxNLMS は回復 415/537 ブロックで、抑制のある手法より遅い（TUNE.json：発散を避けるためステップが公開値の 1/16、Th_FxNLMS はしきい値 200 で実質クリップなし） | 「発散を防ぐ設定（しきい値・コスト・小さいステップ）が、原因を区別せずに同じ更新則として働き、再収束の速さも決める」に改めた（要旨、1 章、2.1 節） |
| 2 | 2.1 節末「すべての手法で更新は誤差に比例せず有界」は、APSIPA 2026 版（正規化更新）には当てはまらない | 2 群と前報を分けて書いた |
| 3 | 最速の比較手法が著者の前報であることを書いていない | 1 章で明記。前報との違い（単一フィルタ、2 要素なし）も 1 文で書いた |
| 4 | 「前景は常に通常のステップ」は誤り（始動時と移行後は大きいステップ） | 「試行の間は通常のステップ」に限定 |
| 5 | 誤差フレームの作り直しの適用範囲に、背景の試行が抜けていた | 「大きいステップを使うとき：始動、試行、移行後」 |
| 6 | 「誤差パワーの増加で検出」と書きながら、実際の検出量は濾波参照との比 | 比で書き、比が上がる他の原因として誤差マイクの外乱を挙げた（7 章の条件と対応） |
| 7 | 回復時間の基準が手法ごとである点を書いていない | 定義を 1 文で書き、係数固定の消音量（0.51 秒後、7.9 対 5.0 dB、10.9 対 6.8 dB）を併記 |
| 8 | 1 章の先行研究の列挙が 2.1 節と重複 | 1 章は代表 3 件に絞った |
| 9 | 2.3 節「スペクトル因子の方向を変更なしで使う」は不正確（正則化あり、濾波参照を因子分解） | 「この構成に従い、正則化した濾波参照のパワースペクトルを因子分解する」 |
| 10 | 要旨 304 語 | 278 語 |

## 文章では直せない点（査読で問われうる）

1. 凸結合の手法が比較 8 手法に入っていない。2.2 節で構成の違いは書いたが、「速い収束と頑健さの両立」という同じ目的の手法なので、比較を求められる可能性がある。
2. 検出の見逃し（near 60/200、far 20/200）を 1 章では書いていない。6 章で件数と、machinegun・car interior noise の例外との関係を書く必要がある。
3. 前報との違いの 1 文（単一フィルタ、スペクトル因子も誤差の作り直しも使わない）は、コード（Freq_FxNLMS_CohVSS）から判断した。前報の論文の記述と照合が要る。
4. 影のフィルタでステップを制御するエコーキャンセラの研究など、二重フィルタの関連文献は Ochiai 1977 しか挙げていない。追加の調査が要る。
5. 評価は 1 室、シミュレーションのみ。1 章には書いてあるが、結論でも限界として書く。

## Abstract

Adaptive algorithms for active control of impulsive noise prevent divergence by limiting the influence of large samples in the reference and error signals or by using a small step size. These algorithms apply the same update rule to a large error whether it is caused by an impulse or by a sudden change in the primary path. The settings that prevent divergence therefore also slow the reconvergence after the change. This paper proposes a frequency-domain filtered-x algorithm that tests a larger step size on a background filter before the result is used for control. During the test, the foreground filter generates the control signal with its normal step size. When the ratio of the error power to the filtered-reference power rises above a multiple of its running average, the background filter is initialized with the foreground coefficients and adapted with the larger step size. Its coefficients are transferred to the foreground filter only when they give a sufficiently smaller error on the same data. Otherwise, they are discarded, and the control signal is identical to that obtained without the test under the same initial state and input. In simulations with measured room impulse responses and ten noise types, the proposed algorithm reduced the median recovery time after a switch of the noise source by 63% and 68% compared with the fastest of eight comparison algorithms. The two values are for new sources close to and far from the original source. For machinegun noise in both conditions and for car interior noise with the close source, the comparison algorithm recovered faster. With a fixed primary path, the noise reduction of the two algorithms differed by less than 0.1 dB.

## 1. Introduction

Impulsive noise contains short bursts whose amplitudes are much larger than the background level. Its amplitude distribution has heavier tails than a Gaussian distribution and is often modeled by a symmetric α-stable distribution [Shao and Nikias 1993]. In feedforward active noise control (ANC), the filtered-x least-mean-square (FxLMS) algorithm updates the control filter in proportion to the product of the filtered reference signal and the error signal [Kuo and Morgan 1996]. An impulsive sample in either signal therefore produces a large update, and such an update can make the control filter diverge.

To prevent this divergence, robust variants of the FxLMS algorithm limit the influence of large samples in the reference and error signals. Sun et al. [Sun 2006] ignored the samples that exceeded a threshold, and Akhtar and Mitsuhashi [Akhtar 2009] replaced them with the threshold values. Wu et al. [Wu 2011] minimized the squared logarithm of the error, so that the cost grows slowly for large errors. Other variants are based on M-estimators [Wu and Qiu 2013, Sun et al. 2015] and on fractional lower-order statistics [Feng et al. 2024], and Section 2 reviews them. A small step size also reduces the risk of divergence, but it slows the convergence.

These algorithms apply the same update rule to a large error regardless of its cause. A sudden change in the primary path, such as a switch to another noise source, also increases the error, and the error remains elevated while the control filter adapts to the new path. The thresholds, cost functions, and step sizes that prevent divergence under impulses therefore also determine the speed of this reconvergence. In our simulations with measured room impulse responses, we selected the step sizes of eight comparison algorithms to avoid divergence. The fastest of them then needed about 1 s in median to recover after a switch of the noise source (Section 6). This algorithm was our previous frequency-domain algorithm with a coherence-based step size [APSIPA 2026].

A larger step size would speed up the reconvergence if it were applied only after a change in the primary path. The ratio of the error power to the filtered-reference power rises after such a change. However, this ratio also rises for other reasons, such as a disturbance at the error microphone, and a larger step size in these cases can degrade the control filter or make it diverge. A rise in the ratio therefore shows only that the primary path may have changed. The proposed algorithm first applies the larger step size to a separate filter and adopts the result only when it reduces the error. We refer to this adaptation as a trial.

The proposed algorithm uses two filters with the same structure. The foreground filter generates the control signal, and the background filter is used only for the trial. When the ratio stays above a multiple of its running average for consecutive blocks, the background filter is initialized with the foreground coefficients and is updated with the larger step size. The foreground filter keeps its normal step size during the trial. The background coefficients are transferred to the foreground filter only after the error computed with them becomes sufficiently smaller than the error computed with the foreground coefficients. Both errors are computed from the measured reference and error signals by using the model of the secondary path. The trial is discarded if this condition is not met within a fixed number of blocks. Because the background filter does not contribute to the control signal, a discarded trial leaves the control signal identical to that obtained without the trial, given the same initial state and input. In contrast, a transfer can also be confirmed when the primary path stays the same. Such a transfer replaces the foreground coefficients and thus affects the control signal. We report how often such transfers occurred.

The trial is useful only if the background filter reduces the error within the trial. For this purpose, the update combines two elements, and we analyze them for finite block lengths. First, the update direction is based on the spectral factor of the filtered-reference power spectrum [Elliott and Rafaely 2000], and the step size is divided by the power of each reference block relative to its running average. We show that, with the coefficients held fixed, the update caused by an isolated impulse in the reference signal does not depend on the amplitude of the impulse. The algorithm therefore uses impulsive samples without clipping them. Second, whenever the larger step size is used, the error frame is recomputed with the current coefficients, as in the modified FxLMS algorithm [Bjarnason 1992]. This recomputation applies to start-up, to the trial, and to a fixed number of blocks after a transfer. We derive the form of this update and select its step size below the boundary between decay and growth that we observed in numerical experiments. Our previous algorithm [APSIPA 2026] had a single filter and used neither element. Ochiai et al. [Ochiai et al. 1977] proposed the structure with foreground and background filters for echo cancellation. We apply this structure to the reconvergence of ANC under impulsive noise, and we analyze and evaluate the combined algorithm.

We evaluated the proposed algorithm in closed-loop simulations with room impulse responses measured in one room [MeshRIR] and with ten noise types (four synthetic signals and six recordings). The recovery time is the time until the short-time error power stays within 3 dB of the level that the same algorithm reaches after the change. Compared with the fastest of the eight comparison algorithms, the proposed algorithm reduced the median recovery time after a switch of the noise source from 0.98 s to 0.36 s for a new source close to the original source and from 1.26 s to 0.40 s for a new source far from it. For machinegun noise in both conditions and for car interior noise with the close source, the comparison algorithm recovered faster. Because the reference level of the recovery time depends on the algorithm, we also measured the noise reduction with the coefficients frozen 0.51 s after the switch. Its median was 7.9 dB for the proposed algorithm and 5.0 dB for the comparison algorithm with the close source, and 10.9 dB and 6.8 dB with the far source. With a fixed primary path and no disturbance at the error microphone, the mean noise reduction of the two algorithms differed by less than 0.1 dB over the full 30 s and over the final 10 s.

The remainder of this paper is organized as follows. Section 2 reviews related work, and Section 3 formulates the problem. Section 4 describes the proposed algorithm and its analysis. Section 5 describes the evaluation framework. Section 6 reports the reconvergence after a primary path change, and Section 7 reports the cost of the trial mechanism when the path does not change. Section 8 concludes the paper.

## 2. Related work

### 2.1 Adaptive algorithms for impulsive noise

The FxLMS algorithm minimizes the mean squared error. For impulsive noise modeled by an α-stable distribution with α < 2, the second-order moment does not exist [Shao and Nikias 1993]. Previous studies therefore modified the update so that large samples have a limited effect on it.

One group of algorithms applies thresholds to the signals. Sun et al. [Sun 2006] ignored the samples that exceeded a threshold during adaptation. Akhtar and Mitsuhashi [Akhtar 2008, Akhtar 2009] instead replaced such samples in the reference and error signals with the threshold values to obtain faster convergence.

A second group replaces the squared error with a cost function that grows slowly for large errors. Wu et al. [Wu 2011] minimized the squared logarithmic transformation of the error. This cost requires no thresholds that depend on the noise statistics. Thanigai et al. [Thanigai 2007] used the three-part M-estimator of Hampel to detect and reject impulses in the error signal. Sun et al. [Sun et al. 2015] extended this algorithm by passing the reference signal through another Hampel M-estimator with thresholds estimated online. Wu and Qiu [Wu and Qiu 2013] reformulated several existing algorithms as M-estimators and proposed an algorithm based on the Fair function. Feng et al. [Feng et al. 2024] derived a normalized algorithm from the fractional lower-order covariance and gave its convergence conditions. Cheng et al. [Cheng 2023] applied the Gaussian error function to the error and combined it with a normalized variable step size based on a sine function.

In our previous work [APSIPA 2026], a frequency-domain FxNLMS algorithm scaled the step size in each frequency bin according to the coherence between the filtered reference and error signals. The scaling factor was limited to a fixed range around the nominal step size.

The algorithms in the two groups bound or compress the contribution of a large error sample to the update. Our previous algorithm keeps a normalized update and changes the step size only within the fixed range. In both cases, one update rule with one set of parameters handles every large error, and these parameters are selected for stability under impulsive noise. The same rule is then applied after a change in the primary path, and it determines the speed of reconvergence (Section 1).

### 2.2 Algorithms with two adaptive filters

Two adaptive filters have been combined to obtain both fast convergence and a small steady-state error. Ferrer et al. [Ferrer 2013] applied the convex combination of two filtered-x filters with different step sizes to ANC. Cheng et al. [Cheng 2022] combined an FxLMS filter with a filter based on the maximum Versoria criterion for impulsive noise. Yu et al. [Yu 2020] combined an FxLMS filter with a filter based on the minimum kernel risk-sensitive loss. In these schemes, a mixing parameter weights the two filters, and both filters contribute to the control signal whenever their weights are nonzero.

Ochiai et al. [Ochiai et al. 1977] used two filters in a different way. Their echo canceler adapted a background model and used a foreground model to synthesize the echo replica. The model with the smaller residual became the new foreground model. This structure prevented false adaptation during double talk from affecting the echo replica. In echo cancellation, the replica is subtracted electrically, and the residual of each model can be computed directly. In ANC, the control signal reaches the error microphone through the secondary path, and only the error of the foreground filter is measured. The proposed algorithm therefore reconstructs the error of the background filter with the model of the secondary path (Section 2.3).

### 2.3 Frequency-domain update and error reconstruction

Kuo et al. [Kuo 2005] implemented the FxLMS algorithm in blocks to reduce its computational load. Elliott and Rafaely [Elliott and Rafaely 1997, Elliott and Rafaely 2000] showed that the bin-normalized frequency-domain LMS algorithm can converge to a biased solution when the filter is constrained to be causal. They applied the spectral factors of the reference power spectrum before and after the causality constraint, and the resulting algorithm converged to the optimal causal filter. For feedforward control, the factorization also includes the minimum-phase part of the plant response. The proposed algorithm follows this construction and factorizes the regularized power spectrum of the filtered reference signal.

Bjarnason [Bjarnason 1992] proposed the modified FxLMS algorithm. This algorithm estimates the primary noise at the error microphone by subtracting the filtered control signal from the measured error. It then computes the error that the current coefficients would produce. This modification removes the delay of the secondary path from the adaptation loop [Yang 2018 review]. The proposed algorithm uses this reconstruction for two purposes. The first is the update with the larger step size during start-up, during a trial, and after a transfer. The second is the comparison between the foreground and background filters.

[2 章のメモ]
- APSIPA 2026 版の説明（周波数ごとのステップをコヒーレンスで伸縮、倍率は 0.6–1.6 の範囲）は algorithms/algorithms.py の Freq_FxNLMS_CohVSS から読んだもの。論文の記述と合っているか確認してください。
- Th_FxNLMS の出典：README は MWSCAS 2008 pp. 330–333。書誌データベース（ISC）では同名論文が pp. 2669–2672。要照合。
- EVSS_FxeLMS の出典：README の題名は誤り。正しくは Y. Cheng, C. Li, S. Chen, Z. Zhou, "An enhanced impulse noise control algorithm using a novel nonlinear function," Circuits Syst. Signal Process., 2023, doi:10.1007/s00034-023-02421-3。
- Thanigai 2007 が FxLMM（Hampel 型）の出典であることは、Sun et al. 2015 の要旨と中国特許 CN104035332A の記述で確認。本文は未読。
- Bjarnason 1992：EUSIPCO 1992, pp. 1053–1056（複数の引用元で一致）。「遅れを適応の経路から除く」は Yang ら（Appl. Sci. 2018, 8(11), 2313 の総説）の記述。著者名は要照合。
- Ferrer 2013：IEEE TASLP 21(1)。頁は 154–165 と 156–167 の 2 説あり。要照合。
- Cheng 2022：Appl. Acoust. 186, 108438, doi:10.1016/j.apacoust.2021.108438。Yu 2020：J. Low Freq. Noise Vib. Active Control 39, doi:10.1177/1461348419838394。
- Elliott and Rafaely 1997：IEEE Signal Process. Lett. 4(12), 337–339, doi:10.1109/97.650039。
- Wu and Qiu 2013：Appl. Acoust. 74(3), 407–412, doi:10.1016/j.apacoust.2012.06.019。Sun et al. 2015：Appl. Acoust. 90, 31–41, doi:10.1016/j.apacoust.2014.10.012。
