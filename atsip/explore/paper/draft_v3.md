# Draft v3 (abstract and introduction)

v1.1 の流れを基本に、v2（Codex）の指摘のうち正しいものを取り込んだ。[..] は著者用のメモ。

## v2 から取り込んだ点
- スペクトル因子は濾波参照のパワースペクトルのもの（v1 の "reference" は誤り。THEORY_frozen.md §1 の Λ = Ŝ_rr + εP̄）。
- 尺度の補正は「参照ブロックのパワー ÷ その指数平均」。
- Akhtar and Mitsuhashi は「しきい値でクリップ」。M-estimator の 2 研究を著者主語で書く。
- 確認事項 2（安定を保証するように読める）→ "reduce the risk of divergence"。
- 確認事項 3（誤差が収束まで必ず大きいと読める）→ "remains elevated while the control filter adapts to the new path"。
- 確認事項 4（安定範囲を数値で決めたと読める）→ 数値実験で見た減衰と増大の境目より下にステップを置いた、と書く（補題 F）。
- 定常の比較に、外乱なしと集計窓（30 秒全体、末尾 10 秒）を明記。
- 検出が連続ブロックでの判定であること。
- 「最良」→「最速」：回復時間の中央値は APSIPA 2026 版が 8 手法中で最短（near 122、far 157 ブロック。次点は 288、185）。

## v2 から取り込まなかった点
- 短い文への分割の多く（"Sun et al. also used an M-estimator." など）。文と文の関係が見えなくなるため。
- 要旨の細部（しきい値の連続ブロック、除数の下限）。要旨が 303 語になるため。
- 確認事項 1：重い裾の分布は文献つきの一般的な記述なので残す。

## Abstract

Adaptive algorithms for active control of impulsive noise limit the influence of large samples in the reference and error signals to reduce the risk of divergence. However, a sudden change in the primary path also increases the error, and the same limitation then reduces the coefficient update while the control filter adapts to the new path. As a result, the reconvergence is slow. This paper proposes a frequency-domain filtered-x algorithm that tests a larger step size on a background filter before the result is used for control. The foreground filter generates the control signal and keeps its normal update. When the ratio of the error power to the filtered-reference power rises above a multiple of its running average, the background filter is initialized with the foreground coefficients and adapted with the larger step size. Its coefficients are transferred to the foreground filter only when they give a sufficiently smaller error on the same data. Otherwise, they are discarded, and the control signal is identical to that obtained without the background filter under the same initial state and input. The update uses a spectral-factor direction and a step size normalized by the relative power of each reference block. The error frames are recomputed with the current coefficients after start-up and after each transfer. We evaluated the algorithm in simulations with measured room impulse responses and ten noise types. Compared with the fastest of eight existing algorithms, the proposed algorithm reduced the median recovery time after a switch of the noise source by 63% for a nearby new source and by 68% for a distant new source. For machinegun noise in both conditions and for car interior noise with a nearby new source, the existing algorithm recovered faster. With a fixed primary path, the noise reduction of the two algorithms differed by less than 0.1 dB.

[304 語。投稿先の語数制限を確認して調整する。63% = 1 − 0.36/0.98、68% = 1 − 0.40/1.26。]

## 1. Introduction

Impulsive noise contains short bursts whose amplitudes are much larger than the background level. Its amplitude distribution has heavier tails than a Gaussian distribution and is often modeled by a symmetric α-stable distribution [Shao and Nikias 1993]. In feedforward active noise control (ANC), the filtered-x least-mean-square (FxLMS) algorithm updates the control filter in proportion to the product of the filtered reference signal and the error signal [Kuo and Morgan 1996]. An impulsive sample in either signal therefore produces a large update, and such an update can make the control filter diverge.

To prevent this divergence, robust variants of the FxLMS algorithm limit the influence of large samples in the reference and error signals. Sun et al. [Sun 2006] ignored the samples that exceeded a threshold. Akhtar and Mitsuhashi [Akhtar 2009] instead clipped the reference and error samples at their thresholds. Wu et al. [Wu 2011] minimized the squared logarithm of the error, so that the cost grows slowly for large errors. Wu and Qiu [Wu and Qiu 2013] and Sun et al. [Sun et al. 2015] replaced the squared error with M-estimators. Feng et al. [Feng et al. 2024] used a fractional lower-order statistic of the error. These variants reduce the risk of divergence under impulsive noise. Section 2 reviews them in more detail.

These variants reduce the update for large samples regardless of their cause. A sudden change in the primary path, such as a switch to another noise source, also increases the error. The error remains elevated while the control filter adapts to the new path. The robust variants therefore reduce the update when a large update is needed, and the reconvergence is slow. In our simulations with measured room impulse responses, the fastest of eight existing algorithms needed about 1 s in median to recover after a switch of the noise source (Section 6).

A larger step size would speed up the reconvergence. A change in the primary path can be detected from an increase in the error power. However, an impulse also increases the error power, and a larger step size during an impulse can degrade the control filter or make it diverge. An increase in the error power therefore shows only that the primary path may have changed. The proposed algorithm first applies the larger step size to a separate filter and adopts the result only when it reduces the error. We refer to this adaptation as a trial.

The proposed algorithm uses two filters with the same structure. The foreground filter generates the control signal and is updated with the normal step size at all times. The background filter is used for the trial. When the ratio of the error power to the filtered-reference power stays above a multiple of its running average for consecutive blocks, the background filter is initialized with the foreground coefficients and is updated with the larger step size. Its coefficients are transferred to the foreground filter only after the error computed with the background coefficients becomes sufficiently smaller than the error computed with the foreground coefficients. Both errors are computed from the measured reference and error signals by using the model of the secondary path. The trial is discarded if this condition is not met within a fixed number of blocks. Because the background filter does not contribute to the control signal, a discarded trial leaves the control signal identical to that obtained without the trial, given the same initial state and input. In contrast, a transfer can also be confirmed when the primary path stays the same. Such a transfer replaces the foreground coefficients and thus affects the control signal. We report how often such transfers occurred.

The trial is useful only if the background filter reduces the error within the trial. For this purpose, the proposed algorithm combines two elements, and we analyze them for finite block lengths. First, the update direction is based on the spectral factor of the filtered-reference power spectrum [Elliott and Rafaely 2000], and the step size is divided by the power of each reference block relative to its running average. We show that, with the coefficients held fixed, the update caused by an isolated impulse in the reference signal does not depend on the amplitude of the impulse. The algorithm therefore uses impulsive samples without clipping them. Second, after start-up and after each transfer, the error frame is recomputed with the current coefficients, as in the modified FxLMS algorithm [ref]. We derive the form of this update and select its step size below the boundary between decay and growth that we observed in numerical experiments. Ochiai et al. [Ochiai et al. 1977] proposed the structure with foreground and background filters for echo cancellation. We apply this structure to the reconvergence of ANC under impulsive noise, and we analyze and evaluate the combined algorithm.

We evaluated the proposed algorithm in closed-loop simulations with room impulse responses measured in one room [MeshRIR] and with ten noise types (four synthetic signals and six recordings). Compared with the fastest of eight existing algorithms, the proposed algorithm reduced the median recovery time after a switch of the noise source from 0.98 s to 0.36 s for a nearby new source and from 1.26 s to 0.40 s for a distant new source. For machinegun noise in both conditions and for car interior noise with a nearby new source, the existing algorithm recovered faster. With a fixed primary path and no disturbance at the error microphone, the mean noise reduction of the two algorithms differed by less than 0.1 dB over the full 30 s and over the final 10 s.

The remainder of this paper is organized as follows. Section 2 reviews related work, and Section 3 formulates the problem. Section 4 describes the proposed algorithm and its analysis. Section 5 describes the evaluation framework. Section 6 reports the reconvergence after a primary path change, and Section 7 reports the cost of the trial mechanism when the path does not change. Section 8 concludes the paper.

## 2. Related work

### 2.1 Adaptive algorithms for impulsive noise

The FxLMS algorithm minimizes the mean squared error. For impulsive noise modeled by an α-stable distribution with α < 2, the second-order moment does not exist [Shao and Nikias 1993]. Previous studies therefore modified the update so that large samples have a limited effect on it.

One group of algorithms applies thresholds to the signals. Sun et al. [Sun 2006] ignored the samples that exceeded a threshold during adaptation. Akhtar and Mitsuhashi [Akhtar 2008, Akhtar 2009] instead replaced such samples in the reference and error signals with the threshold values to obtain faster convergence.

A second group replaces the squared error with a cost function that grows slowly for large errors. Wu et al. [Wu 2011] minimized the squared logarithmic transformation of the error. This cost requires no thresholds that depend on the noise statistics. Thanigai et al. [Thanigai 2007] used the three-part M-estimator of Hampel to detect and reject impulses in the error signal. Sun et al. [Sun et al. 2015] extended this algorithm by passing the reference signal through another Hampel M-estimator with thresholds estimated online. Wu and Qiu [Wu and Qiu 2013] reformulated several existing algorithms as M-estimators and proposed an algorithm based on the Fair function. Feng et al. [Feng et al. 2024] derived a normalized algorithm from the fractional lower-order covariance and gave its convergence conditions. Cheng et al. [Cheng 2023] applied the Gaussian error function to the error and combined it with a normalized variable step size based on a sine function.

In our previous work [APSIPA 2026], a frequency-domain FxNLMS algorithm scaled the step size in each frequency bin according to the coherence between the filtered reference and error signals. The scaling factor was limited to a fixed range around the nominal step size.

In all of these algorithms, the update grows less than in proportion to the error, or stays bounded, when the error becomes large. This property gives the stability under impulsive noise. It also applies when the error increases after a change in the primary path, and it then limits the speed of reconvergence (Section 1).

### 2.2 Algorithms with two adaptive filters

Two adaptive filters have been combined to obtain both fast convergence and a small steady-state error. Ferrer et al. [Ferrer 2013] applied the convex combination of two filtered-x filters with different step sizes to ANC. Cheng et al. [Cheng 2022] combined an FxLMS filter with a filter based on the maximum Versoria criterion for impulsive noise. Yu et al. [Yu 2020] combined an FxLMS filter with a filter based on the minimum kernel risk-sensitive loss. In these schemes, a mixing parameter weights the two filters, and both filters contribute to the control signal whenever their weights are nonzero.

Ochiai et al. [Ochiai et al. 1977] used two filters in a different way. Their echo canceler had a background model, which was adapted, and a foreground model, which synthesized the echo replica. The model with the smaller residual became the new foreground model. This structure prevented false adaptation during double talk from affecting the echo replica. In echo cancellation, the replica is subtracted electrically, and the residual of each model can be computed directly. In ANC, the control signal reaches the error microphone through the secondary path, and only the error of the foreground filter is measured. The proposed algorithm therefore reconstructs the error of the background filter with the model of the secondary path (Section 2.3).

### 2.3 Frequency-domain update and error reconstruction

Kuo et al. [Kuo 2005] implemented the FxLMS algorithm in blocks to reduce its computational load. Elliott and Rafaely [Elliott and Rafaely 1997, Elliott and Rafaely 2000] showed that the bin-normalized frequency-domain LMS algorithm can converge to a biased solution when the filter is constrained to be causal. They applied the spectral factors of the reference power spectrum before and after the causality constraint, and the resulting algorithm converged to the optimal causal filter. For feedforward control, the factorization also includes the minimum-phase part of the plant response. The proposed algorithm uses this spectral-factor direction without modification.

Bjarnason [Bjarnason 1992] proposed the modified FxLMS algorithm. This algorithm estimates the primary noise at the error microphone by subtracting the filtered control signal from the measured error. It then computes the error that the current coefficients would produce. This modification removes the delay of the secondary path from the adaptation loop [Yang 2018 review]. The proposed algorithm uses this reconstruction for two purposes. The first is the update with a large step size after start-up and after a transfer. The second is the comparison between the foreground and background filters.

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
