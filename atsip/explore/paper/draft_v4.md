# Draft v4 (abstract, Sections 1–4)

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

1. 凸結合（C-FXLMS、Ferrer 2013）を比較に追加済み（RESULTS_cc.md、ffa7fb0）。比較は 9 手法になった。最速は APSIPA 2026 版のまま（C-FXLMS の回復中央値は near 2.54 秒、far 2.78 秒）。
2. 検出の見逃し（near 60/200、far 20/200）を 1 章では書いていない。6 章で件数と、machinegun・car interior noise の例外との関係を書く必要がある。
3. 前報との違いの 1 文（単一フィルタ、スペクトル因子も誤差の作り直しも使わない）は、コード（Freq_FxNLMS_CohVSS）から判断した。前報の論文の記述と照合が要る。
4. 影のフィルタでステップを制御するエコーキャンセラの研究など、二重フィルタの関連文献は Ochiai 1977 しか挙げていない。追加の調査が要る。
5. 評価は 1 室、シミュレーションのみ。1 章には書いてあるが、結論でも限界として書く。

## Abstract

Adaptive algorithms for active control of impulsive noise prevent divergence by limiting the influence of large samples in the reference and error signals or by using a small step size. These algorithms apply the same update rule to a large error whether it is caused by an impulse or by a sudden change in the primary path. The settings that prevent divergence therefore also slow the reconvergence after the change. This paper proposes a frequency-domain filtered-x algorithm that tests a larger step size on a background filter before the result is used for control. During the test, the foreground filter generates the control signal with its normal step size. When the ratio of the error power to the filtered-reference power rises above a multiple of its running average, the background filter is initialized with the foreground coefficients and adapted with the larger step size. Its coefficients are transferred to the foreground filter only when they give a sufficiently smaller error on the same data. Otherwise, they are discarded, and the control signal is identical to that obtained without the test under the same initial state and input. In simulations with measured room impulse responses and ten noise types, the proposed algorithm reduced the median recovery time after a switch of the noise source by 63% and 68% compared with the fastest of nine comparison algorithms. The two values are for new sources close to and far from the original source. For machinegun noise in both conditions and for car interior noise with the close source, the comparison algorithm recovered faster. With a fixed primary path, the noise reduction of the two algorithms differed by less than 0.1 dB.

## 1. Introduction

Impulsive noise contains short bursts whose amplitudes are much larger than the background level. Its amplitude distribution has heavier tails than a Gaussian distribution and is often modeled by a symmetric α-stable distribution [Shao and Nikias 1993]. In feedforward active noise control (ANC), the filtered-x least-mean-square (FxLMS) algorithm updates the control filter in proportion to the product of the filtered reference signal and the error signal [Kuo and Morgan 1996]. An impulsive sample in either signal therefore produces a large update, and such an update can make the control filter diverge.

To prevent this divergence, robust variants of the FxLMS algorithm limit the influence of large samples in the reference and error signals. Sun et al. [Sun 2006] ignored the samples that exceeded a threshold, and Akhtar and Mitsuhashi [Akhtar 2009] replaced them with the threshold values. Wu et al. [Wu 2011] minimized the squared logarithm of the error, so that the cost grows slowly for large errors. Other variants are based on M-estimators [Wu and Qiu 2013, Sun et al. 2015] and on fractional lower-order statistics [Feng et al. 2024], and Section 2 reviews them. A small step size also reduces the risk of divergence, but it slows the convergence.

These algorithms apply the same update rule to a large error regardless of its cause. A sudden change in the primary path, such as a switch to another noise source, also increases the error, and the error remains elevated while the control filter adapts to the new path. The thresholds, cost functions, and step sizes that prevent divergence under impulses therefore also determine the speed of this reconvergence. In our simulations with measured room impulse responses, we selected the step sizes of nine comparison algorithms to avoid divergence. The fastest of them then needed about 1 s in median to recover after a switch of the noise source (Section 6). This algorithm was our previous frequency-domain algorithm with a coherence-based step size [APSIPA 2026].

A larger step size would speed up the reconvergence if it were applied only after a change in the primary path. The ratio of the error power to the filtered-reference power rises after such a change. However, this ratio also rises for other reasons, such as a disturbance at the error microphone, and a larger step size in these cases can degrade the control filter or make it diverge. A rise in the ratio therefore shows only that the primary path may have changed. The proposed algorithm first applies the larger step size to a separate filter and adopts the result only when it reduces the error. We refer to this adaptation as a trial.

The proposed algorithm uses two filters with the same structure. The foreground filter generates the control signal, and the background filter is used only for the trial. When the ratio stays above a multiple of its running average for consecutive blocks, the background filter is initialized with the foreground coefficients and is updated with the larger step size. The foreground filter keeps its normal step size during the trial. The background coefficients are transferred to the foreground filter only after the error computed with them becomes sufficiently smaller than the error computed with the foreground coefficients. Both errors are computed from the measured reference and error signals by using the model of the secondary path. The trial is discarded if this condition is not met within a fixed number of blocks. Because the background filter does not contribute to the control signal, a discarded trial leaves the control signal identical to that obtained without the trial, given the same initial state and input. In contrast, a transfer can also be confirmed when the primary path stays the same. Such a transfer replaces the foreground coefficients and thus affects the control signal. We report how often such transfers occurred.

The trial is useful only if the background filter reduces the error within the trial. For this purpose, the update combines two elements, and we analyze them for finite block lengths. First, the update direction is based on the spectral factor of the filtered-reference power spectrum [Elliott and Rafaely 2000], and the step size is divided by the power of each filtered-reference frame relative to its running average. We show that, with the coefficients held fixed, the update caused by an isolated impulse in the reference signal does not depend on the amplitude of the impulse. The algorithm therefore uses impulsive samples without clipping them. Second, whenever the larger step size is used, the error frame is recomputed with the current coefficients, as in the modified FxLMS algorithm [Bjarnason 1992]. This recomputation applies to start-up, to the trial, and to a fixed number of blocks after a transfer. We derive the form of this update and select its step size below the boundary between decay and growth that we observed in numerical experiments. Our previous algorithm [APSIPA 2026] had a single filter and used neither element. Ochiai et al. [Ochiai et al. 1977] proposed the structure with foreground and background filters for echo cancellation. We apply this structure to the reconvergence of ANC under impulsive noise, and we analyze and evaluate the combined algorithm.

We evaluated the proposed algorithm in closed-loop simulations with room impulse responses measured in one room [MeshRIR] and with ten noise types (four synthetic signals and six recordings). The recovery time is the time until the short-time error power stays within 3 dB of the level that the same algorithm reaches after the change. Compared with the fastest of the nine comparison algorithms, the proposed algorithm reduced the median recovery time after a switch of the noise source from 0.98 s to 0.36 s for a new source close to the original source and from 1.26 s to 0.40 s for a new source far from it. For machinegun noise in both conditions and for car interior noise with the close source, the comparison algorithm recovered faster. Because the reference level of the recovery time depends on the algorithm, we also measured the noise reduction with the coefficients frozen 0.51 s after the switch. Its median was 7.9 dB for the proposed algorithm and 5.0 dB for the comparison algorithm with the close source, and 10.9 dB and 6.8 dB with the far source. With a fixed primary path and no disturbance at the error microphone, the mean noise reduction of the two algorithms differed by less than 0.1 dB over the full 30 s and over the final 10 s.

The remainder of this paper is organized as follows. Section 2 reviews related work, and Section 3 formulates the problem. Section 4 describes the proposed algorithm and its analysis. Section 5 describes the evaluation framework. Section 6 reports the reconvergence after a primary path change, and Section 7 reports the cost of the trial mechanism when the path does not change. Section 8 concludes the paper.

## 2. Related work

### 2.1 Adaptive algorithms for impulsive noise

The FxLMS algorithm minimizes the mean squared error. For impulsive noise modeled by an α-stable distribution with α < 2, the second-order moment does not exist [Shao and Nikias 1993]. Previous studies therefore modified the update so that large samples have a limited effect on it.

One group of algorithms applies thresholds to the signals. Sun et al. [Sun 2006] ignored the samples that exceeded a threshold during adaptation. Akhtar and Mitsuhashi [Akhtar 2008, Akhtar 2009] instead replaced such samples in the reference and error signals with the threshold values to obtain faster convergence.

A second group replaces the squared error with a cost function that grows slowly for large errors. Wu et al. [Wu 2011] minimized the squared logarithmic transformation of the error. This cost requires no thresholds that depend on the noise statistics. Thanigai et al. [Thanigai 2007] used the three-part M-estimator of Hampel to detect and reject impulses in the error signal. Sun et al. [Sun et al. 2015] extended this algorithm by passing the reference signal through another Hampel M-estimator with thresholds estimated online. Wu and Qiu [Wu and Qiu 2013] reformulated several existing algorithms as M-estimators and proposed an algorithm based on the Fair function. Feng et al. [Feng et al. 2024] derived a normalized algorithm from the fractional lower-order covariance and gave its convergence conditions. Cheng et al. [Cheng 2023] applied the Gaussian error function to the error and combined it with a normalized variable step size based on a sine function.

In our previous work [APSIPA 2026], a frequency-domain FxNLMS algorithm scaled the step size in each frequency bin according to the coherence between the filtered reference and error signals. The scaling factor was limited to a fixed range around the nominal step size.

The algorithms in the two groups bound or compress the contribution of a large error sample to the update. Our previous algorithm keeps a normalized update and changes the step size only within the fixed range. In both cases, one update rule with one set of parameters handles every large error, and these parameters are selected for stability under impulsive noise. The same rule is then applied after a change in the primary path, and it determines the speed of reconvergence (Section 1).

### 2.2 Algorithms with two adaptive filters

Two adaptive filters have been combined to obtain both fast convergence and a small steady-state error. Ferrer et al. [Ferrer 2013] applied the convex combination of two filtered-x filters with different step sizes to ANC. Cheng et al. [Cheng 2022] combined an FxLMS filter with a filter based on the maximum Versoria criterion for impulsive noise. Yu et al. [Yu 2020] combined an FxLMS filter with a filter based on the minimum kernel risk-sensitive loss. In these schemes, a mixing parameter weights the two filters, and both filters contribute to the control signal whenever their weights are nonzero. We include the algorithm of Ferrer et al. in the evaluation (Section 6).

Ochiai et al. [Ochiai et al. 1977] used two filters in a different way. Their echo canceler adapted a background model and used a foreground model to synthesize the echo replica. The model with the smaller residual became the new foreground model. This structure prevented false adaptation during double talk from affecting the echo replica. In echo cancellation, the replica is subtracted electrically, and the residual of each model can be computed directly. In ANC, the control signal reaches the error microphone through the secondary path, and only the error of the foreground filter is measured. The proposed algorithm therefore reconstructs the error of the background filter with the model of the secondary path (Section 2.3).

### 2.3 Frequency-domain update and error reconstruction

Kuo et al. [Kuo 2005] implemented the FxLMS algorithm in blocks to reduce its computational load. Elliott and Rafaely [Elliott and Rafaely 1997, Elliott and Rafaely 2000] showed that the bin-normalized frequency-domain LMS algorithm can converge to a biased solution when the filter is constrained to be causal. They applied the spectral factors of the reference power spectrum before and after the causality constraint, and the resulting algorithm converged to the optimal causal filter. For feedforward control, the factorization also includes the minimum-phase part of the plant response. The proposed algorithm follows this construction and factorizes the regularized power spectrum of the filtered reference signal.

Bjarnason [Bjarnason 1992] proposed the modified FxLMS algorithm. This algorithm estimates the primary noise at the error microphone by subtracting the filtered control signal from the measured error. It then computes the error that the current coefficients would produce. This modification removes the delay of the secondary path from the adaptation loop [Yang 2018 review]. The proposed algorithm uses this reconstruction for two purposes. The first is the update with the larger step size during start-up, during a trial, and after a transfer. The second is the comparison between the foreground and background filters.

## 3. Problem formulation

### 3.1 Signal model

Fig. 1 shows the single-channel feedforward ANC system considered in this paper. The reference microphone observes the noise $x(k)$ at sample $k$. The noise reaches the error microphone through the primary path $p(k)$ and produces the primary noise $d(k) = p(k) * x(k)$, where $*$ denotes convolution. The control filter has $M$ coefficients $\mathbf{w}(k) = [w_0(k), \ldots, w_{M-1}(k)]^{\mathsf T}$ and generates the control signal

$$y(k) = \mathbf{w}^{\mathsf T}(k)\,\mathbf{x}(k), \qquad \mathbf{x}(k) = [x(k), \ldots, x(k-M+1)]^{\mathsf T}.$$

The control signal reaches the error microphone through the secondary path $s(k)$. The error signal is

$$e(k) = d(k) + s(k) * y(k) + v(k), \tag{1}$$

where $v(k)$ is a disturbance at the error microphone that is uncorrelated with $x(k)$. A model $\hat{s}(k)$ of the secondary path with $L_s$ coefficients is identified in advance and is fixed during control. The filtered reference signal is $r(k) = \hat{s}(k) * x(k)$.

### 3.2 Block frequency-domain adaptation

The control signal is computed for every sample in the time domain, so that the control path has no block delay. The coefficients are updated once every $B$ samples in the frequency domain with the FFT length $N = 2M$. In block $n$, $\mathbf{R}_n = \hat{\mathbf{S}} \odot \mathbf{X}_n$ denotes the spectrum of the filtered reference frame, where $\mathbf{X}_n$ is the FFT of the latest $N$ samples of $x(k)$ and $\hat{\mathbf{S}}$ is the FFT of $\hat{s}(k)$. Similarly, $\mathbf{E}_n$ denotes the FFT of the latest $M$ samples of $e(k)$ preceded by $N - M$ zeros. The frequency-domain FxNLMS algorithm updates the coefficients as

$$\mathbf{w}_{n+1} = \mathbf{w}_n - \mu\, \mathcal{P}_M\, \mathcal{F}^{-1}\!\left\{ \frac{\mathbf{R}_n^{*} \odot \mathbf{E}_n}{\hat{\mathbf{S}}_{rr,n} + \epsilon} \right\}, \tag{2}$$

where $\mu$ is the step size, $\hat{\mathbf{S}}_{rr,n}$ is the exponential average of $|\mathbf{R}_n|^2$, $\epsilon$ is a regularization constant, $\odot$ and the division are element-wise, ${}^{*}$ denotes complex conjugation, $\mathcal{F}^{-1}$ is the inverse FFT, and $\mathcal{P}_M$ keeps the first $M$ samples. The projection $\mathcal{P}_M$ constrains the filter to be causal with $M$ coefficients. The proposed algorithm replaces the direction and the step size in (2), as described in Section 4.

### 3.3 Impulsive noise and primary path change

The noise $x(k)$ is impulsive. Its samples occasionally have amplitudes much larger than the background level, and the same impulses appear in $d(k)$ and, after imperfect cancellation, in $e(k)$. We place no assumption on the distribution of $x(k)$ in the algorithm. The evaluation uses a Gaussian signal, a Gaussian signal with isolated impulses, symmetric α-stable signals, and recorded noise (Section 5).

The primary path changes at an unknown sample $k_c$:

$$p(k) = \begin{cases} p_1(k), & k < k_c, \\ p_2(k), & k \ge k_c. \end{cases} \tag{3}$$

Such a change occurs when the dominant noise source switches to a source at another position. The optimal control filter then changes from the solution for $p_1$ to the solution for $p_2$, and the error increases until the control filter reconverges. The secondary path $s(k)$ and its model $\hat{s}(k)$ stay the same before and after $k_c$. A change in the secondary path requires the model $\hat{s}(k)$ to be identified again, and this paper focuses only on the change in the primary path.

### 3.4 Requirements

We require the following three properties of the adaptive algorithm under impulsive noise.

1. Reconvergence: after the change in (3), the error power returns to its converged level in a short time. Section 5 defines the recovery time that measures this property.
2. Steady state: when the primary path stays the same, the mechanism for fast reconvergence leaves the noise reduction at the level obtained without it.
3. Disturbance: when the error increases because of $v(k)$ in (1) and the primary path stays the same, the mechanism likewise leaves the noise reduction at the level obtained without it.

The first requirement calls for a large update after $k_c$, and the second and third requirements call for the normal update at all other times. The algorithm observes only $x(k)$ and $e(k)$, and $k_c$ is unknown. Section 4 describes the mechanism, and Section 7 measures the second and third requirements against the same algorithm with the mechanism disabled.

[3 章のメモ]
- Fig. 1：系の構成図（x、p、w、s、ŝ、e、v、前景と背景は Fig. 2）。作図が必要。
- 式 (2) の誤差フレームと参照フレームの取り方は Table 1（誤差窓 512、FFT 1024）と THEORY_frozen.md §1 に合わせた。Ŝ_rr の推定法（指数平均の係数）は 4 章で書く。
- 3.3 節「分布に仮定を置かない」：1/g の解析（定理 C）は孤立インパルスを仮定するので、4 章でその仮定を明記する。
- 要件 2 と 3 は「仕組みを外した同じ手法（再始動なし）と同じ水準」と定義した。結果は差 0.01 dB 以下で区間は 0 を含む（変化なし、EN0、ENm10）。比較手法との差（外乱の直後 2 秒は CohVSS-FxNLMS が 0.62 dB、C-FXLMS が 0.27 dB 良い）は、要件とは別に 7 章で書く。

## 4. Proposed algorithm

### 4.1 Overview

Fig. 2 shows the structure of the proposed algorithm. The foreground filter $\mathbf{w}_{\mathrm f}$ generates the control signal, and the background filter $\mathbf{w}_{\mathrm b}$ exists only during a trial. Both filters use the update

$$\mathbf{w}_{n+1} = \mathbf{w}_n - \frac{\sigma_n}{\max(g_n, g_{\min})}\,\frac{2B}{M}\,\boldsymbol{\delta}_n, \tag{4}$$

where $\boldsymbol{\delta}_n$ is the update direction (Section 4.2), $g_n$ is the relative power of the filtered-reference frame (Section 4.3), $g_{\min}$ is its lower bound, and $\sigma_n$ is the step size. The update has two phases. In the early phase, $\sigma_n = \sigma_{\mathrm E}$ is large, and the error frame is recomputed with the current coefficients (Section 4.4). In the late phase, $\sigma_n = \sigma_{\mathrm L}$ is small, and the measured error frame is used (Section 4.5). The foreground filter is in the early phase for the first $n_{\mathrm E}$ blocks after start-up and after each transfer, and it is in the late phase at all other times. The background filter is always in the early phase. Section 4.6 describes when a trial starts and how it ends. Table 1 lists the parameter values.

### 4.2 Update direction

The direction in (2) normalizes each frequency bin by the power spectrum before the causality constraint. Elliott and Rafaely [Elliott and Rafaely 2000] showed that this order biases the solution, and they split the normalization into two spectral factors. We follow this construction. Let $\boldsymbol{\Lambda}_n = \hat{\mathbf{S}}_{rr,n} + \epsilon \bar{P}_n$, where $\bar{P}_n$ is the mean of $\hat{\mathbf{S}}_{rr,n}$ over the frequency bins, and let $\mathbf{F}_n$ be the minimum-phase factor with $|\mathbf{F}_n|^2 = \boldsymbol{\Lambda}_n$. We compute $\mathbf{F}_n$ from the real cepstrum of $\boldsymbol{\Lambda}_n$. The direction is

$$\boldsymbol{\delta}_n = \mathcal{P}_M\, \mathcal{F}^{-1}\!\left\{ \frac{1}{\mathbf{F}_n} \odot \mathcal{F}\, \mathcal{P}_{N/2}\, \mathcal{F}^{-1}\!\left\{ \frac{\mathbf{R}_n^{*} \odot \mathbf{E}_n}{\mathbf{F}_n^{*}} \right\} \right\}, \tag{5}$$

where $\mathcal{P}_{N/2}$ keeps the first $N/2$ samples and sets the others to zero. The division by $\mathbf{F}_n^{*}$ whitens the cross-correlation, $\mathcal{P}_{N/2}$ removes its noncausal part, and the division by $\mathbf{F}_n$ maps the result back to the coefficient domain.

**Proposition 1.** Assume fixed coefficients $\mathbf{w}$, a stationary $x(k)$, $\epsilon = 0$, and $N \to \infty$. Then

$$\mathrm{E}[\boldsymbol{\delta}_n] = \frac{M_e}{N}\left(\mathbf{w} - [\mathbf{w}_\infty]_M\right),$$

where $M_e$ is the length of the error window ($M_e = M$ in this paper), $\mathbf{w}_\infty$ is the causal Wiener filter of infinite length, and $[\cdot]_M$ keeps its first $M$ coefficients.

The proof is given in Appendix A. Under these assumptions, the mean update acts as a scalar on the coefficient error, so that all modes of the coefficient error converge at the same rate regardless of the spectrum of $r(k)$. For the finite length $N = 2M$ and $\epsilon = 10^{-2}$ used in this paper, we evaluated the deviation from this scalar action numerically. The eigenvalues of the mean action, normalized by $M_e/N$, lay between 0.13 and 1.0, and this spread was caused mainly by the regularization.

### 4.3 Step size normalized by the relative frame power

The factor $\mathbf{F}_n$ follows the exponential average $\hat{\mathbf{S}}_{rr,n}$, which changes slowly. When a frame contains an impulse, its power is much larger than the average, and $\boldsymbol{\delta}_n$ grows with the square of the impulse amplitude. We therefore divide the step size by the relative frame power

$$g_n = \frac{\frac{1}{N/2+1}\sum_{f} |R_n(f)|^2}{\bar{P}_n}. \tag{6}$$

**Proposition 2.** Assume that the coefficients, $\mathbf{F}_n$, and $\bar{P}_n$ are fixed, that $v(k) = 0$, and that $x(k) = c(k) * [\varsigma(k)\xi(k)]$, where $c(k)$ is a fixed filter, $\xi(k)$ is a unit-variance white sequence, and $\varsigma(k)$ is a deterministic scale. Then each element of $\boldsymbol{\delta}_n$ is a quadratic form of the samples $z(k) = \varsigma(k)\xi(k)$ in the frame. If only one sample $z(k_0)$ in the frame is nonzero, then $\boldsymbol{\delta}_n = z^2(k_0)\,\mathbf{k}(k_0)$ and $g_n = z^2(k_0)\,\kappa(k_0)$ with a vector $\mathbf{k}(k_0)$ and a scalar $\kappa(k_0)$ that are independent of $z(k_0)$. The update $\boldsymbol{\delta}_n / g_n$ is therefore independent of the amplitude of the impulse.

The proof is given in Appendix B. Proposition 2 concerns an isolated impulse, and it shows that the update caused by such an impulse is bounded without a threshold. The algorithm therefore keeps the impulsive samples in the update. For frames that contain both an impulse and background noise, the ratio $\boldsymbol{\delta}_n/g_n$ depends on the position of the impulse in the frame, because $g_n$ measures the power of the whole frame and $\boldsymbol{\delta}_n$ depends on the part of the response inside the error window. We rely on the simulations for these cases. The lower bound $g_{\min}$ in (4) limits the step size for frames with low power.

### 4.4 Early phase with recomputed error frames

The error window of block $n$ contains $M_e$ samples, and these samples were produced by the coefficients of the last $M_e/B$ blocks. The measured error frame therefore responds to an update with a delay, and this delay limits the step size. In the early phase, we remove this dependence on past coefficients. As in the modified FxLMS algorithm [Bjarnason 1992], the primary noise is estimated as

$$\hat{d}(k) = e(k) - \hat{s}(k) * y(k), \tag{7}$$

and the error frame is recomputed with the coefficients $\mathbf{w}_n$ to be updated:

$$\tilde{e}_n(k) = \hat{d}(k) + \hat{s}(k) * \left[\mathbf{w}_n^{\mathsf T}\mathbf{x}(k)\right] \tag{8}$$

for the $M_e$ samples in the window. $\mathbf{E}_n$ in (5) is then the FFT of $\tilde{e}_n(k)$.

**Proposition 3.** Assume $\hat{s} = s$, ideal whitening ($\boldsymbol{\Lambda}_n = \mathbf{S}_{rr}$, $N \to \infty$), and $g_n = 1$. Let $\mathbf{v}_n = \mathbf{w}_n - \mathbf{w}_o$ be the coefficient error with respect to a reference filter $\mathbf{w}_o$, and let $a = \sigma_{\mathrm E}\,(2B/M)(M_e/N)$. With the recomputed error frame,

$$\mathbf{v}_{n+1} = (\mathbf{I} - a\,\mathbf{G}_n)\,\mathbf{v}_n + \mathbf{b}_n, \qquad \mathbf{G}_n = \mathbf{F}_M^{-1}\,\mathbf{U}_n^{\mathsf T}\mathbf{X}^{r}_n / M_e, \tag{9}$$

where $\mathbf{U}_n$ and $\mathbf{X}^r_n$ are the $M_e \times M$ regression matrices of the whitened filtered reference and of the filtered reference in the window, $\mathbf{F}_M$ is the $M \times M$ convolution matrix of the spectral factor, and $\mathbf{b}_n$ is the contribution of the residual error of $\mathbf{w}_o$.

The proof is given in Appendix C. Equation (9) contains only the current coefficient error. The matrices $\mathbf{G}_n$ of consecutive blocks share samples, and we have no closed-form bound on $a$ for the stability of (9). We therefore examined (9) numerically for white Gaussian input over 64 blocks. The coefficient error decayed for $a \le 0.55$ and grew for $a \ge 0.625$, and the decay after 32 blocks was fastest near $a = 0.375$. Based on this result, we set $\sigma_{\mathrm E} = 1$ ($a = 0.25$ for $g_n = 1$) and $g_{\min} = 0.55$ in the early phase, so that $a/\max(g_n, g_{\min}) \le 0.455$.

### 4.5 Late phase

In the late phase, the measured error frame is used, and the step size is small. From a second-order recursion for small step sizes (Appendix D), the ratio of the excess error to the minimum error in the steady state is approximately

$$\delta \approx \frac{a\,\tau}{2 - a\,(2M/B)}, \qquad \tau = M_e/B, \quad a = \sigma_{\mathrm L}\,(2B/M)(M_e/N). \tag{10}$$

We solve (10) for $\sigma_{\mathrm L}$ with a design value of $\delta$. For $\delta = 0.03$, $\sigma_{\mathrm L} = 0.0566$. The lower bound of $g_n$ in the late phase is $g_{\min} = 0.25$.

### 4.6 Trial with the background filter

**Start of a trial.** For each block, the algorithm computes the ratio

$$z_n = \frac{\sum_{k \in \mathcal{W}_M(n)} e^2(k)}{\sum_{k \in \mathcal{W}_{2M}(n)} r^2(k)}, \tag{11}$$

where $\mathcal{W}_L(n)$ denotes the latest $L$ samples at block $n$. The division by the power of $r(k)$ removes the common scale of the input from the ratio. Let $\bar{z}_n$ be the exponential average of $z_n$. A trial starts at block $n_0$ when $z_n > \gamma\,\bar{z}_{n-1}$ holds for two consecutive blocks. After a trial starts, the next trial can start only after $n_{\mathrm G}$ blocks, and $\bar{z}_n$ is held constant during this interval.

**Adaptation.** At block $n_0$, the background filter is set to $\mathbf{w}_{\mathrm b} = \mathbf{w}_{{\mathrm f}, n_0}$. From block $n_0 + 1$, it is updated by (4) in the early phase, with the error frame recomputed by (8) with $\mathbf{w}_{\mathrm b}$. The foreground filter continues its update in its current phase, and the control signal is generated by $\mathbf{w}_{\mathrm f}$ only.

**Decision.** For each block $n \ge n_0 + 2$, the algorithm recomputes the errors of both filters by (8) for the latest $B$ samples and forms the ratio

$$\rho_n = \frac{\sum_{k \in \mathcal{W}_B(n)} \tilde{e}_{\mathrm b}^2(k)}{\sum_{k \in \mathcal{W}_B(n)} \tilde{e}_{\mathrm f}^2(k)}. \tag{12}$$

Both errors in (12) use the same $\hat{d}(k)$, so that a common scale of the signals cancels in the ratio. The trial ends by one of the following rules.

1. Discard: if $\rho_n > \rho_{\mathrm{abort}}$ or $\rho_n$ is not finite, $\mathbf{w}_{\mathrm b}$ is discarded.
2. Early transfer: if $n \ge n_0 + n_1$ and $\rho_n \le \rho_1$, the coefficients are transferred, $\mathbf{w}_{\mathrm f} \leftarrow \mathbf{w}_{\mathrm b}$.
3. Final decision: at $n = n_0 + n_2$, the coefficients are transferred if $\rho_n \le \rho_2$, and $\mathbf{w}_{\mathrm b}$ is discarded otherwise.

After a transfer, the foreground filter continues in the early phase for the remaining $n_{\mathrm E} - (n - n_0)$ blocks. We use $\rho_{\mathrm{abort}} = 4$, $(n_1, \rho_1) = (4, 0.25)$, and $(n_2, \rho_2) = (16, 0.5)$.

**Proposition 4.** Consider two runs of the algorithm with the same initial state and the same input signals. In the first run, the trials are disabled. In the second run, every trial up to block $n$ is discarded. Then the control signal $y(k)$ and the foreground coefficients of the two runs are identical up to block $n$.

*Proof.* The control signal and the update of $\mathbf{w}_{\mathrm f}$ depend on $x(k)$, $e(k)$, and $\mathbf{w}_{\mathrm f}$. The background filter and the quantities $z_n$, $\bar{z}_n$, and $\rho_n$ are computed from these signals and affect $\mathbf{w}_{\mathrm f}$ only through a transfer. Without a transfer, the recursions of $\mathbf{w}_{\mathrm f}$ and $y(k)$ in the two runs are the same, and their initial states are the same. $\blacksquare$

Proposition 4 requires no assumption on the signals or on the accuracy of $\hat{s}(k)$. It limits the effect of a trial without a path change to the computation during at most $n_2$ blocks, provided that the trial is discarded. A transfer without a path change is outside the scope of Proposition 4, and it replaces the foreground coefficients. The thresholds in (12) determine how often such a transfer occurs, and we report its frequency in Section 7. Conversely, a path change produces no speed-up if the ratio in (11) stays below its threshold or if the trial is discarded. We report these cases in Section 6.

### 4.7 Computational cost

A block in the late phase without a trial requires 118,016 real multiplications, and a block with a trial requires 241,920. For comparison, the time-domain FxNLMS algorithm with the same filter lengths requires 196,608 multiplications for $B$ samples. These numbers include the sample-by-sample filtering for the control signal. Section 7 gives the details.

[4 章のメモ]
- 命題 1 = THEORY 定理 A、命題 2 = 定理 C（系 C2）、命題 3 = 補題 F、命題 4 = 命題 H。本文は「命題」に統一した。
- 命題 1 の数値（固有値 0.13〜1.0）は THEORY §2。命題 3 の数値は theory/v4/mp.py と tail.py（白色ガウス、3 系列、64 ブロック）。
- 式 (10) は THEORY の μ_L(δ) = 2δ/(h(τ + (2M/B)δ)) を a = σ_L h で書き直したもの。δ = 0.03 で 0.0566 を確認。
- 検出の指数平均の係数 0.05、γ = 4、n_G = 64、n_E = 32 は Table 1。
- 4.7 節：118,016／241,920／196,608 は RESULTS_fast.md と一致。Table 3 の "Ref." 列（284,672 など）は効率化前の実装の値で、本文の基準と違う。Table 3 を「提案法（効率的な実装）と FxNLMS」の比較に直す必要がある。
- 付録 A〜D（証明）は未執筆。

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
