# Draft v4 (abstract, Sections 1–8)

v3 を査読者の目で見直して直した版。[..] は著者用のメモ。

## v3 から直した点

| # | v3 の問題 | v4 での扱い |
|---|---|---|
| 1 | 「大きな標本の抑制が再収束を遅くする」という因果は、自分のデータと合わない。抑制のない FxNLMS は回復 415/537 ブロックで、抑制のある手法より遅い（TUNE.json：発散を避けるためステップが公開値の 1/16、旧 Th_FxNLMS はしきい値 200 で実質クリップなし。現在の Th-FxNLMS は atsip/thx の Th2_FxNLMS（パーセンタイルによるクリップ）） | 「発散を防ぐ設定（しきい値・コスト・小さいステップ）が、原因を区別せずに同じ更新則として働き、再収束の速さも決める」に改めた（要旨、1 章、2.1 節） |
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

Adaptive algorithms for active control of impulsive noise prevent divergence by limiting the influence of large samples or by using a small step size. Because these algorithms treat every large error in the same way, the same settings also slow the reconvergence after a sudden change in the primary path. This paper proposes a frequency-domain filtered-x algorithm that adapts a background filter with a larger step size in a trial. Its coefficients are transferred to the foreground filter, which generates the control signal, only when they give a sufficiently smaller error on the same data. A discarded trial leaves the control signal identical to that obtained without the trial, given the same initial state and input. In simulations with measured room impulse responses and ten noise types, the proposed algorithm reduced the median recovery time after a switch of the noise source by 63% to 68% compared with the fastest of ten comparison algorithms, and its coefficients 0.5 s after the switch reduced the noise by 2.7 dB to 3.8 dB more. The advantage did not extend to machine gun noise, which consists of intermittent bursts and lies outside the continuously present noise that the algorithm targets. With a fixed primary path, the mean noise reduction differed from that of this comparison algorithm by less than 0.1 dB.

[200 語。63% = 1 − 0.36/0.98（close）、68% = 1 − 0.40/1.26（far）。要旨の例外は machine gun に絞った。car interior（close）の例外は 1 章と 6.2 節に記載。]

## 1. Introduction

Impulsive noise contains short bursts with amplitudes much larger than the background level, and it is often modeled by a symmetric α-stable distribution [Shao and Nikias 1993]. In feedforward active noise control (ANC), the filtered-x least-mean-square (FxLMS) algorithm updates the control filter in proportion to the product of the filtered reference signal and the error signal [Kuo and Morgan 1996]. An impulsive sample in either signal can therefore make the control filter diverge.

Previous studies prevented this divergence by limiting the influence of large samples with thresholds [Sun 2006, Akhtar 2009] or with cost functions that grow slowly for large errors [Wu 2011, Wu and Qiu 2013, Feng et al. 2024], by controlling the step size [APSIPA 2026], and by combining a fast and a slow filter [Ferrer 2013, Cheng 2022, Yu 2020]. Section 2 reviews them. In all these algorithms, every update acts on the filter that generates the control signal. A sudden change in the primary path, such as a switch to another noise source, also increases the error, and the error remains large until the control filter adapts to the new path. The thresholds, cost functions, and step sizes selected for stability under impulses therefore also determine the speed of this reconvergence. The convex combination does not remove this limitation, because its fast filter also generates the control signal. In our simulations, the fastest of ten comparison algorithms with step sizes selected to avoid divergence, our previous algorithm [APSIPA 2026], needed about 1 s in median to recover after a switch of the noise source. Larger step sizes shortened this time only together with divergence.

This paper addresses this reconvergence in a single-channel feedforward ANC system. We assume that the noise is present continuously and contains impulses, and that the secondary path and its model stay the same. The research question is how to apply a larger step size after a change in the primary path, when the error alone cannot distinguish such a change from impulses or from a disturbance at the error microphone.

The proposed algorithm applies the larger step size first to a background filter, which does not contribute to the control signal. The foreground filter generates the control signal, and it receives the background coefficients only when they give a smaller error than its own coefficients on the same data. We call this adaptation of the background filter a trial. A discarded trial leaves the control signal identical to that obtained without the trial, given the same initial state and input. A transfer that is confirmed without a path change can alter the control signal, and we report how often it occurred.

Foreground and background filters were introduced for echo cancellation [Ochiai et al. 1977]. For ANC, a patent describes a shadow filter whose error is reconstructed with the secondary-path model and whose coefficients are copied to the control filter when this error is smaller [Harman 2022]. The proposed algorithm uses the same reconstruction for its decision. It differs in when and how the background filter is adapted. A trial starts only when the ratio of the error power to the filtered-reference power rises, it ends within 16 blocks, and the background filter is updated with a step size that the foreground filter cannot use under impulsive noise. This update combines three elements: a direction based on the spectral factor of the filtered-reference power spectrum [Elliott and Rafaely 2000], a step size divided by the relative power of each filtered-reference frame, and an error frame recomputed with the current coefficients, as in the modified FxLMS algorithm [Bjarnason 1992]. The proposed algorithm shares the block frequency-domain structure with our previous algorithm [APSIPA 2026], and it does not use the coherence-based step size.

The contributions of this paper are as follows.

1. We design a short trial with a background filter that isolates a large step size from the control signal under impulsive noise. A discarded trial leaves the control signal unchanged (Proposition 2), and we bound the error of the transfer decision caused by an inaccurate secondary-path model (Proposition 3).
2. We combine three elements in the update used for the trial. We show that, for a white input, the update caused by an isolated impulse does not depend on its amplitude (Proposition 1), and we show by ablation which elements the fast and stable reconvergence requires.
3. We compare the proposed algorithm with ten algorithms for ten noise types. The evaluation includes the cases in which no transfer occurred, the case in which comparison algorithms recovered better from a disturbance, and the effect of an error in the secondary-path model.

Section 2 reviews related work, Section 3 formulates the problem, and Section 4 describes the proposed algorithm. Section 5 describes the evaluation framework, Sections 6 and 7 report the results, and Section 8 concludes the paper.

## 2. Related work

### 2.1 Adaptive algorithms for impulsive noise

For impulsive noise with α < 2, the second-order moment does not exist [Shao and Nikias 1993], and previous studies limited the effect of large samples on the update. Sun et al. [Sun 2006] ignored the samples that exceeded a threshold. Akhtar and Mitsuhashi [Akhtar 2008, Akhtar 2009] replaced such samples with the threshold values. Wu et al. [Wu 2011] minimized the squared logarithm of the error. Thanigai et al. [Thanigai 2007] rejected impulses in the error signal with the M-estimator of Hampel. Sun et al. [Sun et al. 2015] applied another Hampel M-estimator to the reference signal. Wu and Qiu [Wu and Qiu 2013] proposed an algorithm based on the Fair function. Feng et al. [Feng et al. 2024] derived a normalized algorithm from the fractional lower-order covariance. Cheng et al. [Cheng 2023] applied the Gaussian error function to the error. Our previous algorithm [APSIPA 2026] scaled the step size in each frequency bin by the coherence between the filtered reference and error signals, within a fixed range. In all these algorithms, one update rule with parameters selected for stability handles every large error.

### 2.2 Algorithms with two adaptive filters

Two adaptive filters have been combined to obtain both fast convergence and a small steady-state error. Ferrer et al. [Ferrer 2013] applied the convex combination of two filtered-x filters with different step sizes to ANC. Cheng et al. [Cheng 2022] combined an FxLMS filter with a filter based on the maximum Versoria criterion. Yu et al. [Yu 2020] combined an FxLMS filter with a filter based on the minimum kernel risk-sensitive loss. In these schemes, both filters contribute to the control signal whenever their weights are nonzero.

Ochiai et al. [Ochiai et al. 1977] adapted a background echo model and synthesized the echo replica with a foreground model. The model with the smaller residual became the new foreground model, so that false adaptation during double talk did not affect the replica. Lindström et al. [Lindstrom 2007] and Iqbal and Grant [Iqbal 2007] later improved the conditions for this transfer. In ANC, only the error of the foreground filter is measured. A patent of Harman Becker Automotive Systems [Harman 2022] describes a shadow filter for ANC whose error is formed from the estimated disturbance and the output of a secondary-path model. The shadow coefficients are copied to the control filter when this error is smaller than the measured error. The proposed algorithm reconstructs the errors in the same way. In contrast to the shadow filter, which adapts continuously, its background filter exists only during a short trial that a detector starts, and it uses an update designed for fast convergence under impulsive noise.

### 2.3 Frequency-domain update and error reconstruction

Elliott and Rafaely [Elliott and Rafaely 1997, Elliott and Rafaely 2000] showed that the bin-normalized frequency-domain LMS algorithm converges to a biased solution when the filter is constrained to be causal. They removed this bias by applying spectral factors of the reference power spectrum before and after the constraint. Bjarnason [Bjarnason 1992] proposed the modified FxLMS algorithm, which estimates the primary noise at the error microphone and computes the error that the current coefficients would produce. This modification removes the delay of the secondary path from the adaptation loop [Yang 2018 review]. Section 4 uses both constructions.

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

where $\mu$ is the step size, $\hat{\mathbf{S}}_{rr,n} = (1-\beta)\hat{\mathbf{S}}_{rr,n-1} + \beta|\mathbf{R}_n|^2$ is the exponential average of the power spectrum with the weight $\beta$, $\epsilon$ is a regularization constant, $\odot$ and the division are element-wise, ${}^{*}$ denotes complex conjugation, $\mathcal{F}^{-1}$ is the inverse FFT, and $\mathcal{P}_M$ keeps the first $M$ samples. The projection $\mathcal{P}_M$ constrains the filter to be causal with $M$ coefficients. The proposed algorithm replaces the direction and the step size in (2), as described in Section 4.

### 3.3 Impulsive noise and primary path change

The noise $x(k)$ is impulsive, and its impulses appear in $d(k)$ and in $e(k)$. The algorithm places no assumption on the distribution of $x(k)$. The primary path changes at an unknown sample $k_c$:

$$p(k) = \begin{cases} p_1(k), & k < k_c, \\ p_2(k), & k \ge k_c. \end{cases} \tag{3}$$

Such a change occurs when the dominant noise source switches to a source at another position, and the error increases until the control filter reconverges. We assume that the secondary path $s(k)$ and its model $\hat{s}(k)$ stay the same. A change in the secondary path requires the model to be identified again and is outside the scope of this paper.

### 3.4 Requirements

We require three properties of the adaptive algorithm.

1. Reconvergence: after the change in (3), the error power returns to its converged level in a short time.
2. Steady state: with a fixed primary path, the mechanism for fast reconvergence leaves the noise reduction at the level obtained without it.
3. Disturbance: the second property also holds when $v(k)$ in (1) increases the error.

The first requirement calls for a large update after $k_c$, and the other two call for the normal update at all other times, although $k_c$ is unknown.

[3 章のメモ]
- Fig. 1：系の構成図（x、p、w、s、ŝ、e、v、前景と背景は Fig. 2）。作図が必要。
- 式 (2) の誤差フレームと参照フレームの取り方は Table 1（誤差窓 512、FFT 1024）と THEORY_frozen.md §1 に合わせた。Ŝ_rr の推定法（指数平均の係数）は 4 章で書く。
- 3.3 節「分布に仮定を置かない」：1/g の解析（定理 C）は孤立インパルスを仮定するので、4 章でその仮定を明記する。
- 要件 2 と 3 は「仕組みを外した同じ手法（再始動なし）と同じ水準」と定義した。結果は差 0.01 dB 以下で区間は 0 を含む（変化なし、EN0、ENm10）。比較手法との差（外乱の直後 2 秒は CohVSS-FxNLMS が 0.62 dB、C-FXLMS が 0.27 dB 良い）は、要件とは別に 7 章で書く。

## 4. Proposed algorithm

### 4.1 Overview

Fig. 2 shows the structure of the proposed algorithm. The foreground filter $\mathbf{w}_{\mathrm f}$ generates the control signal, and the background filter $\mathbf{w}_{\mathrm b}$ exists only during a trial. Both filters use the update

$$\mathbf{w}_{n+1} = \mathbf{w}_n - \frac{\sigma_n}{\max(g_n, g_{\min})}\,\frac{2B}{M}\,\boldsymbol{\delta}_n, \tag{4}$$

where $\boldsymbol{\delta}_n$ is the update direction (Section 4.2), $g_n$ is the relative power of the filtered-reference frame (Section 4.3), $g_{\min}$ is its lower bound, and $\sigma_n$ is the step size. The update has two phases. In the early phase, $\sigma_n = \sigma_{\mathrm E}$ is large, and the error frame is recomputed with the current coefficients (Section 4.4). In the late phase, $\sigma_n = \sigma_{\mathrm L}$ is small, and the measured error frame is used (Section 4.5). The foreground filter is in the early phase for the first $n_{\mathrm E}$ blocks after start-up and, after a transfer, until $n_{\mathrm E}$ blocks have passed since the start of the trial. It is in the late phase at all other times. The background filter is always in the early phase. Section 4.6 describes when a trial starts and how it ends. Table 1 lists the parameter values.

### 4.2 Update direction

The direction in (2) normalizes each frequency bin by the power spectrum before the causality constraint. Elliott and Rafaely [Elliott and Rafaely 2000] showed that this order biases the solution, and they split the normalization into two spectral factors. We follow this construction. Let $\boldsymbol{\Lambda}_n = \hat{\mathbf{S}}_{rr,n} + \epsilon \bar{P}_n$, where $\bar{P}_n$ is the mean of $\hat{\mathbf{S}}_{rr,n}$ over the frequency bins, and let $\mathbf{F}_n$ be the minimum-phase factor with $|\mathbf{F}_n|^2 = \boldsymbol{\Lambda}_n$. We compute $\mathbf{F}_n$ from the real cepstrum of $\boldsymbol{\Lambda}_n$. The direction is

$$\boldsymbol{\delta}_n = \mathcal{P}_M\, \mathcal{F}^{-1}\!\left\{ \frac{1}{\mathbf{F}_n} \odot \mathcal{F}\, \mathcal{P}_{N/2}\, \mathcal{F}^{-1}\!\left\{ \frac{\mathbf{R}_n^{*} \odot \mathbf{E}_n}{\mathbf{F}_n^{*}} \right\} \right\}, \tag{5}$$

where $\mathcal{P}_{N/2}$ keeps the first $N/2$ samples and sets the others to zero.

Under idealized conditions (fixed coefficients, $\hat{s} = s$, a stationary input with a known and strictly positive power spectrum, $\epsilon = 0$, and $N \to \infty$), the analysis of Elliott and Rafaely [Elliott and Rafaely 2000] gives $\mathrm{E}[\boldsymbol{\delta}_n] \propto \mathbf{w} - [\mathbf{w}_\infty]_M$, where $\mathbf{w}_\infty$ is the causal Wiener filter of infinite length and $[\cdot]_M$ keeps its first $M$ coefficients. All modes of the coefficient error then converge at the same rate. This property motivates the spectral factors and does not establish convergence of the algorithm, in which $\hat{\mathbf{S}}_{rr,n}$ is estimated from the same data. For $N = 2M$ and $\epsilon = 10^{-2}$, the eigenvalues of the mean action lay between 0.13 and 1.0 of their ideal value in a numerical evaluation, mainly because of the regularization.

### 4.3 Step size normalized by the relative frame power

The exponential average $\hat{\mathbf{S}}_{rr,n}$ gives the current frame the weight $\beta$. When a frame contains an impulse, its power is much larger than the average, and $\boldsymbol{\delta}_n$ grows with the impulse amplitude. We therefore divide the step size by the relative frame power

$$g_n = \frac{\frac{1}{N/2+1}\sum_{f} |R_n(f)|^2}{\bar{P}_n}. \tag{6}$$

Because $\bar{P}_n$ includes the current frame, $g_n \le 1/\beta$.

**Proposition 1.** Assume that the coefficients are fixed, that $v(k) = 0$, and that $\hat{\mathbf{S}}_{rr,n-1} = q\,|\hat{\mathbf{S}}|^2$ with a constant $q$, as for a white $x(k)$. If $x(k_0) = z$ is the only nonzero input sample that contributes to the reference and error frames of block $n$, then $\boldsymbol{\delta}_n / g_n$ does not depend on $z$.

*Proof.* The spectra $\mathbf{R}_n$ and $\mathbf{E}_n$ are proportional to $z$, and $|\mathbf{R}_n|^2 = z^2|\hat{\mathbf{S}}|^2$. Then $\hat{\mathbf{S}}_{rr,n} = c\,|\hat{\mathbf{S}}|^2$ with $c = (1-\beta)q + \beta z^2$, so that $\boldsymbol{\Lambda}_n$ and $\bar{P}_n$ are proportional to $c$ and $\mathbf{F}_n$ is proportional to $\sqrt{c}$. By (5) and (6), both $\boldsymbol{\delta}_n$ and $g_n$ are proportional to $z^2/c$, and their ratio is independent of $z$. $\blacksquare$

An isolated impulse therefore produces the same update for any amplitude as long as $g_n \ge g_{\min}$, and the algorithm keeps impulsive samples in the update without a threshold. For $g_n < g_{\min}$, the update is smaller. For a colored $x(k)$, two weaker statements hold for any positive $\hat{\mathbf{S}}_{rr,n-1}$. Write $\mathbf{R}_n = z\,\mathbf{R}^{(1)}$ and $\mathbf{E}_n = z\,\mathbf{E}^{(1)}$, and let $p$ be the mean of $|\mathbf{R}^{(1)}|^2$ over the bins. First, as $z \to \infty$, $\boldsymbol{\Lambda}_n / z^2$ approaches $\beta(|\mathbf{R}^{(1)}|^2 + \epsilon p)$, and $\boldsymbol{\delta}_n/g_n$ approaches the direction (5) computed with $\boldsymbol{\Lambda} = |\mathbf{R}^{(1)}|^2 + \epsilon p$, which depends neither on $z$ nor on the past spectrum. Second, because $\boldsymbol{\Lambda}_n \ge \epsilon \bar{P}_n$ in every bin and the projections in (5) do not increase the norm,

$$\left\| \boldsymbol{\delta}_n / g_n \right\|_2 \le \frac{\left\| \mathcal{F}^{-1}\{\mathbf{R}^{(1)*} \odot \mathbf{E}^{(1)}\} \right\|_2}{\epsilon\, p}$$

for every $z$. This bound is loose. In a numerical example with an autoregressive past spectrum (pole at 0.95), the norm of $\boldsymbol{\delta}_n/g_n$ changed by a factor of two between small and large $z$. When a frame also contains background noise, we rely on the simulations.

### 4.4 Early phase with recomputed error frames

The $M_e$ samples in the error window of block $n$ were produced by the coefficients of the last $M_e/B$ blocks, so that the measured error frame responds to an update with a delay that limits the step size. In the early phase, we remove this dependence on past coefficients. As in the modified FxLMS algorithm [Bjarnason 1992], the primary noise is estimated as

$$\hat{d}(k) = e(k) - \hat{s}(k) * y(k), \tag{7}$$

and the error frame is recomputed with the coefficients $\mathbf{w}_n$ to be updated:

$$\tilde{e}_n(k) = \hat{d}(k) + \hat{s}(k) * \left[\mathbf{w}_n^{\mathsf T}\mathbf{x}(k)\right] \tag{8}$$

for the $M_e$ samples in the window. $\mathbf{E}_n$ in (5) is then the FFT of $\tilde{e}_n(k)$.

The recomputed frame changes the form of the update. Assume $\hat{s} = s$, ideal whitening ($\boldsymbol{\Lambda}_n = \mathbf{S}_{rr}$, $N \to \infty$), and $g_n = 1$. Let $\mathbf{v}_n = \mathbf{w}_n - \mathbf{w}_o$ be the coefficient error with respect to a reference filter $\mathbf{w}_o$, and let $a = \sigma_{\mathrm E}\,(2B/M)(M_e/N)$. Under these assumptions, (8) gives $\tilde{\mathbf{e}}_n = \mathbf{X}^r_n \mathbf{v}_n + \mathbf{e}_o$, where $\mathbf{e}_o$ is the error of $\mathbf{w}_o$, and the substitution into (5) and (4) gives

$$\mathbf{v}_{n+1} = (\mathbf{I} - a\,\mathbf{G}_n)\,\mathbf{v}_n + \mathbf{b}_n, \qquad \mathbf{G}_n = \mathbf{F}_M^{-1}\,\mathbf{U}_n^{\mathsf T}\mathbf{X}^{r}_n / M_e, \tag{9}$$

where $\mathbf{U}_n$ and $\mathbf{X}^r_n$ are the $M_e \times M$ regression matrices of the whitened filtered reference and of the filtered reference in the window, $\mathbf{F}_M$ is the $M \times M$ convolution matrix of the spectral factor, and $\mathbf{b}_n$ is the contribution of the residual error of $\mathbf{w}_o$.

Equation (9) contains only the current coefficient error. Because the matrices $\mathbf{G}_n$ of consecutive blocks share samples, we have no closed-form stability bound on $a$, and we examined (9) numerically over 64 blocks with 20 runs for each input, starting from a coefficient error with equal entries and with $\mathbf{b}_n = \mathbf{0}$. These results are an empirical design guide for a short trial, not a stability proof. For white Gaussian input, the coefficient error decayed in all runs for $a \le 0.5$, grew in one run at $a = 0.55$, and grew in 18 runs at $a = 0.625$. The decay was fastest for $a$ between 0.375 and 0.455. The result was the same within one run when the filtered reference was colored by a first-order autoregressive spectrum with a pole at 0.9 or 0.98. For symmetric α-stable input with $\alpha = 1.45$ and $\alpha = 1.20$, the step size was also divided by the running average of the frame power and by $\max(g_n, g_{\min})$. No run grew for $a \le 0.375$, and one run grew at $a = 0.455$ for each $\alpha$, where the mean error still decayed by more than 50 dB. Based on these results, we set $\sigma_{\mathrm E} = 1$, which gives $a = 0.25$ for $g_n = 1$, and $g_{\min} = 0.55$ in the early phase, which limits $a/\max(g_n, g_{\min})$ to 0.455.

### 4.5 Late phase

In the late phase, the measured error frame is used, and the step size is small. We set $\sigma_{\mathrm L} = 0.0566$ and $g_{\min} = 0.25$. In preliminary simulations, the noise reduction over the whole run was the same for $\sigma_{\mathrm L}$ between 0.02 and 0.09.

### 4.6 Trial with the background filter

**Start of a trial.** For each block, the algorithm computes the ratio

$$z_n = \frac{\sum_{k \in \mathcal{W}_M(n)} e^2(k)}{\sum_{k \in \mathcal{W}_{2M}(n)} r^2(k)}, \tag{10}$$

where $\mathcal{W}_L(n)$ denotes the latest $L$ samples at block $n$. The division by the power of $r(k)$ removes the common scale of the input from the ratio. Let $\bar{z}_n$ be the exponential average of $z_n$. A trial starts at block $n_0$ when $z_n > \gamma\,\bar{z}_{n-1}$ holds for two consecutive blocks. After a trial starts, the next trial can start only after $n_{\mathrm G}$ blocks, and $\bar{z}_n$ is held constant during this interval. The average $\bar{z}_n$ is initialized with the first value of $z_n$. A block in which $z_n$ is not finite, as for a silent input, neither updates $\bar{z}_n$ nor counts toward the start of a trial.

**Adaptation.** At block $n_0$, the background filter is set to $\mathbf{w}_{\mathrm b} = \mathbf{w}_{{\mathrm f}, n_0}$. From block $n_0 + 1$, it is updated by (4) in the early phase, with the error frame recomputed by (8) with $\mathbf{w}_{\mathrm b}$. The foreground filter continues its update in its current phase, and the control signal is generated by $\mathbf{w}_{\mathrm f}$ only.

**Decision.** For each block $n \ge n_0 + 2$, the algorithm recomputes the errors of both filters by (8) for the latest $B$ samples and forms the ratio

$$\rho_n = \frac{\sum_{k \in \mathcal{W}_B(n)} \tilde{e}_{\mathrm b}^2(k)}{\sum_{k \in \mathcal{W}_B(n)} \tilde{e}_{\mathrm f}^2(k)}. \tag{11}$$

Both errors in (11) use the same $\hat{d}(k)$, so that a common scale of the signals cancels in the ratio. The trial ends by one of the following rules.

1. Discard: if $\rho_n > \rho_{\mathrm{abort}}$ or $\rho_n$ is not finite, $\mathbf{w}_{\mathrm b}$ is discarded.
2. Early transfer: if $n \ge n_0 + n_1$ and $\rho_n \le \rho_1$, the coefficients are transferred, $\mathbf{w}_{\mathrm f} \leftarrow \mathbf{w}_{\mathrm b}$.
3. Final decision: at $n = n_0 + n_2$, the coefficients are transferred if $\rho_n \le \rho_2$, and $\mathbf{w}_{\mathrm b}$ is discarded otherwise.

After a transfer, the foreground filter continues in the early phase for the remaining $n_{\mathrm E} - (n - n_0)$ blocks. We use $\rho_{\mathrm{abort}} = 4$, $(n_1, \rho_1) = (4, 0.25)$, and $(n_2, \rho_2) = (16, 0.5)$.

**Proposition 2.** Consider two runs of the algorithm with the same initial state and the same input signals. In the first run, the trials are disabled. In the second run, every trial up to block $n$ is discarded. Then the control signal $y(k)$ and the foreground coefficients of the two runs are identical up to block $n$.

*Proof.* The control signal and the update of $\mathbf{w}_{\mathrm f}$ depend on $x(k)$, $e(k)$, and $\mathbf{w}_{\mathrm f}$. The background filter and the quantities $z_n$, $\bar{z}_n$, and $\rho_n$ are computed from these signals and affect $\mathbf{w}_{\mathrm f}$ only through a transfer. Without a transfer, the recursions of $\mathbf{w}_{\mathrm f}$ and $y(k)$ in the two runs are the same, and their initial states are the same. $\blacksquare$

Proposition 2 is a structural property and not a convergence result. It requires no assumption on the signals or on the accuracy of $\hat{s}(k)$, and it says nothing about the stability after a transfer. A transfer without a path change is outside its scope, and Section 7 reports its frequency.

The decision uses the reconstructed errors, and the following result relates it to the errors that the two filters would actually produce. For fixed coefficients $\mathbf{w}$, let $e_{\mathbf{w}}(k) = d(k) + v(k) + s(k) * [\mathbf{w}^{\mathsf T}\mathbf{x}(k)]$ be this error, and let $\rho^{\star}_n$ be the ratio (11) computed with $e_{\mathbf{w}_{\mathrm b}}$ and $e_{\mathbf{w}_{\mathrm f}}$. Let $u(k) = (\mathbf{w}_{\mathrm b} - \mathbf{w}_{\mathrm f})^{\mathsf T}\mathbf{x}(k)$ be the difference of the two filter outputs, and let

$$\kappa_n = \frac{\| (s - \hat{s}) * u \|_n}{\| \hat{s} * u \|_n}, \tag{12}$$

where $\|\cdot\|_n$ is the norm over $\mathcal{W}_B(n)$.

**Proposition 3.** Assume that the foreground coefficients are constant over the samples that contribute to $\mathcal{W}_B(n)$. Then

$$\left| \sqrt{\rho^{\star}_n} - \sqrt{\rho_n} \right| \le \kappa_n \left( 1 + \sqrt{\rho_n} \right).$$

*Proof.* By (7) and (8), $\tilde{e}_{\mathbf{w}} - e_{\mathbf{w}} = (s - \hat{s}) * (y - \mathbf{w}^{\mathsf T}\mathbf{x})$. Under the assumption, $y = \mathbf{w}_{\mathrm f}^{\mathsf T}\mathbf{x}$, so that $\tilde{e}_{\mathrm f} = e_{\mathbf{w}_{\mathrm f}}$ and $e_{\mathbf{w}_{\mathrm b}} - \tilde{e}_{\mathrm b} = (s - \hat{s}) * u$. Because $\tilde{e}_{\mathrm b} - \tilde{e}_{\mathrm f} = \hat{s} * u$, we have $\|(s - \hat{s}) * u\|_n = \kappa_n \|\tilde{e}_{\mathrm b} - \tilde{e}_{\mathrm f}\|_n \le \kappa_n (\|\tilde{e}_{\mathrm b}\|_n + \|\tilde{e}_{\mathrm f}\|_n)$. The triangle inequality for $\|e_{\mathbf{w}_{\mathrm b}}\|_n$ and the division by $\|\tilde{e}_{\mathrm f}\|_n$ give the result. $\blacksquare$

A transfer with $\rho_n \le \rho_i$ therefore replaces the foreground coefficients by coefficients with a smaller actual error on the same data ($\rho^{\star}_n < 1$) whenever $\kappa_n < (1 - \sqrt{\rho_i})/(1 + \sqrt{\rho_i})$. This limit is 0.33 (−9.5 dB) for the early transfer and 0.17 (−15.3 dB) for the final decision. The quantity $\kappa_n$ is the model error weighted by the spectrum of $u(k)$. It equals the relative error of $\hat{s}$ when $u(k)$ is white, and it is larger when $u(k)$ is concentrated at frequencies where the model is inaccurate. The assumption holds approximately in the late phase, where the foreground step size is small. Proposition 3 concerns the data of the decision and does not guarantee the error after the transfer.

[4 章のメモ]
- 命題 1 = THEORY 定理 A、命題 2 = 定理 C（系 C2）、命題 3 = 命題 H。補題 F は本文の導出（式 (9)）にした。付録は置かない。
- 後半のステップは設計値として書いた（設計式は削除）。「0.02〜0.09 で差なし」は E3（δ = 0.01〜0.05 に相当）。
- 命題 1 の数値（固有値 0.13〜1.0）は THEORY §2。式 (9) の数値は theory/v4/mp.py と tail.py（白色ガウス、3 系列、64 ブロック）。
- 検出の指数平均の係数 0.05、γ = 4、n_G = 64、n_E = 32 は Table 1。
- 4.7 節：118,016／241,920／196,608 は RESULTS_fast.md と一致。Table 3 の演算量の欄は、この基準に直す必要がある。

## 5. Evaluation framework

### 5.1 Acoustic paths and signals

We simulated closed-loop control at 16 kHz with room impulse responses from the MeshRIR dataset [MeshRIR], measured in one room. We used eight combinations of a noise source, a secondary source, and an error microphone. The primary path had 4096 coefficients, the secondary path had 2048 coefficients, and the model $\hat{s}(k)$ was the first 512 coefficients of the secondary path. The truncated part contained −23.4 dB to −24.7 dB of the energy of the secondary path, so that the model had an error of this size in all evaluations. The reference signal was the source signal itself, so that the reference path and the acoustic feedback were excluded.

Four noise types were synthetic: a Gaussian signal, a Gaussian signal with isolated impulses, and symmetric α-stable signals with $\alpha = 1.45$ and $\alpha = 1.20$. Six were recordings: machine gun, factory, destroyer engine, and car interior noise from NOISEX-92 [NOISEX], valve noise from MIMII [MIMII], and jackhammer noise from SONYC-UST [SONYC]. All signals were high-pass filtered at 100 Hz, and each run lasted 30 s. For each noise type and condition, 20 seeds determined the paths, the random sequences, and the starting positions in the recordings.

### 5.2 Conditions

In the two source-switch conditions, the primary path changed at 15 s to the response from another loudspeaker to the same microphone. The new loudspeaker was the candidate closest to the original noise source (close source) or the farthest from it (far source). In the no-change condition, the paths were fixed. In the two disturbance conditions, the paths were fixed, and white Gaussian noise was added to the error signal for 0.5 s at 10 s and at 20 s, at 0 dB or −10 dB relative to the primary noise.

### 5.3 Algorithms

Table 1 lists the settings of the proposed algorithm, which were the same in all runs. We also evaluated it with the trials disabled, which the tables list as "Without trials."

The ten comparison algorithms were FxNLMS, the threshold-based algorithm of Akhtar and Mitsuhashi (Th-FxNLMS) [Akhtar 2008, Akhtar 2009], FxlogNLMS [Wu 2011], MFxLMM [Sun et al. 2015], FxNFLOC [Feng et al. 2024], EVSS-FxeLMS [Cheng 2023], the block frequency-domain FxLMS algorithm (FreqFxLMS) [Das 2007], our previous algorithm (CohVSS-FxNLMS) [APSIPA 2026], the convex combination algorithm (C-FXLMS) [Ferrer 2013], and a normalized version of the modified FxLMS algorithm (MFxNLMS) [Bjarnason 1992]. MFxNLMS uses the same reconstruction of the error as the early phase of the proposed algorithm, with a single filter and a fixed step size. Th-FxNLMS clipped the reference and error signals used in the update at percentiles of their last 0.5 s, updated in every block. The window length and the update interval are our choice, because we could not confirm those of the original algorithm. The clipping acted on 2% to 4% of the samples.

All algorithms used 512 coefficients. For each algorithm, we selected the step size from seven values between 1/32 and 2 times the published value, together with the percentiles of Th-FxNLMS and the step-size ratio and component filters of C-FXLMS. The selection used five seeds separate from the evaluation, the no-change and far-source conditions, and nine noise types. The criterion was the smallest number of diverged runs, followed by the largest noise reduction over 30 s. One setting per algorithm was used for all noise types and conditions.

### 5.4 Metrics

As the noise reduction, we use the averaged noise reduction that is common for impulsive noise [Akhtar 2009], $20\log_{10}[A_e(k)/A_d(k)]$, where $A_e(k)$ and $A_d(k)$ are exponential averages of $|e(k)|$ and $|d(k)|$ with a forgetting factor of 0.999. A negative value indicates a reduction. We averaged it over the whole run, over the last 10 s, and over the 2 s from the onset of each disturbance or change, and then over the noise types for each seed. A run is counted as diverged when the error power of a block is not finite or exceeds $10^4$ times the power of the primary noise over the run.

The recovery time is the time from the change until the short-time error power, averaged over 16 blocks (0.128 s), stays within 3 dB of a reference level for 1 s. The reference level is the error power over the last 5 s of a separate run in which the same algorithm operates with the new primary path from the start. Runs without recovery are censored, and medians are computed by the Kaplan–Meier method. For the proposed algorithm, 28 of the 200 runs in each source-switch condition were censored: all 20 runs with the isolated impulses and eight runs with valve noise. Because the reference level depends on the algorithm, we also froze the coefficients 0.13 s and 0.51 s after the change and computed the ratio of the error power to the primary-noise power for these fixed coefficients over the signal after the change, excluding its first 256 ms. The recovery time measures the return of the error level, and the frozen coefficients measure the quality of the control filter.

Differences are paired over runs with the same noise and paths, and they are given as the value of the proposed algorithm minus that of the other algorithm. The 95% confidence interval was obtained by resampling the 20 seeds 2000 times, and we call one algorithm better only when the interval excludes zero. The 20 seeds share eight sets of paths. When the eight sets were resampled instead of the seeds, the limits of the intervals in Table 2 changed by at most 0.14 dB and 0.05 s.

## 6. Reconvergence after a primary path change

### 6.1 Overall results

No run of the proposed algorithm diverged in the 1000 runs. Compared with CohVSS-FxNLMS, the fastest of the ten comparison algorithms, the proposed algorithm shortened the median recovery time by 63% with the close source and by 68% with the far source (Table 2). The convex combination did not shorten the recovery, because the mixing parameter of C-FXLMS stayed above 0.96 and its fast filter set the recovery time.

A larger step size shortened the recovery of the comparison algorithms only together with divergence (Fig. 5). Twice the common step size shortened the median recovery time of CohVSS-FxNLMS to 0.54 s and 0.70 s, but 10% of the runs diverged. At four times, the runs without divergence recovered in 0.30 s and 0.32 s, close to the proposed algorithm, and 31% of the runs diverged. FxNLMS and Th-FxNLMS showed the same trade-off at longer recovery times. No step size of these algorithms gave both the recovery time of the proposed algorithm and no divergence.

The trials produced the reduction. The algorithm without the trials recovered within 0.13 s of CohVSS-FxNLMS. MFxNLMS, which uses the reconstructed error without a trial, needed 1.97 s and 2.43 s. In the proposed algorithm, the coefficients were transferred 0.07 s to 0.08 s after the change, and the early phase of the foreground filter ended about 0.27 s after the change, shortly before the median recovery time.

The frozen coefficients show the same result on a common scale (Fig. 4). With the far source, the coefficients of the proposed algorithm at 0.13 s reduced the noise about as much as those of CohVSS-FxNLMS at 0.51 s. At 0.51 s, the residual power of the proposed algorithm was smaller by a factor of 1.9 with the close source and 2.4 with the far source. The actual error during the adaptation gave the same order. Over the first 0.5 s, 1 s, and 2 s after the change, the ratio of the error power to the primary-noise power was smaller for the proposed algorithm than for CohVSS-FxNLMS by 1.6 dB to 1.9 dB with the close source and by 2.2 dB to 2.5 dB with the far source, and smaller than for C-FXLMS and MFxNLMS by 2.5 dB to 4.1 dB. All intervals excluded zero. Fig. 3 shows the time course for the α-stable signal with $\alpha = 1.45$. The return to within 3 dB of the final level took 65% to 67% less time than for CohVSS-FxNLMS, and the levels before the switch agreed within 0.3 dB.

### 6.2 Results for each noise type

Fig. 6 shows the paired difference of the noise reduction with frozen coefficients for each noise type. The proposed algorithm was better in 31 of the 40 comparisons, CohVSS-FxNLMS was better in six, and the interval included zero in three. The six exceptions were the four comparisons for machine gun noise and the two for car interior noise with the close source. These two noise types accounted for 37 of the 60 runs without a transfer after the change with the close source and for 13 of the 20 such runs with the far source. In such a run, no transfer occurred before the change either, and the proposed algorithm produced the same control signal as the algorithm without the trials (Proposition 2).

For machine gun noise, the recovery time does not measure the reconvergence. The power of 34% of the blocks of this recording was more than 20 dB below the median, and the short-time error power varied over about 55 dB, compared with 11 dB for the α-stable signal with $\alpha = 1.45$. Every algorithm first met the 3 dB criterion 0.12 s after the change and then failed to hold it for 1 s, and all median recovery times were between 0.9 s and 1.1 s. The frozen coefficients separate the algorithms. With the far source, the coefficients of CohVSS-FxNLMS at 0.51 s reduced the noise by 2.6 dB more than those of the proposed algorithm, for which a transfer occurred in nine of the 20 runs. The variant that runs a trial in every block without the detector (Section 7.3) transferred the coefficients in all 20 runs, and its difference from CohVSS-FxNLMS was 0.3 dB with an interval that included zero. With the far source, the missed transfers therefore account for most of the gap. With the close source, a transfer occurred in three of the 20 runs, and CohVSS-FxNLMS was better by 1.1 dB. The variant without the detector remained worse by 0.8 dB, so that this gap originates mainly in the update of the foreground filter. This noise lies outside the scope stated in Section 1, and the result marks mainly the limit of the present detector rather than of the trial itself.

For car interior noise, the switch to the close source raised the error power by only 1.1 dB, below the 6 dB needed to start a trial, and the variant without the detector gave no improvement either. For the actual error over the 2 s after the change, the two measures disagreed for machine gun noise. With the close source, the averaged noise reduction was larger for the proposed algorithm by 0.5 dB ([−1.0, −0.1] dB), whereas the ratio of the error power to the primary-noise power was smaller for CohVSS-FxNLMS by 0.9 dB, and by 1.9 dB with the far source. The power ratio is dominated by the bursts, and by this measure CohVSS-FxNLMS was better for this noise. For car interior noise with the close source, the power ratio over 2 s differed by 0.04 dB with an interval that included zero.

For valve and jackhammer noise, the two algorithms had equal recovery times, although the frozen coefficients of the proposed algorithm reduced the noise by 1.8 dB to 8.2 dB more. The power of these recordings varies over several seconds, and this variation dominated the 3 dB criterion. For the isolated impulses, no algorithm met the criterion.

### 6.3 Comparison algorithms with a setting for each noise type

A setting selected for each noise type can give a comparison algorithm a larger step size where the noise allows it. We repeated the far-source and no-change conditions with such settings for the comparison algorithms, selected with the same seeds and criterion. The α-stable signal with $\alpha = 1.20$, which was not used in the selection, kept the common setting, and so did the proposed algorithm.

These settings shortened the median recovery time of CohVSS-FxNLMS from 1.26 s to 0.91 s and that of C-FXLMS from 2.78 s to 1.81 s. The median recovery time of the proposed algorithm remained 56% and 78% shorter. Its noise reduction over the whole run exceeded that of every comparison algorithm by 0.04 dB to 7.6 dB, and all intervals excluded zero. For individual noise types without a change, C-FXLMS was better for the Gaussian signal (0.07 dB) and destroyer engine noise (0.41 dB), and Th-FxNLMS was better for destroyer engine noise (0.41 dB). FxNFLOC diverged in three of the 400 runs.

For the signal of Fig. 3, the setting of C-FXLMS had an eight times larger step size, which had diverged in 60 of 90 runs in the selection of the common setting. With this step size, the noise reduction of C-FXLMS 0.3 s after the switch was within 0.1 dB of that of the proposed algorithm. The proposed algorithm still returned to within 1 dB of its final level 2.5 to 2.8 times earlier.

[5–6 章のメモ]
- Fig. 3 は paper/anr_curves.py --common（α = 1.45、seed 100–119 の中央値）。3 dB 以内に戻る時刻は、27〜30 秒の平均を最終水準として曲線から求めた。
- 6.3 節の 8 手法の数字は RESULTS_final2.md の補助表（クラスごとの最良設定、stat と RS-far、a120 は共通設定）。回復 114 ブロック = 0.91 秒、他は 279〜648 ブロック。発散は FxNFLOC の 2（stat）＋1（RS-far）。
- 6.3 節の C-FXLMS（入力別設定）の数字は out/anr_curves_class.npz から：0.3／0.5／1.0 秒後の ANR、最終水準（27〜30 秒の平均）から 1 dB 以内に戻る時刻。
- C-FXLMS の入力別設定の評価は RESULTS_cc_class.md（5b62d2d）：回復 226 ブロック = 1.81 秒、L1 whole の差 −0.62（stat）、−1.11（far）、直後 2 秒 −2.51 [−2.73, −2.25]。入力別の判定は 提案法 34／C-FXLMS 2／0 を含む 3／欠測 1。α = 1.45 の設定（ステップ 0.3、比 0.25）は、この環境で選んだもの（out/anr_cc_class.json）と一致。
- 「60/90 で発散」は TUNE_cc.json の grid（nx1|r0.25：div 60。調整は 9 入力 × 2 条件 × 5 seed = 90 実行）。
- MeshRIR、NOISEX-92、MIMII、SONYC-UST の書誌は未記入。「32 音源・441 点」は S32-M441 の名称から。要照合。
- 5.2 節の外乱：白色ガウス雑音。レベルは最初の 10 秒の一次騒音のパワーに対する値（prop2.error_noise で確認）。
- 5.3 節：調整に使った入力は 9 種（a120 を除く）、条件は変化なしと far。本文では省いた。必要なら 1 文足す。
- 6.1 節「他の 8 手法の中央値」：near は 288〜604 ブロック（2.3〜4.8 秒）、far は 185〜662 ブロック（1.5〜5.3 秒）。FxlogNLMS は打ち切りが 80〜90 件あり、far の 185 ブロックはこの手法の値。
- 6.1 節の移行の時刻：確定した試行（near 140、far 180）について、変化から試行の開始まで中央値 1.5／1 ブロック、移行まで 10／9 ブロック（四分位 8〜15）。
- 6.2 節の見逃しの内訳（near）：volvo 20、machinegun 17、valve 8、jackhammer 6、II300 5、a120 4。far：machinegun 11、II300 4、a120 2、volvo 2、valve 1。
- 6.2 節の machine gun と car interior の説明は数値で裏づけた：変化前の消音量（machine gun −5.0 dB、他は −8〜−13 dB）、試行が始まった実行（5/20、12/20）、car interior（near）の誤差の増加 1.1 dB（再始動なしの L1_pre −5.3 → L1_ev −4.2）。変化後 2 秒と 30 秒全体の対応差は今回計算（machine gun：−0.52 [−1.00, −0.07]、−0.28 [−0.88, +0.27]、全体 −0.79、−0.98。car interior near：−0.06 [−0.16, +0.04]、全体 −0.26）。「バースト間に騒音がほぼない」は波形の性質で、数値（低パワー区間の割合）は未計算。
- 6.2 節の回復時間（クラス別）：提案法 22〜44 ブロック、CohVSS-FxNLMS 93〜204 ブロック（G、a145、a120、factory1、destroyerengine）。valve は両者 538、jackhammer は 148。
- 変化なしの条件では 2494 回の試行が始まり、移行は 0 回（7 章で使う）。

## 7. Fixed primary path, design choices, and robustness

This section examines the second and third requirements of Section 3.4, the contribution of each element, the computational cost, the sensitivity to the parameters, and the effect of an error in the secondary-path model.

### 7.1 No change in the primary path

In the no-change condition, all 2494 trials were discarded, about one every 2.4 s, and the control signal was identical to that obtained without the trials in every run (Table 3). The mean noise reduction differed from that of CohVSS-FxNLMS by less than 0.1 dB, although the interval did not exclude a difference slightly above 0.1 dB, and it exceeded that of C-FXLMS by about 1.0 dB. For destroyer engine noise, C-FXLMS was better by 0.4 dB.

### 7.2 Disturbance at the error microphone

With a disturbance, fewer than 0.5% of the trials ended with a transfer, and only the runs with such a transfer differed from those without the trials. Over the 2 s after each disturbance, the difference from the algorithm without the trials was at most 0.01 dB, and both intervals included zero.

After the 0 dB disturbance, CohVSS-FxNLMS and C-FXLMS reduced the noise by 0.62 dB and 0.27 dB more than the proposed algorithm. The algorithm without the trials showed the same gap, which therefore comes from the late-phase update of the foreground filter. C-FXLMS switched to its slow filter during the disturbance (mixing parameter 0.04). With the −10 dB disturbance, CohVSS-FxNLMS was better by 0.20 dB, and C-FXLMS was worse by 0.27 dB.

### 7.3 Contribution of each element

Table 4 lists three variants of the trial. Without the division by $g_n$, seven to nine of the 200 runs diverged in each condition. When the coefficients with the larger step size controlled the noise from the start of a trial, the recovery time was the same within 0.02 s, but the control signal changed in 71 of the 200 runs without a change. The background filter therefore keeps the output unchanged at no cost in speed. When a trial ran in every block without the detector, the recovery was 10% to 17% faster, and the noise reduction after the 0 dB disturbance was 0.50 dB larger ([0.37, 0.64] dB). This variant made 96 transfers without a change and had 1.9 times the average cost, with the same peak cost. The detector trades this speed-up for an unchanged output and a lower average cost.

We also removed the other two elements of the update, with parameters either equal to those of the proposed algorithm or selected in the same way as for the comparison algorithms. No run of these variants diverged. Without the reconstruction of the error frame, the median recovery time was 0.55 s and 0.72 s with the selected step size and 1.03 s and 1.12 s with that of the proposed algorithm, and the coefficients frozen at 0.51 s reduced the noise by 1.9 dB to 5.4 dB less. The reconstruction is therefore the element that allows the large step size. Without the spectral factor and with the same step sizes, the recovery time and the noise reduction of the coefficients frozen at 0.51 s agreed with those of the proposed algorithm within 0.02 s and 0.1 dB. The noise reduction over the whole run was smaller by 0.16 dB to 0.19 dB, and by 0.8 dB for destroyer engine noise without a change. The spectral factor thus improves the converged level rather than the speed of the reconvergence.

Finally, we applied the trial to our previous algorithm. The foreground filter used the update of CohVSS-FxNLMS, and the background filter used the same update with a step size eight times larger, selected from 2 to 16 times. The median recovery time was 0.74 s and 0.89 s, and the coefficients frozen at 0.51 s reduced the noise by 1.4 dB and 2.1 dB less than those of the proposed algorithm. The trial alone therefore shortened the recovery of CohVSS-FxNLMS by 25% to 29%, and the update of Section 4 shortened it to less than half of this value. This variant recovered better from the 0 dB disturbance, by 0.62 dB, as CohVSS-FxNLMS did.

### 7.4 Computational cost

Per block of 128 samples (Table 5), the late phase requires 0.60 times the real multiplications of the time-domain FxNLMS algorithm, and a block with a trial requires 1.23 times. A trial was active in 7% of the blocks, and the average cost was 0.65 times that of the FxNLMS algorithm. A real-time implementation must provide for the peak during a trial. By the same count, CohVSS-FxNLMS requires about 0.63 times the multiplications of the FxNLMS algorithm, close to the late phase of the proposed algorithm. The counts exclude the logarithm and the complex exponential that the spectral factor requires for each bin once per block, and the three power functions and one exponential per bin of CohVSS-FxNLMS.

### 7.5 Sensitivity to the parameters

We varied four parameters of the proposed algorithm one at a time. Table 6 shows the result for the threshold $\gamma$ in (10). A smaller $\gamma$ started a trial after more changes and also led to more transfers without a path change. For $\gamma$ between 2 and 4, the median recovery time stayed within 0.04 s of the default, and with $\gamma = 8$, it was about twice as long.

For the transfer thresholds $(\rho_1, \rho_2)$ of $(0.18, 0.35)$ and $(0.35, 0.7)$ and for the final decision at 8 or 32 blocks, the median recovery time stayed within 0.05 s of the default, and the noise reduction over the whole run stayed within 0.04 dB. The looser thresholds increased the transfers without a path change from 0 to 18 in the no-change condition and from 13 to 32 with the 0 dB disturbance. The step size $\sigma_{\mathrm E} = 0.5$ gave median recovery times of 0.42 s and 0.37 s without divergence, and $\sigma_{\mathrm E} = 1.5$ caused divergence in three of the 200 runs in each condition. This value corresponds to $a \le 0.68$, which is above the range in which the error decayed in Section 4.4.

### 7.6 Error in the secondary-path model

In addition to the truncation error of about −24 dB (Section 5.1), we added white Gaussian noise to the coefficients of $\hat{s}(k)$ at −40 dB, −30 dB, and −20 dB relative to their energy and repeated the no-change and source-switch conditions (Table 7). The actual secondary path was unchanged. With an error of −40 dB, no run diverged, and the median recovery times changed by at most 0.01 s. With −30 dB, the proposed algorithm diverged in nine of the 600 runs, and CohVSS-FxNLMS diverged in three. With −20 dB, the proposed algorithm and C-FXLMS diverged in 54 runs each, and CohVSS-FxNLMS diverged in 21. The proposed algorithm is therefore more sensitive to the model error than CohVSS-FxNLMS. Most of this sensitivity originates in the update of the foreground filter, because the algorithm without the trials diverged in six and 51 runs. At each of these two levels, the proposed algorithm diverged in three more runs, within 2 s after a transfer. These three runs were the three conditions of one noise sequence. With a model error, a transfer can thus cause divergence in rare cases.

In the runs without divergence, the advantage in the recovery time remained. With −20 dB, the median recovery time of the proposed algorithm was 0.44 s and 0.51 s, compared with 1.03 s and 1.33 s for CohVSS-FxNLMS. With −30 dB and without a change, the noise reduction of the proposed algorithm over the last 10 s was smaller than that of CohVSS-FxNLMS by 0.3 dB ([0.06, 0.78] dB). 

To check the decisions, we computed $\rho^{\star}_n$ and $\kappa_n$ of Proposition 3 with the true secondary path for all 364,970 decisions. The inequality held in all but six of them, and in these six the foreground coefficients had changed within the memory of the secondary path, which violates the assumption. Without an added model error, using $\rho^{\star}_n$ instead of $\rho_n$ changed the outcome of 0.03% to 0.18% of the decisions in each condition, and the two ratios had a correlation coefficient above 0.99. Of the 321 transfers after a path change, 310 had $\kappa_n < 0.33$, and one, for machine gun noise, was reversed ($\rho^{\star}_n \ge 1$) with $\kappa_n = 1.5$. Frozen over the following 2 s, the transferred coefficients gave a power ratio 6.0 dB to 6.2 dB lower in median than the coefficients they replaced, and none of them was worse. The 13 transfers in the 0 dB disturbance condition also improved the coefficients, by 0.76 dB in median, because the disturbance had degraded the foreground filter.

With an added error of −20 dB, the correlation fell to between 0.81 and 0.84, and 19 of the 23 transfers without a path change were reversed. At all levels, every reversed transfer had $\kappa_n > 0.9$, well above the limits of Proposition 3, whereas the median of $\kappa_n$ over the transfers in the source-switch conditions was between 0.09 and 0.24. All six runs in which only the proposed algorithm diverged followed a reversed transfer, with $\rho_n$ between 0.14 and 0.19 and $\rho^{\star}_n$ between 3.5 and 4.6. The condition of Proposition 3 is sufficient and conservative. With the added error of −20 dB, only 28% to 44% of the transfers in the source-switch conditions satisfied $\kappa_n < 0.17$, and 88% of these transfers were not reversed.

## 8. Conclusion

We proposed a frequency-domain filtered-x algorithm for impulsive noise that tests a larger step size on a background filter during a short trial and transfers the result to the foreground filter only when it reduces the reconstructed error. The trial isolates the large step size from the control signal, and a discarded trial leaves the control signal unchanged. In simulations with ten noise types, the trials shortened the median recovery time after a switch of the noise source by about two thirds compared with the fastest of ten comparison algorithms, without divergence. For the comparison algorithms whose step size we varied, no step size gave this recovery time without divergence. With a fixed primary path, the control signal was identical to that obtained without the trials.

The trial mechanism and the foreground filter have different limitations. The mechanism depends on the detector. For machine gun noise, which consists of intermittent bursts, a transfer occurred in fewer than half of the runs, and our previous algorithm reduced the noise more after the switch. No transfer occurred for a small change in the primary path. With an inaccurate secondary-path model, the decision can transfer worse coefficients. The foreground filter recovered less from a disturbance at the error microphone than two comparison algorithms, by 0.3 dB to 0.6 dB, and it diverged more often than our previous algorithm with an additional error of −30 dB or larger in the secondary-path model.

The results apply to noise that is present continuously and contains impulses, with a fixed secondary path and an accurate model of it. The evaluation used simulations with paths from one room and no acoustic feedback. Future work includes a start condition for intermittent noise, an update that is less sensitive to errors in the secondary-path model, the evaluation in other rooms, and a real-time implementation.

[7–8 章のメモ]
- 7.1 節：変化なしの 2494 回の試行の内訳は、打ち切り（ρ > 4）188、判定で破棄 2297、実行の終わりで未了 9。
- 7.2 節：「RS-far で 1 回の余分な移行」は Table 3 の 0/181/1。
- 7.3 節の変種：1/g なし（P5-early-noG）、すぐ鳴らす版（P4-cap）、常時試行（TP-always-cap）。数値は今回こちらで対応差を計算した（figs.py の paired と同じ関数）。P4-cap の「71 回」は Table 3 の unmatched transfers（変化なし）。
- **7.3 節の常時試行の結果は、査読で「検出器は要らないのでは」と問われうる。** 常時試行は見逃しがなく、外乱の後は 0.5 dB 良い。提案法（検出器あり）の利点は、平均の演算量（0.65 倍。常時試行は 1.23 倍）と、変化なしでの出力の一致（200/200。常時試行は 96 回の移行）。この 2 点を本文に書いた。常時試行の機関銃での回復は 122／117 ブロックで、提案法と同じ（検出が原因ではない可能性がある。6.2 節の説明と合わせて要検討）。
- 7.4 節：平均は 8 実行（30,000 ブロック）の値。条件の内訳は RESULTS_fast.md。
- 8 章の「15 回の移行」は EN0 の 13 と ENm10 の 2。

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
