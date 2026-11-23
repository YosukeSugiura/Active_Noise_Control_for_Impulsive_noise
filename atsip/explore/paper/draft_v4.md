# Draft v4 (abstract, Sections 1–8)

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

Adaptive algorithms for active control of impulsive noise prevent divergence by limiting the influence of large samples or by using a small step size. Because these algorithms treat every large error in the same way, the same settings also slow the reconvergence after a sudden change in the primary path. This paper proposes a frequency-domain filtered-x algorithm that adapts a background filter with a larger step size in a trial. Its coefficients are transferred to the foreground filter, which generates the control signal, only when they give a sufficiently smaller error on the same data. A discarded trial leaves the control signal identical to that obtained without the trial, given the same initial state and input. In simulations with measured room impulse responses and ten noise types, the proposed algorithm reduced the median recovery time after a switch of the noise source by 63% to 68% compared with the fastest of nine comparison algorithms. For machine gun noise, which consists of intermittent bursts, the comparison algorithm recovered faster. With a fixed primary path, the noise reduction of the two algorithms differed by less than 0.1 dB.

[191 語。63% = 1 − 0.36/0.98（close）、68% = 1 − 0.40/1.26（far）。要旨の例外は machine gun に絞った。car interior（close）の例外は 1 章と 6.2 節に記載。]

## 1. Introduction

Impulsive noise contains short bursts with amplitudes much larger than the background level, and it is often modeled by a symmetric α-stable distribution [Shao and Nikias 1993]. In feedforward active noise control (ANC), the filtered-x least-mean-square (FxLMS) algorithm updates the control filter in proportion to the product of the filtered reference signal and the error signal [Kuo and Morgan 1996]. An impulsive sample in either signal can therefore make the control filter diverge.

Previous studies prevented this divergence in three ways. The first is to limit the influence of large samples. Thresholds remove or clip large samples in the reference and error signals [Sun 2006, Akhtar 2009], and cost functions that grow slowly for large errors bound their contribution to the update [Wu 2011, Wu and Qiu 2013, Feng et al. 2024]. The second is to control the step size. Normalized algorithms divide the step size by the signal power, and our previous algorithm scaled the step size in each frequency bin by the coherence between the filtered reference and error signals within a fixed range [APSIPA 2026]. The third is to combine two filters with different step sizes. Ferrer et al. [Ferrer 2013] applied the convex combination of a fast and a slow filtered-x filter to ANC, and later studies used such combinations for impulsive noise [Cheng 2022, Yu 2020].

In all these algorithms, a large error is handled by the same update rule regardless of its cause. A sudden change in the primary path, such as a switch to another noise source, also increases the error, and the error remains large until the control filter adapts to the new path. The thresholds, cost functions, and step sizes selected for stability under impulses therefore also determine the speed of this reconvergence. The convex combination does not remove this limitation, because its fast filter also generates the control signal and must also remain stable under impulses. In our simulations, we selected the step sizes of nine comparison algorithms to avoid divergence. The fastest of them, our previous algorithm, then needed about 1 s in median to recover after a switch of the noise source.

This paper addresses this reconvergence in a single-channel feedforward ANC system. We assume that the noise is present continuously and contains impulses, and that the secondary path and its model stay the same. The research question is how to apply a larger step size after a change in the primary path, when the error alone cannot distinguish such a change from impulses or from a disturbance at the error microphone. The ratio of the error power to the filtered-reference power rises after a path change, but it also rises with such a disturbance. A rise in the ratio therefore indicates only that the path may have changed.

The proposed algorithm applies the larger step size first to a background filter, which does not contribute to the control signal. The foreground filter generates the control signal, and it receives the background coefficients only when they give a smaller error than its own coefficients on the same data. We call this adaptation of the background filter a trial. A discarded trial leaves the control signal identical to that obtained without the trial, given the same initial state and input. A transfer that is confirmed without a path change can alter the control signal, and we report how often it occurred. Ochiai et al. [Ochiai et al. 1977] used foreground and background filters in an echo canceler. In ANC, the error of the background filter cannot be measured, and the proposed algorithm reconstructs it with the model of the secondary path.

A trial is short, so that the background filter must converge within a few blocks under impulsive noise. Its update direction is based on the spectral factor of the filtered-reference power spectrum [Elliott and Rafaely 2000], and its step size is divided by the power of each filtered-reference frame relative to its running average. The error frame is recomputed with the current coefficients, as in the modified FxLMS algorithm [Bjarnason 1992]. The proposed algorithm shares the block frequency-domain filtered-x structure with our previous algorithm [APSIPA 2026]. The trial with the background filter and these three elements of the update are new in this paper, and the coherence-based step size is not used.

The contributions of this paper are as follows.

1. We propose a trial with a background filter for the reconvergence of ANC, and we show that a discarded trial leaves the control signal unchanged (Proposition 3).
2. We combine three elements in the update used for the trial, and we show that the update caused by an isolated impulse does not depend on the amplitude of the impulse (Proposition 2).
3. We compare the proposed algorithm with nine algorithms for ten noise types. The evaluation includes the cases in which no transfer occurred and the case in which comparison algorithms recovered better from a disturbance.

Section 2 reviews related work in more detail, and Section 3 formulates the problem. Section 4 describes the proposed algorithm. Section 5 describes the evaluation framework, Section 6 reports the reconvergence after a change in the primary path, and Section 7 reports the cost of the trial mechanism when the path stays the same. Section 8 concludes the paper.

## 2. Related work

### 2.1 Adaptive algorithms for impulsive noise

For impulsive noise with α < 2, the second-order moment does not exist [Shao and Nikias 1993], and previous studies limited the effect of large samples on the update. Sun et al. [Sun 2006] ignored the samples that exceeded a threshold. Akhtar and Mitsuhashi [Akhtar 2008, Akhtar 2009] replaced such samples with the threshold values. Wu et al. [Wu 2011] minimized the squared logarithm of the error. Thanigai et al. [Thanigai 2007] rejected impulses in the error signal with the M-estimator of Hampel. Sun et al. [Sun et al. 2015] applied another Hampel M-estimator to the reference signal. Wu and Qiu [Wu and Qiu 2013] proposed an algorithm based on the Fair function. Feng et al. [Feng et al. 2024] derived a normalized algorithm from the fractional lower-order covariance. Cheng et al. [Cheng 2023] applied the Gaussian error function to the error. Our previous algorithm [APSIPA 2026] scaled the step size in each frequency bin by the coherence between the filtered reference and error signals, within a fixed range. In all these algorithms, one update rule with parameters selected for stability handles every large error.

### 2.2 Algorithms with two adaptive filters

Two adaptive filters have been combined to obtain both fast convergence and a small steady-state error. Ferrer et al. [Ferrer 2013] applied the convex combination of two filtered-x filters with different step sizes to ANC. Cheng et al. [Cheng 2022] combined an FxLMS filter with a filter based on the maximum Versoria criterion. Yu et al. [Yu 2020] combined an FxLMS filter with a filter based on the minimum kernel risk-sensitive loss. In these schemes, both filters contribute to the control signal whenever their weights are nonzero.

Ochiai et al. [Ochiai et al. 1977] adapted a background echo model and synthesized the echo replica with a foreground model. The model with the smaller residual became the new foreground model, so that false adaptation during double talk did not affect the replica. Lindström et al. [Lindstrom 2007] and Iqbal and Grant [Iqbal 2007] later improved the conditions for this transfer. In ANC, only the error of the foreground filter is measured, because the control signal reaches the error microphone through the secondary path. The proposed algorithm therefore reconstructs the error of the background filter with the model of the secondary path.

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

where $\mu$ is the step size, $\hat{\mathbf{S}}_{rr,n}$ is the exponential average of $|\mathbf{R}_n|^2$, $\epsilon$ is a regularization constant, $\odot$ and the division are element-wise, ${}^{*}$ denotes complex conjugation, $\mathcal{F}^{-1}$ is the inverse FFT, and $\mathcal{P}_M$ keeps the first $M$ samples. The projection $\mathcal{P}_M$ constrains the filter to be causal with $M$ coefficients. The proposed algorithm replaces the direction and the step size in (2), as described in Section 4.

### 3.3 Impulsive noise and primary path change

The noise $x(k)$ is impulsive, and its impulses appear in $d(k)$ and in $e(k)$. The algorithm places no assumption on the distribution of $x(k)$. The primary path changes at an unknown sample $k_c$:

$$p(k) = \begin{cases} p_1(k), & k < k_c, \\ p_2(k), & k \ge k_c. \end{cases} \tag{3}$$

Such a change occurs when the dominant noise source switches to a source at another position, and the error increases until the control filter reconverges. The secondary path $s(k)$ and its model $\hat{s}(k)$ stay the same, because a change in the secondary path requires the model to be identified again.

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

where $\boldsymbol{\delta}_n$ is the update direction (Section 4.2), $g_n$ is the relative power of the filtered-reference frame (Section 4.3), $g_{\min}$ is its lower bound, and $\sigma_n$ is the step size. The update has two phases. In the early phase, $\sigma_n = \sigma_{\mathrm E}$ is large, and the error frame is recomputed with the current coefficients (Section 4.4). In the late phase, $\sigma_n = \sigma_{\mathrm L}$ is small, and the measured error frame is used (Section 4.5). The foreground filter is in the early phase for the first $n_{\mathrm E}$ blocks after start-up and after each transfer, and it is in the late phase at all other times. The background filter is always in the early phase. Section 4.6 describes when a trial starts and how it ends. Table 1 lists the parameter values.

### 4.2 Update direction

The direction in (2) normalizes each frequency bin by the power spectrum before the causality constraint. Elliott and Rafaely [Elliott and Rafaely 2000] showed that this order biases the solution, and they split the normalization into two spectral factors. We follow this construction. Let $\boldsymbol{\Lambda}_n = \hat{\mathbf{S}}_{rr,n} + \epsilon \bar{P}_n$, where $\bar{P}_n$ is the mean of $\hat{\mathbf{S}}_{rr,n}$ over the frequency bins, and let $\mathbf{F}_n$ be the minimum-phase factor with $|\mathbf{F}_n|^2 = \boldsymbol{\Lambda}_n$. We compute $\mathbf{F}_n$ from the real cepstrum of $\boldsymbol{\Lambda}_n$. The direction is

$$\boldsymbol{\delta}_n = \mathcal{P}_M\, \mathcal{F}^{-1}\!\left\{ \frac{1}{\mathbf{F}_n} \odot \mathcal{F}\, \mathcal{P}_{N/2}\, \mathcal{F}^{-1}\!\left\{ \frac{\mathbf{R}_n^{*} \odot \mathbf{E}_n}{\mathbf{F}_n^{*}} \right\} \right\}, \tag{5}$$

where $\mathcal{P}_{N/2}$ keeps the first $N/2$ samples and sets the others to zero. The division by $\mathbf{F}_n^{*}$ whitens the cross-correlation, $\mathcal{P}_{N/2}$ removes its noncausal part, and the division by $\mathbf{F}_n$ maps the result back to the coefficient domain.

**Proposition 1.** Assume fixed coefficients $\mathbf{w}$, a stationary $x(k)$, $\epsilon = 0$, and $N \to \infty$. Then

$$\mathrm{E}[\boldsymbol{\delta}_n] = \frac{M_e}{N}\left(\mathbf{w} - [\mathbf{w}_\infty]_M\right),$$

where $M_e$ is the length of the error window ($M_e = M$ in this paper), $\mathbf{w}_\infty$ is the causal Wiener filter of infinite length, and $[\cdot]_M$ keeps its first $M$ coefficients.

This result follows from the analysis of Elliott and Rafaely [Elliott and Rafaely 2000]. With fixed coefficients, the expectation of $\mathbf{R}_n^{*} \odot \mathbf{E}_n$ is proportional to $S_{rr}W + S_{rd}$, where $W$ is the frequency response of $\mathbf{w}$ and $S_{rd}$ is the cross spectrum of $r(k)$ and $d(k)$. The division by $F^{*}$ gives $FW + S_{rd}/F^{*}$, whose causal part is $F(W - W_\infty)$ by the Wiener–Hopf solution, and the division by $F$ gives $W - W_\infty$. The mean update therefore acts as a scalar on the coefficient error, and all modes converge at the same rate. For $N = 2M$ and $\epsilon = 10^{-2}$, the eigenvalues of the mean action normalized by $M_e/N$ lay between 0.13 and 1.0 in a numerical evaluation, mainly because of the regularization.

### 4.3 Step size normalized by the relative frame power

The factor $\mathbf{F}_n$ follows the exponential average $\hat{\mathbf{S}}_{rr,n}$, which changes slowly. When a frame contains an impulse, its power is much larger than the average, and $\boldsymbol{\delta}_n$ grows with the square of the impulse amplitude. We therefore divide the step size by the relative frame power

$$g_n = \frac{\frac{1}{N/2+1}\sum_{f} |R_n(f)|^2}{\bar{P}_n}. \tag{6}$$

**Proposition 2.** Assume that the coefficients, $\mathbf{F}_n$, and $\bar{P}_n$ are fixed, that $v(k) = 0$, and that $x(k) = c(k) * [\varsigma(k)\xi(k)]$, where $c(k)$ is a fixed filter, $\xi(k)$ is a unit-variance white sequence, and $\varsigma(k)$ is a deterministic scale. Then each element of $\boldsymbol{\delta}_n$ is a quadratic form of the samples $z(k) = \varsigma(k)\xi(k)$ in the frame. If only one sample $z(k_0)$ in the frame is nonzero, then $\boldsymbol{\delta}_n = z^2(k_0)\,\mathbf{k}(k_0)$ and $g_n = z^2(k_0)\,\kappa(k_0)$ with a vector $\mathbf{k}(k_0)$ and a scalar $\kappa(k_0)$ that are independent of $z(k_0)$. The update $\boldsymbol{\delta}_n / g_n$ is therefore independent of the amplitude of the impulse.

This result follows from (5) and (6), because $\mathbf{R}_n$ and $\mathbf{E}_n$ are linear in $z(k)$ with fixed coefficients and $v(k) = 0$. An isolated impulse therefore produces a bounded update without a threshold, and the algorithm keeps impulsive samples in the update. When a frame also contains background noise, $\boldsymbol{\delta}_n/g_n$ depends on the position of the impulse in the frame, and we rely on the simulations for this case. The lower bound $g_{\min}$ limits the step size for frames with low power.

### 4.4 Early phase with recomputed error frames

The error window of block $n$ contains $M_e$ samples, and these samples were produced by the coefficients of the last $M_e/B$ blocks. The measured error frame therefore responds to an update with a delay, and this delay limits the step size. In the early phase, we remove this dependence on past coefficients. As in the modified FxLMS algorithm [Bjarnason 1992], the primary noise is estimated as

$$\hat{d}(k) = e(k) - \hat{s}(k) * y(k), \tag{7}$$

and the error frame is recomputed with the coefficients $\mathbf{w}_n$ to be updated:

$$\tilde{e}_n(k) = \hat{d}(k) + \hat{s}(k) * \left[\mathbf{w}_n^{\mathsf T}\mathbf{x}(k)\right] \tag{8}$$

for the $M_e$ samples in the window. $\mathbf{E}_n$ in (5) is then the FFT of $\tilde{e}_n(k)$.

The recomputed frame changes the form of the update. Assume $\hat{s} = s$, ideal whitening ($\boldsymbol{\Lambda}_n = \mathbf{S}_{rr}$, $N \to \infty$), and $g_n = 1$. Let $\mathbf{v}_n = \mathbf{w}_n - \mathbf{w}_o$ be the coefficient error with respect to a reference filter $\mathbf{w}_o$, and let $a = \sigma_{\mathrm E}\,(2B/M)(M_e/N)$. Under these assumptions, (8) gives $\tilde{\mathbf{e}}_n = \mathbf{X}^r_n \mathbf{v}_n + \mathbf{e}_o$, where $\mathbf{e}_o$ is the error of $\mathbf{w}_o$, and the substitution into (5) and (4) gives

$$\mathbf{v}_{n+1} = (\mathbf{I} - a\,\mathbf{G}_n)\,\mathbf{v}_n + \mathbf{b}_n, \qquad \mathbf{G}_n = \mathbf{F}_M^{-1}\,\mathbf{U}_n^{\mathsf T}\mathbf{X}^{r}_n / M_e, \tag{9}$$

where $\mathbf{U}_n$ and $\mathbf{X}^r_n$ are the $M_e \times M$ regression matrices of the whitened filtered reference and of the filtered reference in the window, $\mathbf{F}_M$ is the $M \times M$ convolution matrix of the spectral factor, and $\mathbf{b}_n$ is the contribution of the residual error of $\mathbf{w}_o$.

Equation (9) contains only the current coefficient error. Because the matrices $\mathbf{G}_n$ of consecutive blocks share samples, we have no closed-form stability bound on $a$, and we examined (9) numerically over 64 blocks with five runs for each input. For white Gaussian input, the coefficient error decayed for $a \le 0.55$ and grew for $a \ge 0.625$, and the decay after 32 blocks was fastest for $a$ between 0.375 and 0.455. The boundary was the same when the filtered reference was colored by a first-order autoregressive spectrum with a pole at 0.9 or 0.98. For symmetric α-stable input with $\alpha = 1.45$ and $\alpha = 1.20$, the step size was also divided by the running average of the frame power and by $\max(g_n, g_{\min})$. The error then decayed in all runs for $a \le 0.5$ and grew in one run with $\alpha = 1.20$ at $a = 0.55$. Based on these results, we set $\sigma_{\mathrm E} = 1$ ($a = 0.25$ for $g_n = 1$) and $g_{\min} = 0.55$ in the early phase, so that $a/\max(g_n, g_{\min}) \le 0.455$.

### 4.5 Late phase

In the late phase, the measured error frame is used, and the step size is small. We set $\sigma_{\mathrm L} = 0.0566$ and $g_{\min} = 0.25$. In preliminary simulations, the noise reduction over the whole run was the same for $\sigma_{\mathrm L}$ between 0.02 and 0.09.

### 4.6 Trial with the background filter

**Start of a trial.** For each block, the algorithm computes the ratio

$$z_n = \frac{\sum_{k \in \mathcal{W}_M(n)} e^2(k)}{\sum_{k \in \mathcal{W}_{2M}(n)} r^2(k)}, \tag{10}$$

where $\mathcal{W}_L(n)$ denotes the latest $L$ samples at block $n$. The division by the power of $r(k)$ removes the common scale of the input from the ratio. Let $\bar{z}_n$ be the exponential average of $z_n$. A trial starts at block $n_0$ when $z_n > \gamma\,\bar{z}_{n-1}$ holds for two consecutive blocks. After a trial starts, the next trial can start only after $n_{\mathrm G}$ blocks, and $\bar{z}_n$ is held constant during this interval.

**Adaptation.** At block $n_0$, the background filter is set to $\mathbf{w}_{\mathrm b} = \mathbf{w}_{{\mathrm f}, n_0}$. From block $n_0 + 1$, it is updated by (4) in the early phase, with the error frame recomputed by (8) with $\mathbf{w}_{\mathrm b}$. The foreground filter continues its update in its current phase, and the control signal is generated by $\mathbf{w}_{\mathrm f}$ only.

**Decision.** For each block $n \ge n_0 + 2$, the algorithm recomputes the errors of both filters by (8) for the latest $B$ samples and forms the ratio

$$\rho_n = \frac{\sum_{k \in \mathcal{W}_B(n)} \tilde{e}_{\mathrm b}^2(k)}{\sum_{k \in \mathcal{W}_B(n)} \tilde{e}_{\mathrm f}^2(k)}. \tag{11}$$

Both errors in (11) use the same $\hat{d}(k)$, so that a common scale of the signals cancels in the ratio. The trial ends by one of the following rules.

1. Discard: if $\rho_n > \rho_{\mathrm{abort}}$ or $\rho_n$ is not finite, $\mathbf{w}_{\mathrm b}$ is discarded.
2. Early transfer: if $n \ge n_0 + n_1$ and $\rho_n \le \rho_1$, the coefficients are transferred, $\mathbf{w}_{\mathrm f} \leftarrow \mathbf{w}_{\mathrm b}$.
3. Final decision: at $n = n_0 + n_2$, the coefficients are transferred if $\rho_n \le \rho_2$, and $\mathbf{w}_{\mathrm b}$ is discarded otherwise.

After a transfer, the foreground filter continues in the early phase for the remaining $n_{\mathrm E} - (n - n_0)$ blocks. We use $\rho_{\mathrm{abort}} = 4$, $(n_1, \rho_1) = (4, 0.25)$, and $(n_2, \rho_2) = (16, 0.5)$.

**Proposition 3.** Consider two runs of the algorithm with the same initial state and the same input signals. In the first run, the trials are disabled. In the second run, every trial up to block $n$ is discarded. Then the control signal $y(k)$ and the foreground coefficients of the two runs are identical up to block $n$.

*Proof.* The control signal and the update of $\mathbf{w}_{\mathrm f}$ depend on $x(k)$, $e(k)$, and $\mathbf{w}_{\mathrm f}$. The background filter and the quantities $z_n$, $\bar{z}_n$, and $\rho_n$ are computed from these signals and affect $\mathbf{w}_{\mathrm f}$ only through a transfer. Without a transfer, the recursions of $\mathbf{w}_{\mathrm f}$ and $y(k)$ in the two runs are the same, and their initial states are the same. $\blacksquare$

Proposition 3 requires no assumption on the signals or on the accuracy of $\hat{s}(k)$. A transfer without a path change is outside its scope, and Section 7 reports its frequency.

[4 章のメモ]
- 命題 1 = THEORY 定理 A、命題 2 = 定理 C（系 C2）、命題 3 = 命題 H。補題 F は本文の導出（式 (9)）にした。付録は置かない。
- 後半のステップは設計値として書いた（設計式は削除）。「0.02〜0.09 で差なし」は E3（δ = 0.01〜0.05 に相当）。
- 命題 1 の数値（固有値 0.13〜1.0）は THEORY §2。式 (9) の数値は theory/v4/mp.py と tail.py（白色ガウス、3 系列、64 ブロック）。
- 検出の指数平均の係数 0.05、γ = 4、n_G = 64、n_E = 32 は Table 1。
- 4.7 節：118,016／241,920／196,608 は RESULTS_fast.md と一致。Table 3 の演算量の欄は、この基準に直す必要がある。

## 5. Evaluation framework

### 5.1 Acoustic paths and signals

We simulated closed-loop control at 16 kHz with room impulse responses from the MeshRIR dataset [MeshRIR], measured in one room from 32 loudspeaker positions to 441 microphone positions. We used eight combinations of a noise source, a secondary source, and an error microphone. The primary path had 4096 coefficients, the secondary path had 2048 coefficients, and the model $\hat{s}(k)$ was the first 512 coefficients of the secondary path. The truncated part contained −23.4 dB to −24.7 dB of the energy of the secondary path, so that the model had an error of this size in all evaluations. The reference signal was the source signal itself, so that the reference path and the acoustic feedback were excluded.

Four noise types were synthetic: a Gaussian signal, a Gaussian signal with isolated impulses, and symmetric α-stable signals with $\alpha = 1.45$ and $\alpha = 1.20$. Six were recordings: machine gun, factory, destroyer engine, and car interior noise from NOISEX-92 [NOISEX], valve noise from MIMII [MIMII], and jackhammer noise from SONYC-UST [SONYC]. All signals were high-pass filtered at 100 Hz, and each run lasted 30 s. For each noise type and condition, 20 seeds determined the paths, the random sequences, and the starting positions in the recordings.

### 5.2 Conditions

In the two source-switch conditions, the primary path changed at 15 s to the response from another loudspeaker to the same microphone. The new loudspeaker was the candidate closest to the original noise source (close source) or the farthest from it (far source). In the no-change condition, the paths were fixed. In the two disturbance conditions, the paths were fixed, and white Gaussian noise was added to the error signal for 0.5 s at 10 s and at 20 s, at 0 dB or −10 dB relative to the primary noise.

### 5.3 Algorithms

Table 1 lists the settings of the proposed algorithm, which were the same in all runs. We also evaluated it with the trials disabled, which the tables list as "Without trials."

The nine comparison algorithms were FxNLMS, the threshold-based algorithm of Akhtar and Mitsuhashi (Th-FxNLMS) [Akhtar 2008], FxlogNLMS [Wu 2011], MFxLMM [Sun et al. 2015], FxNFLOC [Feng et al. 2024], EVSS-FxeLMS [Cheng 2023], the block frequency-domain FxLMS algorithm (FreqFxLMS) [Das 2007], our previous algorithm (CohVSS-FxNLMS) [APSIPA 2026], and the convex combination algorithm (C-FXLMS) [Ferrer 2013]. All of them used 512 coefficients. For each algorithm, we selected the step size from seven values between 1/32 and 2 times the published value. We also selected the threshold of Th-FxNLMS and the step-size ratio and component filters of C-FXLMS. The selection used five seeds separate from the evaluation, the no-change and far-source conditions, and nine noise types. The criterion was the smallest number of diverged runs, followed by the largest noise reduction over 30 s. One setting per algorithm was used for all noise types and conditions.

### 5.4 Metrics

The noise reduction is the ratio of the error power to the primary-noise power in decibels, so that a negative value indicates a reduction. We computed it over the whole run, over the last 10 s, and over the 2 s after each disturbance, and averaged it over the noise types for each seed.

The recovery time is the time from the change until the short-time error power stays within 3 dB of the level that the same algorithm reaches after the change, for 1 s. Runs without recovery are censored, and medians are computed by the Kaplan–Meier method. Because this reference level depends on the algorithm, we also froze the coefficients 0.13 s and 0.51 s after the change and computed the noise reduction of these fixed coefficients over the signal after the change, excluding its first 256 ms.

Differences are paired over runs with the same noise and paths, and they are given as the value of the proposed algorithm minus that of the other algorithm. The 95% confidence interval was obtained by resampling the 20 seeds 2000 times. We call one algorithm better only when the interval excludes zero.

## 6. Reconvergence after a primary path change

### 6.1 Overall results

No run of the proposed algorithm diverged in the 1000 runs. Compared with CohVSS-FxNLMS, the fastest of the nine comparison algorithms, the proposed algorithm shortened the median recovery time by 63% with the close source and by 68% with the far source (Table 2). The convex combination did not shorten the recovery. The mixing parameter of C-FXLMS stayed above 0.96, so that its fast filter, with a step size selected to avoid divergence, set the recovery time.

The trials produced the reduction, because the algorithm without the trials recovered within 0.13 s of CohVSS-FxNLMS. The coefficients were transferred 0.07 s to 0.08 s after the change, and the early phase of the foreground filter ended about 0.27 s after the change, shortly before the median recovery time.

The frozen coefficients show the same result on a common scale (Fig. 4). With the far source, the coefficients of the proposed algorithm at 0.13 s reduced the noise about as much as those of CohVSS-FxNLMS at 0.51 s. At 0.51 s, the residual power of the proposed algorithm was smaller by a factor of 1.9 with the close source and 2.4 with the far source. Fig. 3 shows the time course for the α-stable signal with $\alpha = 1.45$. The return to within 3 dB of the final level was 65% to 67% faster than for CohVSS-FxNLMS, and the levels before the switch agreed within 0.3 dB.

### 6.2 Results for each noise type

Fig. 5 shows the difference for each noise type. The proposed algorithm was better in 31 of the 40 comparisons, CohVSS-FxNLMS was better in six, and the interval included zero in three. The six exceptions were machine gun noise and car interior noise with the close source. These two noise types accounted for 37 of the 60 runs without a transfer after the change with the close source and for 13 of the 20 such runs with the far source. In such a run, either no trial started or every trial was discarded, and the proposed algorithm operates as the algorithm without the trials (Proposition 3).

For machine gun noise, the recovery time does not measure the reconvergence. In this recording, the power of 34% of the blocks was more than 20 dB below the median, and the short-time error power varied over about 55 dB, compared with 11 dB for the α-stable signal with $\alpha = 1.45$. Every algorithm first met the 3 dB criterion 0.12 s after the change and then failed to hold it for 1 s, and the median recovery times of all algorithms were between 0.9 s and 1.1 s. The noise reduction with frozen coefficients separates the algorithms. With the far source, the coefficients of CohVSS-FxNLMS at 0.51 s reduced the noise by 2.6 dB more than those of the proposed algorithm. A transfer occurred in nine of the 20 runs of the proposed algorithm. The variant that runs a trial in every block without the detector (Section 7.3) transferred the coefficients in all 20 runs, and its difference from CohVSS-FxNLMS was 0.3 dB with an interval that included zero. With the far source, the missed transfers therefore account for most of the gap. With the close source, a transfer occurred in three of the 20 runs, and CohVSS-FxNLMS was better by 1.1 dB. The variant without the detector remained worse than CohVSS-FxNLMS by 0.8 dB, so that the gap originates mainly in the update of the foreground filter. A start condition suited to intermittent noise is left for future work.

For car interior noise, the switch to the close source raised the error power by only 1.1 dB, below the 6 dB needed to start a trial. The variant without the detector gave no improvement for this case either (0.05 dB, with an interval that included zero), so that little reconvergence was required. In both noise types, the error of the proposed algorithm over the 2 s after the change was not larger than that of CohVSS-FxNLMS. The difference was −0.5 dB ([−1.0, −0.1] dB) and −0.3 dB for machine gun noise and −0.1 dB for car interior noise, and the last two intervals included zero.

For valve and jackhammer noise, the two algorithms had equal recovery times, although the frozen coefficients of the proposed algorithm reduced the noise by 1.8 dB to 8.2 dB more. The power of these recordings varies over several seconds, and this variation dominated the 3 dB criterion. For the isolated impulses, no algorithm met the criterion.

### 6.3 Comparison algorithms with a setting for each noise type

A setting selected for each noise type can give a comparison algorithm a larger step size where the noise allows it. We repeated the far-source and no-change conditions with such settings for the comparison algorithms, selected with the same seeds and criterion. The α-stable signal with $\alpha = 1.20$, which was not used in the selection, kept the common setting, and so did the proposed algorithm.

These settings shortened the median recovery time of CohVSS-FxNLMS from 1.26 s to 0.91 s and that of C-FXLMS from 2.78 s to 1.81 s. The median recovery time of the proposed algorithm remained 56% and 78% shorter. It recovered no later than C-FXLMS for every noise type except valve noise, where the medians differed by 0.01 s. Its noise reduction over the whole run exceeded that of every comparison algorithm by 0.04 dB to 7.6 dB, and all intervals excluded zero. Without a change, C-FXLMS was better for the Gaussian signal (0.07 dB) and destroyer engine noise (0.41 dB). FxNFLOC diverged in three of the 400 runs.

For the signal of Fig. 3, the setting of C-FXLMS had an eight times larger step size, which had diverged in 60 of 90 runs in the selection of the common setting. With this step size, the noise reduction of C-FXLMS 0.3 s after the switch was within 0.1 dB of that of the proposed algorithm. The proposed algorithm still returned to within 1 dB of its final level 2.5 to 2.8 times earlier. The larger step size thus sped up only the first part of the reconvergence.

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

## 7. Cost of the trial mechanism

This section examines the second and third requirements of Section 3.4, the contribution of each element, the computational cost, the sensitivity to the parameters, and the effect of an error in the secondary-path model.

### 7.1 No change in the primary path

In the no-change condition, all 2494 trials were discarded, about one every 2.4 s, and the control signal was identical to that obtained without the trials in every run (Table 3). The noise reduction differed from that of CohVSS-FxNLMS by less than 0.1 dB and exceeded that of C-FXLMS by about 1.0 dB. For destroyer engine noise, C-FXLMS was better by 0.4 dB.

### 7.2 Disturbance at the error microphone

With a disturbance, fewer than 0.5% of the trials ended with a transfer, and only the runs with such a transfer differed from those without the trials. Over the 2 s after each disturbance, the difference from the algorithm without the trials was at most 0.01 dB, and both intervals included zero.

After the 0 dB disturbance, CohVSS-FxNLMS and C-FXLMS reduced the noise by 0.62 dB and 0.27 dB more than the proposed algorithm. The algorithm without the trials showed the same gap, which therefore comes from the late-phase update of the foreground filter. C-FXLMS switched to its slow filter during the disturbance (mixing parameter 0.04). With the −10 dB disturbance, CohVSS-FxNLMS was better by 0.20 dB, and C-FXLMS was worse by 0.27 dB.

### 7.3 Contribution of each element

Table 4 lists three variants. Without the division by $g_n$, seven to nine of the 200 runs diverged in each condition. When the coefficients with the larger step size controlled the noise from the start of a trial, the recovery time was the same within 0.02 s, but the control signal changed in 71 of the 200 runs without a change. The background filter therefore keeps the output unchanged at no cost in speed. When a trial ran in every block without the detector, the recovery was 10% to 17% faster, and the noise reduction after the 0 dB disturbance was 0.50 dB larger ([0.37, 0.64] dB). This variant, however, made 96 transfers without a change and had 1.9 times the average cost. The detector trades this speed-up for an unchanged output and a lower average cost. The peak cost during a trial is the same for both.

### 7.4 Computational cost

Per block of 128 samples (Table 5), the late phase requires 0.60 times the real multiplications of the time-domain FxNLMS algorithm, and a block with a trial requires 1.23 times. A trial was active in 7% of the blocks, and the average was 0.65 times. A real-time implementation must provide for the peak during a trial.

### 7.5 Sensitivity to the parameters

We varied four parameters of the proposed algorithm one at a time in the no-change, 0 dB disturbance, and two source-switch conditions. Table 6 shows the result for the threshold $\gamma$ in (10). A smaller $\gamma$ started a trial after more changes and also led to more transfers without a path change. With $\gamma = 2$, the number of runs without a trial within 1 s after the change fell from 41 to 3 with the close source, and ten transfers occurred in the no-change condition. For $\gamma$ between 2 and 4, the median recovery time stayed within 0.04 s of the default. With $\gamma = 8$, it was about twice as long as the default.

The other three parameters had smaller effects. For the transfer thresholds $(\rho_1, \rho_2)$ of $(0.18, 0.35)$ and $(0.35, 0.7)$ and for the final decision at 8 or 32 blocks, the median recovery time stayed within 0.05 s of the default, and the noise reduction over the whole run stayed within 0.04 dB. The looser thresholds $(0.35, 0.7)$ increased the transfers without a path change from 0 to 18 in the no-change condition and from 13 to 32 with the disturbance. For the early-phase step size, $\sigma_{\mathrm E} = 0.5$ gave median recovery times of 0.42 s and 0.37 s without divergence, and $\sigma_{\mathrm E} = 1.5$ caused divergence in three of the 200 runs in each condition. This value corresponds to $a \le 0.68$, which is above the stable range found in Section 4.4.

### 7.6 Error in the secondary-path model

The proposed algorithm uses the model $\hat{s}(k)$ for the filtered reference, for the recomputed error, and for the decision of a trial. In addition to the truncation error of about −24 dB (Section 5.1), we added white Gaussian noise to the coefficients of $\hat{s}(k)$ at −40 dB, −30 dB, and −20 dB relative to their energy and repeated the no-change and source-switch conditions (Table 7). The actual secondary path was unchanged.

With an error of −40 dB, no run diverged, and the median recovery times changed by at most 0.01 s. With −30 dB, the proposed algorithm diverged in nine of the 600 runs, and CohVSS-FxNLMS diverged in three. With −20 dB, the proposed algorithm and C-FXLMS diverged in 54 runs each, and CohVSS-FxNLMS diverged in 21. The proposed algorithm is therefore more sensitive to the model error than CohVSS-FxNLMS. Most of this sensitivity belongs to the foreground filter, because the algorithm without the trials diverged in six and 51 runs. The remaining three runs at each of these two levels used the same noise sequence in the three conditions, and the proposed algorithm diverged within 2 s after a transfer. With a model error, a transfer can thus cause divergence, although it was rare.

In the runs without divergence, the advantage in the recovery time remained. With −20 dB, the median recovery time of the proposed algorithm was 0.44 s and 0.51 s, compared with 1.03 s and 1.33 s for CohVSS-FxNLMS. With −30 dB and without a change, the noise reduction of the proposed algorithm over the last 10 s was smaller than that of CohVSS-FxNLMS by 0.3 dB ([0.06, 0.78] dB). An additional error of −40 dB therefore left the results unchanged, and an additional error of −30 dB or larger made the proposed algorithm more prone to divergence than CohVSS-FxNLMS.

## 8. Conclusion

We proposed a frequency-domain filtered-x algorithm for impulsive noise that adapts a background filter with a larger step size and transfers its coefficients to the foreground filter only when they reduce the reconstructed error. A discarded trial leaves the control signal identical to that obtained without it. In simulations with ten noise types, the trials shortened the median recovery time after a switch of the noise source by about two thirds compared with the fastest of nine comparison algorithms. With a fixed primary path, the control signal was identical to that obtained without the trials.

For machine gun noise, which consists of intermittent bursts, a transfer occurred in fewer than half of the runs, and no transfer occurred for a small change in the primary path. After a disturbance at the error microphone, two comparison algorithms reduced the noise by 0.3 dB to 0.6 dB more. With an additional error of −30 dB or larger in the secondary-path model, the proposed algorithm diverged in more runs than our previous algorithm. The evaluation was limited to simulations with paths from one room, a fixed secondary path, and no acoustic feedback. Future work includes a start condition for intermittent noise, an update that is less sensitive to errors in the secondary-path model, the evaluation in other rooms, and a real-time implementation.

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
