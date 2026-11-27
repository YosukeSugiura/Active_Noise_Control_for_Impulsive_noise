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

Adaptive algorithms for active control of impulsive noise avoid divergence by limiting the influence of large samples or by using a small step size. Because every update acts directly on the control filter, these settings also slow the reconvergence after a sudden change in the primary path. This paper proposes a frequency-domain filtered-x algorithm that tests a larger step size on a background filter during a short trial. The foreground filter generates the control signal and receives the background coefficients only when they give a sufficiently smaller error on the same data. A discarded trial leaves the control signal identical to that obtained without the trial, given the same initial state and input. In simulations with measured room impulse responses and ten noise types, the proposed algorithm shortened the median recovery time after a switch of the noise source by 63% to 68% compared with the fastest of eleven comparison algorithms, without divergence. The coefficients obtained 0.5 s after the switch reduced the noise by 2.7 dB to 3.8 dB more on average. With a fixed primary path, the two algorithms differed by less than 0.1 dB.

## 1. Introduction

Impulsive noise contains short bursts with amplitudes much larger than the background level. It is often modeled by a symmetric α-stable distribution with characteristic exponent α between 0 and 2, in which a smaller α gives heavier tails [Shao and Nikias 1993]. In feedforward active noise control (ANC), the filtered-x least-mean-square (FxLMS) algorithm [Kuo and Morgan 1996] can diverge when an impulse enters the filtered reference or error signal.

Previous algorithms prevent this divergence with thresholds or robust error functions [Sun 2006, Akhtar 2009, Wu 2011, Feng et al. 2024], with step-size control [APSIPA 2026], or with a combination of a fast and a slow filter [Ferrer 2013]. In all of them, every update acts directly on the control filter. The settings chosen for stability under impulses therefore also limit the reconvergence after a change in the primary path, the acoustic path from the noise source to the error microphone. In our simulations, the fastest of eleven comparison algorithms needed about 1 s in median to return to its converged error level after a switch of the noise source. Larger step sizes shortened this time only together with divergence.

This paper addresses this reconvergence in a single-channel feedforward ANC system with a fixed secondary path. The error alone cannot distinguish a path change from impulses or from a disturbance at the error microphone. The proposed algorithm applies a larger step size first to a background filter. The foreground filter generates the control signal and receives the background coefficients only when they give a smaller reconstructed error on the same data. We call this adaptation of the background filter a trial. A trial starts only when the ratio of the error power to the filtered-reference power rises, and it ends within 16 blocks (0.13 s). A discarded trial leaves the control signal identical to that obtained without the trial, given the same initial state and input. A transfer without a path change can alter the control signal, and we report how often it occurred.

Background filters with a transfer of coefficients are known from echo cancellation [Ochiai et al. 1977], and filters compared by an error reconstructed with the secondary-path model appear in ANC [Harman 2022, Yin 2023]. The proposed algorithm differs in that the background filter is adapted only during a short trial, with an update designed for a large step size under impulsive noise.

This paper makes three contributions. First, we design the trial and its decision rules. We show that a discarded trial leaves the control signal unchanged (Proposition 2), and we bound the error of the transfer decision caused by an inaccurate secondary-path model (Proposition 3). Second, we design the update used in the trial with a spectral-factor direction [Elliott and Rafaely 2000], a step size normalized by the relative frame power, and an error frame reconstructed with the current coefficients [Bjarnason 1992]. We show that, for a white input, the update caused by an isolated impulse does not depend on its amplitude (Proposition 1). Third, we compare the proposed algorithm with eleven algorithms for ten noise types, including the cases in which it was not better, and we examine each element by ablation.

Sections 2 to 4 review related work, formulate the problem, and describe the proposed algorithm. Sections 5 to 8 describe the evaluation and its results, and Section 9 concludes the paper.

## 2. Related work

### 2.1 Adaptive algorithms for impulsive noise

The second-order moment of an α-stable distribution does not exist for α < 2 [Shao and Nikias 1993]. Previous studies therefore limited the effect of large samples on the update in three ways.

The first group of algorithms clips or ignores large samples. Sun et al. [Sun 2006] ignored the samples that exceeded a threshold. Other algorithms replaced such samples with the threshold values [Akhtar 2008, Akhtar 2009]. Hampel's M-estimator was applied to reject impulses in the error signal [Thanigai 2007] and in the reference signal [Sun et al. 2015].

The second group minimizes a cost function that grows slowly for large errors. Wu et al. [Wu 2011] minimized the squared logarithm of the error. The Fair function was also used [Wu and Qiu 2013]. Feng et al. [Feng et al. 2024] derived a normalized algorithm from the fractional lower-order covariance. Cheng et al. [Cheng 2023] applied the Gaussian error function to the error.

The third group controls the step size. Our previous algorithm [APSIPA 2026] scaled the step size in each frequency bin by the coherence between the filtered reference and error signals, within a fixed range. In all three groups, the update acts directly on the filter that generates the control signal, with parameters selected for stability.

### 2.2 Algorithms with two or more filters

Ferrer et al. [Ferrer 2013] applied the convex combination of two filtered-x filters with different step sizes to ANC. An FxLMS filter was also combined with a filter based on the maximum Versoria criterion [Cheng 2022] or on the minimum kernel risk-sensitive loss [Yu 2020]. In these schemes, both filters contribute to the control signal. A background filter that does not affect the output was introduced for echo cancellation [Ochiai et al. 1977]. In ANC, the error of such a filter must be reconstructed with the secondary-path model. A patent [Harman 2022] copies the coefficients of a shadow filter to the control filter when this reconstructed error is smaller, without specifying the update of the shadow filter. Yin et al. [Yin 2023] proposed the adaptive parallel filter method (APFM). In each frame, it selects among filters trained in advance, the current filter, and a zero filter by the reconstructed error, and it adapts only the selected filter.

### 2.3 Frequency-domain update and error reconstruction

The bin-normalized frequency-domain LMS algorithm converges to a biased solution when the filter is constrained to be causal [Elliott and Rafaely 1997]. Applying spectral factors of the reference power spectrum before and after the constraint removes this bias [Elliott and Rafaely 2000]. For multichannel ANC, causal preconditioning filters obtained by spectral factorization of the reference cross-spectral density matrix speed up the convergence without adding delay [Wang 2025]. The modified FxLMS algorithm [Bjarnason 1992] estimates the primary noise at the error microphone and computes the error that the current coefficients would produce. This modification removes the delay of the secondary path from the adaptation loop [Yang 2018 review].

### 2.4 Position of this work

The algorithms in Section 2.1 limit every update of the control filter. The proposed algorithm instead adapts a background filter with a large step size during a short trial after a detected change, and it uses the constructions of Section 2.3 in this update. Unlike the methods in Section 2.2, the background filter starts from the current coefficients, needs no filters trained in advance, and is transferred only with a margin. A background filter that adapts continuously made transfers without a path change in our evaluation (Section 7.1). Section 6.1 compares the proposed algorithm with APFM.

## 3. Problem formulation

### 3.1 Signal model

Fig. 1 shows the single-channel feedforward ANC system considered in this paper. The reference microphone observes the noise $x(k)$ at sample $k$. The noise reaches the error microphone through the primary path $p(k)$ and produces the primary noise $d(k) = p(k) * x(k)$, where $*$ denotes convolution. The control filter has $M$ coefficients $\mathbf{w}(k) = [w_0(k), \ldots, w_{M-1}(k)]^{\mathsf T}$ and generates the control signal

$$y(k) = \mathbf{w}^{\mathsf T}(k)\,\mathbf{x}(k), \qquad \mathbf{x}(k) = [x(k), \ldots, x(k-M+1)]^{\mathsf T}.$$

The control signal reaches the error microphone through the secondary path $s(k)$. The error signal is

$$e(k) = d(k) + s(k) * y(k) + v(k), \tag{1}$$

where $v(k)$ is a disturbance at the error microphone that is uncorrelated with $x(k)$. A model $\hat{s}(k)$ of the secondary path with $L_s$ coefficients is identified in advance and is fixed during control. The filtered reference signal is $r(k) = \hat{s}(k) * x(k)$.

### 3.2 Block frequency-domain adaptation

The control signal is computed for every sample in the time domain. The control path therefore has no block delay. The coefficients are updated once every $B$ samples in the frequency domain with a fast Fourier transform (FFT) of length $N = 2M$, and $\mathbf{w}_n$ denotes the coefficients in block $n$. In this block, $\mathbf{R}_n = \hat{\mathbf{S}} \odot \mathbf{X}_n$ denotes the spectrum of the filtered reference frame, where $\mathbf{X}_n$ is the FFT of the latest $N$ samples of $x(k)$ and $\hat{\mathbf{S}}$ is the FFT of $\hat{s}(k)$. Similarly, $\mathbf{E}_n$ denotes the FFT of the latest $M_e$ samples of $e(k)$ preceded by $N - M_e$ zeros, with $M_e = M$ in this paper. The frequency-domain normalized FxLMS (FxNLMS) algorithm updates the coefficients as

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

The first requirement calls for a large update after $k_c$. In contrast, the other two call for the normal update at all other times. However, the algorithm does not know $k_c$.

[3 章のメモ]
- Fig. 1：系の構成図（x、p、w、s、ŝ、e、v、前景と背景は Fig. 2）。作図が必要。
- 式 (2) の誤差フレームと参照フレームの取り方は Table 1（誤差窓 512、FFT 1024）と THEORY_frozen.md §1 に合わせた。Ŝ_rr の推定法（指数平均の係数）は 4 章で書く。
- 3.3 節「分布に仮定を置かない」：1/g の解析（定理 C）は孤立インパルスを仮定するので、4 章でその仮定を明記する。
- 要件 2 と 3 は「仕組みを外した同じ手法（再始動なし）と同じ水準」と定義した。結果は差 0.01 dB 以下で区間は 0 を含む（変化なし、EN0、ENm10）。比較手法との差（外乱の直後 2 秒は CohVSS-FxNLMS が 0.62 dB、C-FXLMS が 0.27 dB 良い）は、要件とは別に 7 章で書く。

## 4. Proposed algorithm

### 4.1 Overview

Fig. 2 shows the structure of the proposed algorithm. The foreground filter $\mathbf{w}_{\mathrm f}$ generates the control signal, and the background filter $\mathbf{w}_{\mathrm b}$ exists only during a trial. Both filters use the update

$$\mathbf{w}_{n+1} = \mathbf{w}_n - \frac{\sigma_n}{\max(g_n, g_{\min})}\,\frac{2B}{M}\,\boldsymbol{\delta}_n, \tag{4}$$

where $\boldsymbol{\delta}_n$ is the update direction (Section 4.2), $g_n$ is the relative power of the filtered-reference frame (Section 4.3), $g_{\min}$ is its lower bound, and $\sigma_n$ is the step size. The step size is $\sigma_{\mathrm E}$ or $\sigma_{\mathrm L}$, and the lower bound is $g_{\min,\mathrm E}$ or $g_{\min,\mathrm L}$, according to the phase. The update has an early and a late phase. The foreground filter enters the early phase at start-up and at each transfer, that is, when a trial has confirmed a change. It returns to the late phase $n_{\mathrm E}$ blocks after start-up or after the start of the trial. A discarded trial does not change the phase, and Proposition 2 relies on this. The background filter is always in the early phase. The early phase uses a large step size $\sigma_{\mathrm E}$ and an error frame reconstructed with the current coefficients (Section 4.4). The late phase uses a small step size $\sigma_{\mathrm L}$ and the measured error frame (Section 4.5).

A trial starts when the error power rises relative to the filtered-reference power. The background filter then adapts from the foreground coefficients for at most 16 blocks, and its coefficients are either transferred to the foreground filter or discarded (Section 4.6). Section 4.7 gives two properties of the trial. Table 1 lists the parameter values.

### 4.2 Update direction

The direction in (2) normalizes each frequency bin by the power spectrum before the causality constraint. This order biases the solution, and we split the normalization into two spectral factors as in [Elliott and Rafaely 2000]. Let $\boldsymbol{\Lambda}_n = \hat{\mathbf{S}}_{rr,n} + \epsilon \bar{P}_n$, where $\bar{P}_n$ is the mean of $\hat{\mathbf{S}}_{rr,n}$ over the $N/2+1$ bins up to the Nyquist frequency, and let $\mathbf{F}_n$ be the minimum-phase factor with $|\mathbf{F}_n|^2 = \boldsymbol{\Lambda}_n$. We compute $\mathbf{F}_n$ from the real cepstrum of $\boldsymbol{\Lambda}_n$. A small positive lower bound on $\boldsymbol{\Lambda}_n$ prevents a division by zero for a silent input. The direction is

$$\boldsymbol{\delta}_n = \mathcal{P}_M\, \mathcal{F}^{-1}\!\left\{ \frac{1}{\mathbf{F}_n} \odot \mathcal{F}\, \mathcal{P}_{N/2}\, \mathcal{F}^{-1}\!\left\{ \frac{\mathbf{R}_n^{*} \odot \mathbf{E}_n}{\mathbf{F}_n^{*}} \right\} \right\}, \tag{5}$$

where $\mathcal{P}_{N/2}$ keeps the first $N/2$ samples and sets the others to zero.

Under idealized conditions (fixed coefficients, $\hat{s} = s$, a stationary input with a known and strictly positive power spectrum, $\epsilon = 0$, and $N \to \infty$), $\mathrm{E}[\boldsymbol{\delta}_n] \propto \mathbf{w} - [\mathbf{w}_\infty]_M$ [Elliott and Rafaely 2000], where $\mathbf{w}_\infty$ is the causal Wiener filter of infinite length and $[\cdot]_M$ keeps its first $M$ coefficients. In this case, all modes of the coefficient error converge at the same rate. This property motivates the spectral factors. It does not establish convergence of the algorithm, because the algorithm estimates $\hat{\mathbf{S}}_{rr,n}$ from the same data.

### 4.3 Step size normalized by the relative frame power

In $\hat{\mathbf{S}}_{rr,n}$, the current frame has the weight $\beta$. When a frame contains an impulse, its power is much larger than the average, and $\boldsymbol{\delta}_n$ grows with the impulse amplitude. We therefore divide the step size by the relative frame power

$$g_n = \frac{\sum_{f=0}^{N/2} |R_n(f)|^2}{\sum_{f=0}^{N/2} \hat{S}_{rr,n}(f)}. \tag{6}$$

Here, $R_n(f)$ and $\hat{S}_{rr,n}(f)$ are the elements of $\mathbf{R}_n$ and $\hat{\mathbf{S}}_{rr,n}$ in bin $f$. Because $\hat{\mathbf{S}}_{rr,n}$ includes the current frame, $g_n \le 1/\beta$.

**Proposition 1.** Assume that the coefficients are fixed, that $v(k) = 0$, and that $\hat{\mathbf{S}}_{rr,n-1} = q\,|\hat{\mathbf{S}}|^2$ with a constant $q$, as for a white $x(k)$. If $x(k_0) = A \ne 0$ is the only nonzero input sample that contributes to the reference and error frames of block $n$, then $\boldsymbol{\delta}_n / g_n$ does not depend on $A$.

*Proof.* The spectra $\mathbf{R}_n$ and $\mathbf{E}_n$ are proportional to $A$, and $|\mathbf{R}_n|^2 = A^2|\hat{\mathbf{S}}|^2$. Then $\hat{\mathbf{S}}_{rr,n} = c\,|\hat{\mathbf{S}}|^2$ with $c = (1-\beta)q + \beta A^2$. Thus, $\boldsymbol{\Lambda}_n$ and $\bar{P}_n$ are proportional to $c$ and $\mathbf{F}_n$ is proportional to $\sqrt{c}$. By (5) and (6), both $\boldsymbol{\delta}_n$ and $g_n$ are proportional to $A^2/c$, and their ratio is independent of $A$. $\blacksquare$

Thus, an isolated impulse produces the same update for any amplitude as long as $g_n \ge g_{\min}$, and the algorithm keeps impulsive samples in the update without a threshold. For $g_n < g_{\min}$, the update is smaller.

For a colored $x(k)$, the update is not independent of $A$, but two weaker statements hold for any positive $\hat{\mathbf{S}}_{rr,n-1}$. Let $\mathbf{R}_n = A\,\mathbf{R}^{(1)}$ and $\mathbf{E}_n = A\,\mathbf{E}^{(1)}$, and let $P_1$ be the mean of $|\mathbf{R}^{(1)}|^2$ over the bins. First, as $A \to \infty$, $\boldsymbol{\delta}_n/g_n$ approaches the direction (5) computed with $\boldsymbol{\Lambda} = |\mathbf{R}^{(1)}|^2 + \epsilon P_1$. This limit depends neither on $A$ nor on the past spectrum. Second, because $\boldsymbol{\Lambda}_n \ge \epsilon \bar{P}_n$ in every bin and the projections in (5) do not increase the norm,

$$\left\| \boldsymbol{\delta}_n / g_n \right\|_2 \le \frac{\left\| \mathcal{F}^{-1}\{\mathbf{R}^{(1)*} \odot \mathbf{E}^{(1)}\} \right\|_2}{\epsilon\, P_1}$$

for every $A \ne 0$. In a numerical example with an autoregressive past spectrum (pole at 0.95), the norm of $\boldsymbol{\delta}_n/g_n$ changed by a factor of two between small and large $A$. For frames that also contain background noise, we rely on the simulations.

### 4.4 Early phase with reconstructed error frames

The $M_e$ samples in the error window of block $n$ were produced by the coefficients of the last $M_e/B$ blocks. The measured error frame therefore responds to an update with a delay, and this delay limits the step size. To use a large step size in the early phase, we remove this dependence on past coefficients. As in the modified FxLMS algorithm [Bjarnason 1992], the primary noise is estimated as

$$\hat{d}(k) = e(k) - \hat{s}(k) * y(k), \tag{7}$$

and the error frame is reconstructed with the coefficients $\mathbf{w}_n$ to be updated:

$$\tilde{e}_n(k) = \hat{d}(k) + \hat{s}(k) * \left[\mathbf{w}_n^{\mathsf T}\mathbf{x}(k)\right] \tag{8}$$

for the $M_e$ samples in the window. In the early phase, $\mathbf{E}_n$ in (5) is the FFT of this frame.

The reconstructed frame changes the form of the update. Assume $\hat{s} = s$, ideal whitening ($\boldsymbol{\Lambda}_n = \mathbf{S}_{rr}$ and $N \to \infty$, where $\mathbf{S}_{rr}$ is the true power spectrum of $r(k)$), and $g_n = 1$. Let $\boldsymbol{\theta}_n = \mathbf{w}_n - \mathbf{w}_o$ be the coefficient error with respect to a reference filter $\mathbf{w}_o$, and let $a = \sigma_{\mathrm E}\,(2B/M)(M_e/N)$. Under these assumptions, (8) gives $\tilde{\mathbf{e}}_n = \mathbf{X}^r_n \boldsymbol{\theta}_n + \mathbf{e}_o$, where $\mathbf{e}_o$ is the error of $\mathbf{w}_o$. The substitution into (5) and (4) gives

$$\boldsymbol{\theta}_{n+1} = (\mathbf{I} - a\,\mathbf{G}_n)\,\boldsymbol{\theta}_n + \mathbf{b}_n, \qquad \mathbf{G}_n = \mathbf{F}_M^{-1}\,\mathbf{U}_n^{\mathsf T}\mathbf{X}^{r}_n / M_e, \tag{9}$$

where $\mathbf{U}_n$ and $\mathbf{X}^r_n$ are the $M_e \times M$ regression matrices of the whitened filtered reference, i.e., $r(k)$ filtered by the inverse of the spectral factor, and of $r(k)$ in the window, $\mathbf{F}_M$ is the $M \times M$ convolution matrix of the spectral factor, and $\mathbf{b}_n$ is the contribution of the residual error of $\mathbf{w}_o$.

Equation (9) contains only the current coefficient error. However, the matrices $\mathbf{G}_n$ of consecutive blocks share samples, and we have no closed-form stability bound on $a$. We therefore iterated (9) over 64 blocks with $\mathbf{b}_n = \mathbf{0}$, in 20 runs for each input, and use the result as a design guide rather than a stability proof. For white Gaussian input, the coefficient error decayed in all runs for $a \le 0.5$ and grew in 18 of 20 runs at $a = 0.625$. A first-order autoregressive coloring with a pole at 0.9 or 0.98 changed this result by at most one run. For symmetric α-stable input with $\alpha = 1.45$ and $\alpha = 1.20$, with the step size also divided by the running average of the frame power and by $\max(g_n, g_{\min})$, no run grew for $a \le 0.375$, and one run grew at $a = 0.455$. Based on these results, we set $\sigma_{\mathrm E} = 1$ and $g_{\min,\mathrm E} = 0.55$. These values give $a = 0.25$ for $g_n = 1$ and limit $a/\max(g_n, g_{\min})$ to 0.455. This limit is not a safe bound, because one run grew at this value. We chose it as a compromise between the speed of the decay and the margin to growth.

### 4.5 Late phase

In the late phase, the measured error frame is used, and the step size is small. We set $\sigma_{\mathrm L} = 0.0566$ and $g_{\min,\mathrm L} = 0.25$. The noise reduction over the whole run was the same for $\sigma_{\mathrm L}$ between 0.02 and 0.09.

### 4.6 Trial with the background filter

**Start of a trial.** For each block, the algorithm computes the ratio

$$z_n = \frac{\sum_{k \in \mathcal{W}_M(n)} e^2(k)}{\sum_{k \in \mathcal{W}_{2M}(n)} r^2(k)}, \tag{10}$$

where $\mathcal{W}_L(n)$ denotes the latest $L$ samples at block $n$. The error in $\mathcal{W}_M(n)$ depends on the filtered reference in $\mathcal{W}_{2M}(n)$, because the control filter has $M$ coefficients. The division by the power of $r(k)$ removes the common scale of the input from the ratio. Let $\bar{z}_n$ be the exponential average of $z_n$ with the weight listed in Table 1. A trial starts at block $n_0$ when $z_n > \gamma\,\bar{z}_{n-1}$ holds for two consecutive blocks. After a trial starts, the next trial can start only after $n_{\mathrm G}$ blocks, and $\bar{z}_n$ is held constant during this interval. The average $\bar{z}_n$ is initialized with the first value of $z_n$. A block in which $z_n$ is not finite, as for a silent input, neither updates $\bar{z}_n$ nor counts toward the start of a trial.

**Adaptation.** At block $n_0$, the background filter is set to $\mathbf{w}_{\mathrm b} = \mathbf{w}_{{\mathrm f}, n_0}$. From block $n_0 + 1$, it is updated by (4) in the early phase, with the error frame reconstructed by (8) with $\mathbf{w}_{\mathrm b}$. The foreground filter continues its update in its current phase, and the control signal is generated by $\mathbf{w}_{\mathrm f}$ only.

**Decision.** For each block $n \ge n_0 + 2$, the algorithm reconstructs the errors of both filters by (8) for the latest $B$ samples and forms the ratio

$$\rho_n = \frac{\sum_{k \in \mathcal{W}_B(n)} \tilde{e}_{\mathrm b}^2(k)}{\sum_{k \in \mathcal{W}_B(n)} \tilde{e}_{\mathrm f}^2(k)}. \tag{11}$$

Both errors in (11) use the same $\hat{d}(k)$. A common scale of the signals therefore cancels in the ratio. The trial ends by one of the following rules.

1. Discard: if $\rho_n > \rho_{\mathrm{abort}}$ or $\rho_n$ is not finite, $\mathbf{w}_{\mathrm b}$ is discarded.
2. Early transfer: if $n \ge n_0 + n_1$ and $\rho_n \le \rho_1$, the coefficients are transferred, $\mathbf{w}_{\mathrm f} \leftarrow \mathbf{w}_{\mathrm b}$.
3. Final decision: at $n = n_0 + n_2$, the coefficients are transferred if $\rho_n \le \rho_2$, and $\mathbf{w}_{\mathrm b}$ is discarded otherwise.

After a transfer, the foreground filter continues in the early phase for the remaining $n_{\mathrm E} - (n - n_0)$ blocks. We use $\rho_{\mathrm{abort}} = 4$, $(n_1, \rho_1) = (4, 0.25)$, and $(n_2, \rho_2) = (16, 0.5)$. Algorithm 1 summarizes the processing in each block.

### 4.7 Properties of the trial

**Proposition 2.** Consider two runs of the algorithm with the same initial state and the same input signals. In the first run, the trials are disabled. In the second run, every trial up to block $n$ is discarded. Then the control signal $y(k)$ and the foreground coefficients of the two runs are identical up to block $n$.

*Proof.* The control signal and the update of $\mathbf{w}_{\mathrm f}$ depend on $x(k)$, $e(k)$, and $\mathbf{w}_{\mathrm f}$. The background filter and the quantities $z_n$, $\bar{z}_n$, and $\rho_n$ are computed from these signals and affect $\mathbf{w}_{\mathrm f}$ only through a transfer. Without a transfer, the recursions of $\mathbf{w}_{\mathrm f}$ and $y(k)$ in the two runs are the same, and their initial states are the same. $\blacksquare$

Proposition 2 is a structural property. It requires no assumption on the signals or on the accuracy of $\hat{s}(k)$. However, it does not cover a transfer, and Sections 6.4 and 6.5 report how often a transfer occurred without a path change.

The decision uses the reconstructed errors. Proposition 3 relates it to the errors that the two filters would actually produce. For fixed coefficients $\mathbf{w}$, let $e_{\mathbf{w}}(k) = d(k) + v(k) + s(k) * [\mathbf{w}^{\mathsf T}\mathbf{x}(k)]$ be this error, and let $\rho^{\star}_n$ be the ratio (11) computed with $e_{\mathbf{w}_{\mathrm b}}$ and $e_{\mathbf{w}_{\mathrm f}}$. Let $u(k) = (\mathbf{w}_{\mathrm b} - \mathbf{w}_{\mathrm f})^{\mathsf T}\mathbf{x}(k)$ be the difference of the two filter outputs, and let

$$\kappa_n = \frac{\| (s - \hat{s}) * u \|_n}{\| \hat{s} * u \|_n}, \qquad \eta_n = \frac{\| (s - \hat{s}) * (y - \mathbf{w}_{\mathrm f}^{\mathsf T}\mathbf{x}) \|_n}{\| \tilde{e}_{\mathrm f} \|_n}, \tag{12}$$

where $\|\cdot\|_n$ is the norm over $\mathcal{W}_B(n)$ and both denominators are assumed nonzero. The quantity $\kappa_n$ is the model error weighted by the spectrum of $u(k)$. The quantity $\eta_n$ accounts for the past updates of the foreground filter, because $y(k)$ was generated by past coefficients. It vanishes when the foreground coefficients were constant over the samples that contribute to $\mathcal{W}_B(n)$.

**Proposition 3.** If $\eta_n < 1$, then

$$\left| \sqrt{\rho^{\star}_n} - \sqrt{\rho_n} \right| \le \frac{(\kappa_n + \eta_n)\left( 1 + \sqrt{\rho_n} \right)}{1 - \eta_n}.$$

*Proof.* By (7) and (8), $\tilde{e}_{\mathbf{w}} - e_{\mathbf{w}} = (s - \hat{s}) * (y - \mathbf{w}^{\mathsf T}\mathbf{x})$ for fixed $\mathbf{w}$. With $\xi = (s - \hat{s}) * (y - \mathbf{w}_{\mathrm f}^{\mathsf T}\mathbf{x})$, this gives $e_{\mathbf{w}_{\mathrm f}} = \tilde{e}_{\mathrm f} - \xi$ and $e_{\mathbf{w}_{\mathrm b}} = \tilde{e}_{\mathrm b} - \xi + (s - \hat{s}) * u$. Because $\tilde{e}_{\mathrm b} - \tilde{e}_{\mathrm f} = \hat{s} * u$, we have $\|(s - \hat{s}) * u\|_n = \kappa_n \|\tilde{e}_{\mathrm b} - \tilde{e}_{\mathrm f}\|_n \le \kappa_n (\|\tilde{e}_{\mathrm b}\|_n + \|\tilde{e}_{\mathrm f}\|_n)$. The triangle inequality then bounds $\|e_{\mathbf{w}_{\mathrm b}}\|_n$ from both sides by $\|\tilde{e}_{\mathrm b}\|_n \pm [\eta_n + \kappa_n(1 + \sqrt{\rho_n})]\,\|\tilde{e}_{\mathrm f}\|_n$ and $\|e_{\mathbf{w}_{\mathrm f}}\|_n$ by $(1 \pm \eta_n)\|\tilde{e}_{\mathrm f}\|_n$, and the ratio of these bounds gives the result. $\blacksquare$

Thus, a transfer replaces the foreground coefficients by coefficients with a smaller actual error on the same data ($\rho^{\star}_n < 1$) whenever $\kappa_n (1 + \sqrt{\rho_n}) + 2\eta_n < 1 - \sqrt{\rho_n}$. For $\eta_n = 0$, this condition is $\kappa_n < (1 - \sqrt{\rho_n})/(1 + \sqrt{\rho_n})$, and the right-hand side is at least 0.33 (−9.5 dB) for an early transfer and 0.17 (−15.3 dB) for a transfer at the final decision. For a white $u(k)$ and a long window, $\kappa_n$ approaches the relative error of $\hat{s}$. It is larger when $u(k)$ is concentrated at frequencies where the model is inaccurate. Proposition 3 is an analysis of the decision and not a test that the algorithm can apply, because $\kappa_n$ and $\eta_n$ require the true secondary path. It also concerns the data of the decision and does not guarantee the error after the transfer. Section 8.2 therefore also evaluates the transferred coefficients on the following 2 s.


## 5. Evaluation framework

### 5.1 Acoustic paths and signals

We simulated closed-loop control at 16 kHz with room impulse responses from the MeshRIR dataset [MeshRIR], measured in one room. We used eight combinations of a noise source, a secondary source, and an error microphone. The primary path had 4096 coefficients, the secondary path had 2048 coefficients, and the model $\hat{s}(k)$ was the first 512 coefficients of the secondary path. The truncated part contained −23.4 dB to −24.7 dB of the energy of the secondary path. The model therefore had an error of this size in all evaluations. The reference signal was the source signal itself, and the reference path and the acoustic feedback were excluded. The causality constraint was therefore looser than in a system with a reference microphone.

Four noise types were synthetic: a Gaussian signal, a Gaussian signal with isolated impulses, and symmetric α-stable signals with $\alpha = 1.45$ and $\alpha = 1.20$. The isolated impulses had 300 times the standard deviation of the Gaussian signal and a random sign, and they occurred every 0.5 s with a uniform jitter of ±0.1 s. Six were recordings: machine gun, factory, destroyer engine, and car interior noise from NOISEX-92 [NOISEX], valve noise from MIMII [MIMII], and jackhammer noise from SONYC-UST [SONYC]. All signals were high-pass filtered at 100 Hz, and each run lasted 30 s. For each noise type and condition, 20 seeds determined the paths, the random sequences, and the starting positions in the recordings.

### 5.2 Conditions

In the two source-switch conditions, the primary path changed at 15 s to the response from another loudspeaker to the same microphone. The new loudspeaker was the candidate closest to the original noise source (close source) or the farthest from it (far source). In the no-change condition, the paths were fixed. In the two disturbance conditions, the paths were fixed, and white Gaussian noise was added to the error signal for 0.5 s at 10 s and at 20 s, at 0 dB or −10 dB relative to the primary noise.

### 5.3 Algorithms

Table 1 lists the settings of the proposed algorithm and of the eleven comparison algorithms. The settings of the proposed algorithm were fixed before the evaluation, and the evaluation seeds 100 to 119 were not used to choose them. We also evaluated the proposed algorithm with the trials disabled, listed as "Without trials." FreqFxLMS [Das 2007] is the block frequency-domain FxLMS algorithm without normalization of the spectrum. All algorithms used 512 coefficients and one setting for all noise types and conditions. The step size of each comparison algorithm was selected from seven values between 1/32 and 2 times the published value (from $2^{-8}$ to $2^{3}$ for APFM, because its paper gives no value). The selection used the seeds 10 to 14, the no-change and far-source conditions, and nine noise types. The criterion was the smallest number of diverged runs, followed by the largest ANR over 30 s. For APFM, the new primary path was not used for training, because the proposed algorithm does not assume it to be known.

### 5.4 Metrics

Table 11 defines the measures. A negative ANR, power ratio, or noise reduction with frozen coefficients indicates a reduction. The recovery time depends on the reference level of each algorithm, and the noise reduction with frozen coefficients compares the coefficients on a common scale. For the proposed algorithm, 28 of the 200 runs in each source-switch condition did not recover: all 20 runs with the isolated impulses and eight with valve noise.

Differences are means of paired differences between runs with the same noise and paths, given as the proposed algorithm minus the other. Their 95% intervals were obtained by resampling the 20 seeds 2000 times, and we call one algorithm better only when the interval excludes zero. Paired differences of the recovery time use only the runs in which both algorithms recovered, and Section 6.1 therefore also counts the recovered runs among all runs. Resampling the eight sets of paths instead of the 20 seeds changed the limits of the intervals in Fig. 7 by at most 0.14 dB and 0.05 s.

## 6. Results

This section follows the three requirements of Section 3.4. Sections 6.1 to 6.3 examine the reconvergence after a source switch, Section 6.4 examines the no-change condition, and Section 6.5 examines the disturbance conditions.

### 6.1 Reconvergence after a source switch

The proposed algorithm shortened the median recovery time by 63% (95% interval [55%, 68%]) with the close source and by 68% ([54%, 75%]) with the far source compared with CohVSS-FxNLMS, the fastest of the eleven comparison algorithms. Fig. 7 shows the paired differences, and Table 10 lists the medians. No run of the proposed algorithm diverged in the 1000 runs. The convex combination did not shorten the recovery, because the mixing parameter of C-FXLMS stayed above 0.96 and its fast filter set the recovery time.

Fig. 3 shows an example for the α-stable signal with $\alpha = 1.45$. The return to within 3 dB of the final level took 65% to 67% less time than for CohVSS-FxNLMS, and the levels before the switch agreed within 0.3 dB.

The trials produced this reduction. The algorithm without the trials recovered within 0.13 s of CohVSS-FxNLMS. MFxNLMS uses the reconstructed error without a trial, and it needed 1.97 s and 2.43 s. In the proposed algorithm, the coefficients were transferred 0.07 s to 0.08 s after the change, and the early phase of the foreground filter ended about 0.27 s after the change, shortly before the median recovery time.

The coefficients themselves converged faster. Fig. 4 shows the noise reduction with frozen coefficients, a scale common to all algorithms. With the far source, the coefficients of the proposed algorithm at 0.13 s reduced the noise about as much as those of CohVSS-FxNLMS at 0.51 s. At 0.51 s, the residual power of the proposed algorithm was smaller by a factor of 1.9 with the close source and 2.4 with the far source.

The actual error during the adaptation was also smaller. Over the first 0.5 s, 1 s, and 2 s after the change, the power ratio of the proposed algorithm was lower than that of CohVSS-FxNLMS by 1.6 dB to 1.9 dB with the close source and by 2.2 dB to 2.5 dB with the far source. It was lower than those of C-FXLMS and MFxNLMS by 2.5 dB to 4.1 dB. All intervals excluded zero.

Table 10 summarizes all algorithms. FxNFLOC diverged in nine of its 1000 runs and APFM in one, and the other comparison algorithms did not diverge. FxlogNLMS recovered in 1.48 s with the far source. However, its noise reduction without a change was only −2.3 dB, and its reference level for the 3 dB criterion was therefore high.

APFM recovered in 1.20 s and 2.90 s, and its coefficients frozen at 0.51 s reduced the noise by only 0.3 dB. After the switch, its noise reduction over the last 10 s was 1.5 dB and 2.4 dB, compared with 7.0 dB without a change. Without a change, the selected candidate changed in 22% of the blocks. With a switch, the zero filter was selected in 7% to 9% of the blocks, compared with 0.8% without a change. With the same step size and without the selection, its update (VSS-NFBFxLMS) reduced the noise by only 3.3 dB without a change and recovered in 5.4 s and 9.1 s after a switch. The noise reduction of APFM without a change therefore came mainly from the filter trained in advance for that path. In this configuration, the selection among fixed candidates did not shorten the reconvergence.

A larger step size did not give the comparison algorithms this speed without divergence. Fig. 5 shows the median recovery time and the percentage of diverged runs for step sizes from 0.5 to 16 times the common setting. At twice the common step size, CohVSS-FxNLMS recovered in 0.54 s and 0.70 s in median, and 10% of its runs diverged. At four times, its runs without divergence recovered in 0.30 s and 0.32 s, and 31% of its runs diverged. FxNLMS and Th-FxNLMS showed the same trade-off at longer recovery times, and Th-FxNLMS diverged in 7% and 33% of the runs at two and four times the common step size. Among the six step sizes that we tested, none gave the recovery time of the proposed algorithm without divergence. For C-FXLMS, the selected step size was already the largest without divergence in the selection. For MFxNLMS, twice the selected step size did not diverge in the selection, but it reduced the noise over the whole run by 0.3 dB less. Fig. 5 also shows the proposed algorithm with 0.5 and 1.5 times $\sigma_{\mathrm E}$. Its recovery time changed by at most 0.09 s, and 1.5% of the runs diverged at 1.5 times (Section 7.3). Because the medians in Fig. 5 exclude the diverged runs, Table 8 counts the recovered runs among all 200 runs. Within 0.5 s and within 2 s, the proposed algorithm recovered in more runs than CohVSS-FxNLMS with any of the three step sizes.

### 6.2 Reconvergence for each noise type

Fig. 6 shows the paired difference of the noise reduction with frozen coefficients for each noise type. Of the 40 comparisons, the proposed algorithm was better in 31, CohVSS-FxNLMS was better in six, and the interval included zero in three. The six exceptions were the four comparisons for machine gun noise and the two for car interior noise with the close source. In these two noise types, transfers after the change were rare. Without a transfer, the proposed algorithm gives the same control signal as the algorithm without the trials (Proposition 2). With the close source, 60 of the 200 runs had no transfer, and 37 of them belonged to these two noise types. With the far source, 13 of the 20 such runs belonged to them.

For machine gun noise, a transfer occurred in three of the 20 runs with the close source and in nine with the far source. The coefficients of CohVSS-FxNLMS at 0.51 s reduced the noise by 1.1 dB and 2.6 dB more than those of the proposed algorithm. The variant that runs a trial in every block without the detector (Section 7.1) transferred the coefficients in all runs. With the far source, its difference from CohVSS-FxNLMS was 0.3 dB with an interval that included zero, and the missed transfers thus account for most of the gap. With the close source, this variant remained worse by 0.8 dB, and the gap originates mainly in the update of the foreground filter. The recovery time did not separate the algorithms for this noise, because the error power varied over about 55 dB between bursts and pauses.

For car interior noise, the switch to the close source raised the error power by only 1.1 dB. This rise is below the 6 dB ($\gamma = 4$) needed to start a trial. The variant without the detector gave no improvement either. The power ratio over the 2 s after the change differed from that of CohVSS-FxNLMS by 0.04 dB with an interval that included zero.

For valve and jackhammer noise, the recovery times of the two algorithms were equal. However, the frozen coefficients of the proposed algorithm reduced the noise by 1.8 dB to 8.2 dB more. The power of these recordings varies over several seconds, and this variation dominated the 3 dB criterion. For the isolated impulses, no algorithm met the criterion.

### 6.3 Comparison with settings selected for each noise type

A setting selected for each noise type can give a comparison algorithm a larger step size where the noise allows it. To examine this case, we repeated the far-source and no-change conditions with such settings for the comparison algorithms except APFM, selected with the same seeds and criterion. The α-stable signal with $\alpha = 1.20$ was not used in the selection and kept the common setting. The proposed algorithm also kept its common setting.

These settings shortened the median recovery time of CohVSS-FxNLMS from 1.26 s to 0.91 s and that of C-FXLMS from 2.78 s to 1.81 s. However, the median recovery time of the proposed algorithm remained 56% and 78% shorter. Its noise reduction over the whole run exceeded that of every comparison algorithm by 0.04 dB to 7.6 dB, and all intervals excluded zero. For individual noise types without a change, C-FXLMS was better for the Gaussian signal by 0.07 dB. For destroyer engine noise, C-FXLMS and Th-FxNLMS were both better by 0.41 dB. FxNFLOC diverged in three of the 400 runs.

For the signal of Fig. 3, the setting of C-FXLMS had an eight times larger step size. In the selection of the common setting, this step size had caused divergence in 60 of 90 runs. With this step size, the noise reduction of C-FXLMS 0.3 s after the switch was within 0.1 dB of that of the proposed algorithm. However, the proposed algorithm returned to within 1 dB of its final level 2.5 to 2.8 times earlier.

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

### 6.4 No change in the primary path

Without a path change, the trials left the control signal unchanged. In the no-change condition, all 2494 trials were discarded, about one every 2.4 s, and the control signal was identical to that obtained without the trials in every run. Table 3 lists the noise reduction. Over the whole run, the mean noise reduction was 0.05 dB larger than that of CohVSS-FxNLMS ([−0.08, −0.03] dB), and over the last 10 s it was 0.07 dB smaller ([0.04, 0.11] dB). It exceeded that of C-FXLMS by about 1.0 dB. For destroyer engine noise, C-FXLMS was better by 0.4 dB.

### 6.5 Disturbance at the error microphone

With a disturbance, the trials rarely led to a transfer and did not degrade the noise reduction. Fewer than 0.5% of the trials ended with a transfer (13 transfers with the 0 dB disturbance and two with the −10 dB disturbance). Only the runs with such a transfer differed from those without the trials, and 187 and 198 of the 200 runs gave an identical output. Over the 2 s after each disturbance, the difference from the algorithm without the trials was about 0.01 dB or less, and both intervals included zero.

However, two comparison algorithms recovered better from the 0 dB disturbance. After this disturbance, CohVSS-FxNLMS and C-FXLMS reduced the noise by 0.62 dB and 0.27 dB more than the proposed algorithm. The algorithm without the trials showed the same gap. The gap therefore comes from the late-phase update of the foreground filter and not from the trials. C-FXLMS switched to its slow filter during the disturbance (mixing parameter 0.04). With the −10 dB disturbance, CohVSS-FxNLMS was better by 0.20 dB, and C-FXLMS was worse by 0.27 dB.

## 7. Analysis of the design

### 7.1 Contribution of each element

Table 4 lists three variants of the trial. The first variant omits the division by $g_n$. Seven to nine of its 200 runs diverged in each condition. The second variant has no background filter, and the coefficients with the larger step size control the noise from the start of a trial. Its recovery time was the same within 0.02 s. However, its control signal changed in 71 of the 200 runs without a change. Thus, the background filter keeps the output unchanged at no cost in speed. The third variant has no detector. Whenever no trial is active, it starts a new trial from the current foreground coefficients, with the same update and the same decision rules. Its background filter therefore adapts continuously, as the shadow filter in the patent [Harman 2022] does. Because the patent does not specify the update or the comparison, this variant is a structural counterpart of the shadow filter and not a reimplementation of it. Its recovery took 10% to 17% less time, and its noise reduction after the 0 dB disturbance was 0.50 dB larger ([0.37, 0.64] dB). However, it made 96 transfers without a change and had 1.9 times the average cost, with the same peak cost. The detector therefore does not make the trial faster. It keeps the output unchanged without a path change and about halves the average cost, at the price of a slightly slower recovery.

We also removed the other two elements of the update, with parameters either equal to those of the proposed algorithm or selected in the same way as for the comparison algorithms. No run of these variants diverged. Without the reconstruction of the error frame, the median recovery time was 0.55 s and 0.72 s with the selected step size and 1.03 s and 1.12 s with that of the proposed algorithm. The coefficients frozen at 0.51 s reduced the noise by 1.9 dB to 5.4 dB less. Thus, in the present configuration, the reconstruction is the element that allows the large step size. Without the spectral factor and with the same step sizes, the recovery time and the noise reduction of the coefficients frozen at 0.51 s agreed with those of the proposed algorithm within 0.02 s and 0.1 dB. However, the noise reduction over the whole run was smaller by 0.16 dB to 0.19 dB, and by 0.8 dB for destroyer engine noise without a change. Thus, the spectral factor improves the converged level rather than the speed of the reconvergence.

Finally, we applied the trial to our previous algorithm. The foreground filter used the update of CohVSS-FxNLMS, and the background filter used the same update with a step size eight times larger, selected from 2 to 16 times. The median recovery time was 0.74 s and 0.89 s, and the coefficients frozen at 0.51 s reduced the noise by 1.4 dB and 2.1 dB less than those of the proposed algorithm. Compared with CohVSS-FxNLMS, the trial therefore shortened the recovery by 25% to 29%. The proposed algorithm uses the update of Section 4 in both filters, and it recovered in less than half of the time of this variant. This variant recovered better from the 0 dB disturbance, by 0.62 dB, as CohVSS-FxNLMS did.

### 7.2 Computational cost

Table 5 lists the cost per block of 128 samples. The late phase requires 0.60 times the real multiplications of the time-domain FxNLMS algorithm, and a block with a trial requires 1.23 times. A trial was active in 7% of the blocks, and the average cost was 0.65 times that of the FxNLMS algorithm. A real-time implementation must provide for the peak during a trial. By the same count, CohVSS-FxNLMS requires about 0.63 times the multiplications of the FxNLMS algorithm, close to the late phase of the proposed algorithm. The counts exclude the logarithm and the complex exponential that the spectral factor requires for each bin once per block, and the three power functions and one exponential per bin of CohVSS-FxNLMS.

### 7.3 Sensitivity to the parameters

The threshold $\gamma$ trades missed changes against transfers without a path change, and the other parameters had small effects. With the default values, no run diverged in the main evaluation or in this sensitivity analysis. We varied four parameters of the proposed algorithm one at a time. Table 6 shows the result for the threshold $\gamma$ in (10). For $\gamma$ between 2 and 4, the median recovery time stayed within 0.04 s of the default, and with $\gamma = 8$, it was about twice as long.

For the transfer thresholds $(\rho_1, \rho_2)$ of $(0.18, 0.35)$ and $(0.35, 0.7)$ and for the final decision at 8 or 32 blocks, the median recovery time stayed within 0.05 s of the default, and the noise reduction over the whole run stayed within 0.04 dB. The looser thresholds increased the transfers without a path change from 0 to 18 in the no-change condition and from 13 to 32 with the 0 dB disturbance. The step size $\sigma_{\mathrm E} = 0.5$ gave median recovery times of 0.42 s and 0.37 s without divergence, and $\sigma_{\mathrm E} = 1.5$ caused divergence in three of the 200 runs in each condition. With this value, $a/\max(g_n, g_{\min})$ reaches 0.68 and exceeds the range in which the error decayed in Section 4.4.

## 8. Error in the secondary-path model

### 8.1 Divergence and recovery

The proposed algorithm tolerated an added model error of −40 dB, but it diverged more often than CohVSS-FxNLMS with an added error of −30 dB or more. In addition to the truncation error of about −24 dB (Section 5.1), we added white Gaussian noise to the coefficients of $\hat{s}(k)$ at −40 dB, −30 dB, and −20 dB relative to their energy and repeated the no-change and source-switch conditions. Table 7 lists the results. The actual secondary path was unchanged. With −40 dB, no run diverged, and the median recovery times changed by at most 0.01 s. With −30 dB, the proposed algorithm diverged in nine of the 600 runs, and CohVSS-FxNLMS diverged in three. With −20 dB, the proposed algorithm and C-FXLMS diverged in 54 runs each, and CohVSS-FxNLMS diverged in 21.

Most of this sensitivity originates in the update of the foreground filter, because the algorithm without the trials also diverged in six runs with −30 dB and in 51 runs with −20 dB. At each of these two levels, the proposed algorithm diverged in three more runs, within 2 s after a transfer. These three runs were the three conditions of one noise sequence, and Section 8.2 examines their transfers.

However, the advantage in the recovery time remained. With −20 dB, the median recovery time over the runs without divergence was 0.44 s and 0.51 s for the proposed algorithm, compared with 1.03 s and 1.33 s for CohVSS-FxNLMS. Counted over all 200 runs of each source-switch condition, including the diverged runs, the proposed algorithm recovered within 0.5 s in 98 and 90 runs with −20 dB, and CohVSS-FxNLMS in 38 and 31. Within 2 s, the counts were 135 and 128 for the proposed algorithm and 136 and 130 for CohVSS-FxNLMS. With −30 dB, the proposed algorithm recovered within 0.5 s in 110 and 105 runs, and CohVSS-FxNLMS in 42 and 40. With −30 dB and without a change, the noise reduction of the proposed algorithm over the last 10 s was smaller than that of CohVSS-FxNLMS by 0.3 dB ([0.06, 0.78] dB).

### 8.2 Check of the transfer decisions

The decisions based on the reconstructed errors agreed with those based on the actual errors, and every reversed transfer violated the sufficient condition of Proposition 3. For all 364,970 decisions, we computed $\rho^{\star}_n$, $\kappa_n$, and $\eta_n$ with the true secondary path. Table 9 summarizes the result. We call a transfer reversed when $\rho^{\star}_n \ge 1$. The median of $\eta_n$ was about 0.009, and its 99th percentile was 0.19 to 0.32. In 195 decisions (0.05%), $\eta_n \ge 1$ and Proposition 3 does not apply. None of them led to a transfer. The bound of Proposition 3 held in all other decisions. With $\eta_n = 0$, the bound failed in six decisions, in which the foreground coefficients had changed within the memory of the secondary path.

Without an added model error, using $\rho^{\star}_n$ instead of $\rho_n$ changed the outcome of 0.03% to 0.18% of the decisions in each condition. One of the 321 transfers after a change was reversed, for machine gun noise with $\kappa_n = 1.5$. The transfers also selected better coefficients for the following data. We froze the transferred coefficients and the coefficients they replaced over the following 2 s. The transferred coefficients gave a power ratio 6.0 dB to 6.2 dB lower in median, and none of them was worse. The 13 transfers with the 0 dB disturbance improved the coefficients by 0.76 dB in median, because the disturbance had degraded the foreground filter.

In contrast, the agreement decreased with an added error of −20 dB. Of the transfers after a change, 17 of 311 were reversed, and 19 of the 23 transfers without a change were reversed. Over all model errors, none of the 841 transfers that met the sufficient condition was reversed, and every reversed transfer had $\kappa_n > 0.9$. All six runs in which only the proposed algorithm diverged contained a reversed transfer before the divergence, with $\rho_n$ between 0.14 and 0.19 and $\rho^{\star}_n$ between 3.5 and 4.6.

## 9. Conclusion

We proposed a frequency-domain filtered-x algorithm for impulsive noise. The algorithm tests a larger step size on a background filter during a short trial and transfers the result to the foreground filter only when it reduces the reconstructed error. A discarded trial leaves the control signal unchanged. In simulations with ten noise types, the trials shortened the median recovery time after a switch of the noise source by about two thirds compared with the fastest of eleven comparison algorithms, without divergence for the secondary-path model of the main evaluation. None of the tested step sizes of three comparison algorithms gave this recovery time without divergence.

The algorithm has three limitations. First, the trial depends on the detector. Machine gun noise consists of intermittent bursts. For this noise, a transfer occurred in fewer than half of the runs, and our previous algorithm reduced the noise more after the switch. Second, the foreground filter recovered less from a disturbance at the error microphone than two comparison algorithms, by 0.3 dB to 0.6 dB. Third, with an added error of −30 dB or more in the secondary-path model, the algorithm diverged more often than our previous algorithm, and the decision transferred worse coefficients in some trials. The evaluation used simulations with paths from one room, a fixed secondary path, and no acoustic feedback. Future work includes a start condition for intermittent noise, an update that is less sensitive to errors in the secondary-path model, and a real-time implementation.

The code of the proposed algorithm and of the evaluation is available at https://github.com/YosukeSugiura/Active_Noise_Control_for_Impulsive_noise.

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
