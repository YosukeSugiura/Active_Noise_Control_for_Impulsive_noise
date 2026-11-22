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

Adaptive algorithms for active control of impulsive noise limit the influence of large samples in the reference and error signals. This limitation keeps the control filter stable under impulsive noise. However, a sudden change in the primary path also increases the error, and the error stays large until the control filter reconverges. The same limitation reduces the coefficient update during this period, and the reconvergence becomes slow. This paper proposes a frequency-domain filtered-x algorithm that accelerates adaptation only when a trial adaptation reduces the error. The algorithm uses two filters. The foreground filter generates the control signal, and the background filter is used for the trial. When the ratio of the error power to the filtered-reference power exceeds its running average, the background filter is initialized with the foreground coefficients and adapted with a larger step size. Its coefficients are transferred to the foreground filter only when they give a sufficiently smaller error on the same data. Otherwise, the trial is discarded, and the control signal is identical to that obtained without the trial under the same initial state and input. To make the background filter converge quickly, the update uses a spectral-factor direction and a step size normalized by the input power of each block. In addition, the error frames are recomputed with the current coefficients after start-up and after each transfer. We evaluated the algorithm in simulations with measured room impulse responses and ten noise types. Compared with the best of eight existing algorithms, the proposed algorithm reduced the median recovery time after a switch of the noise source by 63% for a nearby new source and by 68% for a distant new source. For machinegun noise in both conditions and for car interior noise with a nearby new source, the existing algorithm recovered faster. Without a path change, the noise reduction of the two algorithms differed by less than 0.1 dB.

[約 265 語。63% = 1 − 0.36/0.98、68% = 1 − 0.40/1.26。0.1 dB 未満：30 秒全体 0.05 dB、末尾 10 秒 0.07 dB。]

## 1. Introduction

Impulsive noise contains short bursts whose amplitudes are much larger than the background level. Its amplitude distribution has heavier tails than a Gaussian distribution and is often modeled by a symmetric α-stable distribution [Shao and Nikias 1993]. In feedforward active noise control (ANC), the filtered-x least-mean-square (FxLMS) algorithm updates the control filter in proportion to the product of the filtered reference signal and the error signal [Kuo and Morgan 1996]. An impulsive sample in either signal therefore produces a large update, and such an update can make the control filter diverge.

To prevent this divergence, robust variants of the FxLMS algorithm limit the influence of large samples in the reference and error signals. Sun et al. [Sun 2006] ignored the samples that exceeded a threshold. Akhtar and Mitsuhashi [Akhtar 2009] replaced such samples in the reference and error signals with the threshold values. Wu et al. [Wu 2011] minimized the squared logarithm of the error. This cost grows slowly for large errors. Other studies replaced the squared error with an M-estimator [Wu and Qiu 2013, Sun et al. 2015] or with a fractional lower-order statistic [Feng et al. 2024]. These variants keep the control filter stable under impulsive noise. Section 2 reviews them in more detail.

These variants reduce the update for large samples regardless of their cause. A sudden change in the primary path, such as a switch to another noise source, also increases the error. The error then stays large until the control filter converges to the new optimal coefficients. The robust variants therefore reduce the update when a large update is needed, and the reconvergence is slow. In our evaluation with measured room impulse responses, the best of eight existing algorithms needed about 1 s in median to recover after a switch of the noise source (Section 6).

A larger step size would speed up the reconvergence. However, an impulse also increases the error power, and a larger step size during an impulse can degrade the control filter or make it diverge. A detector that monitors the error power responds to both events. We therefore use the detector only to start a trial adaptation, and we accept the result of the trial only when it reduces the error.

The proposed algorithm uses two filters with the same structure. The foreground filter generates the control signal and is updated with the normal step size at all times. When the detector responds, the background filter is initialized with the foreground coefficients and is updated with a larger step size. Its coefficients are transferred to the foreground filter only after the error computed with the background coefficients becomes sufficiently smaller than the error computed with the foreground coefficients. Both errors are computed from the measured reference and error signals by using the model of the secondary path. The trial is discarded if this condition is not met within a fixed number of blocks. Because the background filter does not contribute to the control signal, a discarded trial leaves the control signal unchanged, given the same initial state and input. In contrast, a transfer that is confirmed without a path change can change the control signal. We therefore report how often such transfers occurred.

The trial is useful only if the background filter converges quickly. For this purpose, the proposed algorithm combines two elements, and we analyze them for finite block lengths. First, the update direction is based on the spectral factor of the reference power spectrum [Elliott and Rafaely 2000], and the step size is divided by the input power of each block. We show that, after this division, the update caused by an isolated impulse does not depend on the amplitude of the impulse. The algorithm therefore uses impulsive samples without clipping them. Second, after start-up and after each transfer, the error frame is recomputed with the current coefficients, as in the modified FxLMS algorithm [ref]. We derive the form of this update and determine the range of stable step sizes numerically. The structure with foreground and background filters was proposed for echo cancellation [Ochiai et al. 1977]. The contribution of this paper lies in its use for the reconvergence of ANC under impulsive noise and in the analysis and evaluation of the combined algorithm.

We evaluated the proposed algorithm in closed-loop simulations with room impulse responses measured in one room [MeshRIR] and with ten noise types (four synthetic signals and six recordings). Compared with the best of eight existing algorithms, the proposed algorithm reduced the median recovery time after a switch of the noise source from 0.98 s to 0.36 s for a nearby new source and from 1.26 s to 0.40 s for a distant new source. For machinegun noise in both conditions and for car interior noise with a nearby new source, the existing algorithm recovered faster. Without a path change, the noise reduction of the two algorithms differed by less than 0.1 dB.

The remainder of this paper is organized as follows. Section 2 reviews related work, and Section 3 formulates the problem. Section 4 describes the proposed algorithm and its analysis. Section 5 describes the evaluation framework. Section 6 reports the reconvergence after a primary path change, and Section 7 reports the cost of the trial mechanism when the path does not change. Section 8 concludes the paper.

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
