# Draft v0.1 (abstract, introduction, Section 6 opening)

Style: Sugiura's style guide (Notion) and the additional rules of 2026-11-22. No semicolons. One main claim per sentence.
Numbers follow THEORY_frozen.md (8th edition), RESULTS_final2.md and RESULTS_coef2.md.
Bracketed notes [..] are for the authors and will be removed.

## Abstract

Feedforward active noise control (ANC) of impulsive noise has two requirements. The controller must remain stable under large bursts in the input. It must also return to a low residual level after the acoustic paths change. Existing update rules for impulsive noise address the first requirement. However, their behavior after a sudden change in the primary path is rarely evaluated. To address this problem, we propose a frequency-domain filtered-x algorithm with a background restart. When a detector indicates a possible change, a background filter starts adaptation from the foreground coefficients. The background filter is updated during the trial. Its coefficients are transferred to the foreground only after the background filter gives a lower error than the foreground. A discarded trial leaves the foreground output unchanged under the same initial state and input. The update uses a spectral-factor direction and a step normalized by the input scale of each block. After start-up and after each detected change, the update uses an error frame recomputed with the current coefficients for a short period. We derive the mean behavior of the direction for finite block lengths. We also derive a stability bound for the update with the recomputed error. We evaluated the method with measured room impulse responses, ten noise types, and eight comparison methods. Following a sudden change in the primary path, the proposed method reduced the median recovery time from 0.98 s to 0.36 s and from 1.26 s to 0.40 s in two source-switching conditions. These values are compared with the best comparison method. For machinegun noise, the comparison method recovered faster in both conditions. Without a path change, the noise reduction over 30 s differed from that of the best comparison method by −0.05 dB (95% CI [−0.08, −0.03]).

[About 230 words. Applied Acoustics allows 250 words. Check the ATSIP limit.]

## 1. Introduction

Impulsive noise contains short bursts with amplitudes far above the background level. Hammering, pile driving, valves, and firearms produce such noise. In feedforward ANC with a filtered-x least-mean-square (FxLMS) update, a single burst can produce a large coefficient step. This step can drive the controller unstable. To limit this effect, earlier studies clip the reference or error signal [refs], compress the error with a logarithmic transform [refs], or replace the squared error with an M-estimator cost [refs]. Other studies normalize the step by a robust estimate of the input power [refs].

These update rules are evaluated mostly with fixed acoustic paths. In practice, the primary path changes when the noise source moves or when another source becomes dominant. After such a change, the controller must move from one solution to another. A step size chosen for stability under impulsive input is small. Thus, the controller returns to a low residual level slowly. [We hypothesize that this trade-off explains why reconvergence is rarely evaluated. Check the original papers before keeping this sentence.]

This paper addresses reconvergence after a sudden change in the primary path under impulsive input. The proposed method keeps the steady-state update small. In addition, it runs a second adaptation in the background when a change is likely. A detector compares the error power with the reference power. When the detector triggers, a background filter starts from the foreground coefficients. The background filter adapts with a larger step. Its error frame is recomputed with its own coefficients. The background filter is updated during the trial. Its coefficients are transferred to the foreground only after the background filter gives a lower recomputed error than the foreground. If the trial is discarded, the foreground output is the same as the output of the method without restart. This identity holds under the same initial state and input. In contrast, an incorrectly confirmed transfer can change the output. We report how often such transfers occurred.

The contributions are as follows.

1. We normalize the update by the input scale of each block. We show that the expected update of the spectral-factor direction, given the scale sequence, is a quadratic form in the normalized input. We also show that an isolated impulse gives an update without gradient noise after this normalization. Thus, the method does not clip or transform the signals.
2. We recompute the error frame with the current coefficients only for a short period after start-up and after each detected change. We derive a stability bound for this update. We choose the step from this bound.
3. We restart adaptation as a background trial. The transfer rule compares the recomputed errors of the two filters. We show that a discarded trial does not change the foreground output.

The spectral-factor direction [Elliott and Rafaely], the recomputed-error update [modified FxLMS refs], and the two-filter structure [Ochiai 1977; echo cancellation refs] are known. This paper combines them for reconvergence under impulsive input. We analyze the combination for finite block lengths.

We evaluated the method with measured room impulse responses (MeshRIR [ref]). The inputs were four synthetic and six recorded noise types. The eight comparison methods used step sizes selected on separate seeds. Section 2 reviews related work. Section 3 states the problem. Section 4 describes the method and its analysis. Section 5 describes the evaluation. Section 6 reports reconvergence after a primary path change. Section 7 reports the cost of the restart mechanism without a path change.

## 6. Reconvergence after a primary path change (opening paragraphs)

We changed the primary path 15 s after start-up. To do so, we switched the noise source to another loudspeaker position near or far from the original position. The secondary path and its model did not change. The recovery time is the time from the change until the short-time error power stays within 3 dB of a reference level. The reference level is the level that the same method reaches when it runs on the post-change paths from the start.

Following the change, the proposed method reduced the median recovery time in both source-switching conditions (Table 2). The median was 0.36 s for the near switch and 0.40 s for the far switch. Compared with these values, the best comparison method gave 0.98 s and 1.26 s. This method is the frequency-domain method with coherence-based step control [our APSIPA 2026 paper]. The 95% confidence intervals of the paired differences excluded zero in both conditions.

The recovery time uses a reference level specific to each method. Thus, we also evaluated the noise reduction with frozen coefficients. We froze the coefficients 0.13 s and 0.51 s after the change. We then applied them to the post-change signal, excluding the first 256 ms. For the far switch, the median over all inputs was −6.7 dB and −10.9 dB for the proposed method. Compared with these values, the comparison method gave −0.9 dB and −6.8 dB. At both time points and in both conditions, the 95% confidence intervals of the paired differences were below zero. We also compared the methods for each input, condition, and time point. In these 40 comparisons, the proposed method was better in 31 and worse in 6. The confidence interval included zero in the other 3 (Fig. 4).

For machinegun noise, the comparison method was better in both conditions and at both time points. For car interior noise (volvo), the comparison method was better after the near switch. In most of these runs, the detector did not lead to a confirmed transfer. In such runs, the proposed method behaved as the method without restart. [Give the per-input transfer counts here.]

[Notes for the authors]
- "Better" and "worse" mean that the 95% confidence interval of the paired difference excludes zero. State this criterion once in Section 5.
- "Significant", "robust", "novel", and "equivalent" are not used.
- "We show" is used only for results proved in Section 4 or the appendix. Experimental results use "reduced" or "we observed".
- "Recovery time" (short-time error, method-specific level) and "noise reduction with frozen coefficients" are kept as separate terms.
