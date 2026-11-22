# Draft v0 (abstract, introduction, Section 6 opening)

Numbers follow THEORY_frozen.md (8th edition), RESULTS_final2.md and RESULTS_coef2.md.
Bracketed notes [..] are for the authors and will be removed.

## Abstract

Feedforward active noise control (ANC) of impulsive noise must keep the controller stable under large input outliers, and it must also return to a low residual level after the acoustic paths change. Existing update rules for impulsive noise address the first requirement; their behavior after a sudden change in the primary path is rarely evaluated. We propose a frequency-domain filtered-x algorithm that restarts adaptation in a background filter when a change is detected. The background filter is updated during the trial. Its coefficients are transferred to the foreground only after the background filter yields a lower error than the foreground on the same block. A discarded trial leaves the foreground output unchanged under the same initial state and input. The update uses a spectral-factor direction, a step normalized by the input scale of each block, and, for a short period after start-up and after each detected change, an error frame recomputed with the current coefficients. We derive the mean behavior of the direction for finite block lengths and a stability bound for the recomputed-error update. In simulations with measured room impulse responses, four synthetic and six recorded noise types, and eight comparison methods, the proposed method reduced the median recovery time after a sudden change in the primary path from 0.98 s to 0.36 s and from 1.26 s to 0.40 s in two source-switching conditions, relative to the best comparison method. For machinegun noise, the comparison method recovered faster in both conditions. Without a path change, the noise reduction over 30 s differed from that of the best comparison method by −0.05 dB (95% CI [−0.08, −0.03]).

[≈200 words. Applied Acoustics limits the abstract to 250 words; check ATSIP.]

## 1. Introduction

Impulsive noise, such as noise from hammering, pile driving, valves, and firearms, contains short bursts whose amplitude can exceed the background level by tens of decibels. In feedforward ANC with a filtered-x least-mean-square (FxLMS) type update, a single burst can produce a large coefficient step and drive the controller unstable. Many update rules have been proposed to limit this effect: clipping the reference or error signal [refs], applying a logarithmic or other compressive transform to the error [refs], replacing the squared error with an M-estimator cost [refs], and normalizing the step by a robust estimate of the input power [refs].

These update rules are evaluated mostly with fixed acoustic paths. In practice, the primary path changes when the noise source moves or when a different source becomes dominant. After such a change, the controller has to move from one solution to another. A step size chosen to keep the controller stable under impulsive input is small, so the controller returns to a low residual level slowly. [We hypothesize that this trade-off is the reason why the evaluation of reconvergence is rare. Confirm against the original papers before keeping this sentence.]

This paper addresses reconvergence after a sudden change in the primary path under impulsive input. We keep the update small in the steady state and add a restart mechanism that runs a second adaptation in the background. When a detector indicates a possible change, a background filter starts from the foreground coefficients and adapts with a larger step on an error frame recomputed with its own coefficients. The background filter is updated during the trial. Its coefficients are transferred to the foreground only after the background filter yields a lower recomputed error than the foreground. If the trial is discarded, the foreground output is the same as that of the method without restart, under the same initial state and input. An incorrectly confirmed transfer can change the output; we report how often this occurred.

The contributions are as follows.

1. We condition the update on the input scale of each block. We show that, for a spectral-factor direction, the expected update given the scale sequence is a quadratic form in the normalized input, and that an isolated impulse produces an update without gradient noise after the scale normalization. The method therefore does not clip or transform the signals.
2. We recompute the error frame with the current coefficients only during a short period after start-up and after each detected change, and we choose the step from a stability bound that we derive for this update.
3. We restart adaptation as a background trial with a transfer rule based on the recomputed error. We show that a discarded trial does not change the foreground output.

The spectral-factor direction [Elliott and Rafaely], the recomputed-error update [modified FxLMS refs], and the two-filter structure [Ochiai 1977; echo cancellation refs] are known. This paper combines them for reconvergence under impulsive input and analyzes the combination for finite block lengths.

We evaluated the method with measured room impulse responses (MeshRIR [ref]), four synthetic and six recorded impulsive or intermittent noise types, and eight comparison methods whose step sizes were selected on separate seeds. Section 2 reviews related work. Section 3 states the problem. Section 4 describes the method and its analysis. Section 5 describes the evaluation. Section 6 reports reconvergence after a primary path change, and Section 7 reports the cost of the restart mechanism in conditions without a path change.

## 6. Reconvergence after a primary path change (opening paragraphs)

We changed the primary path 15 s after start-up by switching the noise source to another loudspeaker position (near or far from the original position). The secondary path and the secondary path model did not change. The recovery time is the time after the change until the short-time error power stays within 3 dB of the level that the same method reaches when it runs on the post-change paths from the start.

Following the change, the proposed method reduced the median recovery time in both source-switching conditions (Table 2): 0.36 s versus 0.98 s for the near switch and 0.40 s versus 1.26 s for the far switch, compared with the frequency-domain method with coherence-based step control [our APSIPA 2026 paper], which was the best comparison method. The 95% confidence intervals of the paired differences excluded zero in both conditions.

Because the recovery time uses a method-specific reference level, we also evaluated the noise reduction obtained with frozen coefficients. We froze the coefficients 0.13 s and 0.51 s after the change and applied them to the post-change signal, excluding the first 256 ms. Over all inputs, the proposed method gave −6.7 dB and −10.9 dB for the far switch, against −0.9 dB and −6.8 dB for the comparison method; the 95% confidence intervals of the paired differences were below zero at both time points and in both conditions. Across the 40 combinations of input, condition, and time point, the proposed method gave a lower value in 31, the comparison method in 6, and the confidence interval included zero in 3 (Fig. 4).

For machinegun noise, the comparison method gave lower values in both conditions and at both time points. For car interior noise (volvo), it gave lower values after the near switch. In these cases the detector did not trigger a confirmed transfer in most runs, and the proposed method then behaved as the method without restart. [Give the per-input transfer counts here.]

[Notes for the authors]
- "significant", "robust", "novel" are not used. "Equivalent" is not used for the steady-state comparison; the text gives the difference and its interval.
- "We show" is used only for results proved in Section 4 or the appendix; experimental results use "the method reduced", "we observed".
- Keep "recovery time" (short-time error, method-specific level) and "noise reduction with frozen coefficients" as separate terms throughout.
