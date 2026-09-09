# Testing the remaining parameter allowance - frozen before runs

The target permits 70% FFN reduction, not necessarily exactly 75%. Strict shared
SwiGLU has the strongest compressed 800-step result so far but remains 2.45%
above full SwiGLU. Scalar affine specialization changed NLL by only .00099 and
added runtime; do not pursue that branch.

Test strict shared SwiGLU at h=608, the largest multiple of 32 under the 70%
reduction cap at d=192,L=4. Unique FFN weights=3*192*608=350208: 70.3125% below
the original 1179648. Full model=1728192. This preserves the target rather than
changing the acceptance threshold.

Controls at 800 steps, seed 17:
1. shared_swiglu, h=608;
2. swiglu_narrow, h=152: identical unique FFN parameters, fewer operations;
3. swiglu, h=608: identical forward matrix FLOPs, more parameters.
Existing full SwiGLU h=512 remains the primary conventional reference.

The candidate spends 18.75% more FFN matrix FLOPs than h=512, about 8.04% more
total model matrix FLOPs under the recorded estimate. This is explicitly a
parameter/compute tradeoff. Report the compute-matched h=608 reference even if
it weakens the candidate's case. No extra tokens, altered attention or tokenizer.

If quality still misses the target, do not keep increasing width past the cap.
If it meets the early target, freeze this configuration and repeat seeds 29,43,
then test more data and scale before any acceptance claim. Sharing itself is
prior art and cannot be called a novel breakthrough.


Completed NLL: shared h608 3.137058; narrow h152 3.149475; full h608 3.051180.
The candidate misses the unchanged full h512 reference's 1% quality limit.
Do not promote widening or expand past the parameter cap. See the
[completed comparison](second_screen_report.md).
