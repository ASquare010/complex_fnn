# H152: can a learned readout make reversible geometry useful?

Previous turn: progress. H151 proves the fixed-tail obstruction; a learned outer
linear map evades it. Test learning before further timing refinement. This is a
small falsification pilot, not resource promotion of H149/H150, whose failures
remain unchanged. No large language-model run is authorized by this screen.

Dimension32, four layers. Reversible core uses existing fused learned shape,
native fixed shape, or native affine control; add a32x32 readout and bias.
Initialize core biases zero, retain matched nonzero theta/Q initialization,
readout weight=A^T where A=Q4...Q1, readout bias zero. This approximately cancels
fixed rotation, not nonlinear shape. Learned/fixed shape start identically.
At theta=bias=0, W=M*A^-1 exactly represents any linear target; verify separately.
Learned/affine have1312 parameters, fixed1184, plus16384B of mixing matrices.

Controls: four residual FFN blocks with ReLU, LeakyReLU(.01), PReLU(one scalar
per block), GELU, SiLU, and SwiGLU. Ordinary hidden32, SwiGLU hidden21; matched
seeded initialization across ungated activations. Budget GELU hidden4 (1168
parameters) and a plain linear map (1056) are additional controls. Eleven arms.
No activation's superiority is assumed; all get the same learning-rate search.

Three tasks: random orthogonal linear target, fixed wide GELU teacher (hidden128),
and cyclic pair product mixed by an orthogonal map. Fixed target functions across
seeds; independent input/model seeds263/277/293. Samples8192 per task/seed:
train4096, validation2048, reporting2048. Target centering/global RMS are fit on
training only. Inputs Gaussian; no augmentation. Same300x128 sampled training
indices for every arm/rate within a dataset. AdamW betas(.9,.95), decay0,clip1,
learning rates.001/.003,300 updates, FP32/TF32off, four CPU threads. Exactly
11arms x2rates x3tasks x3seeds=198fits,59400updates/backwards. One model resident,
199 clean GPU boundaries. No adaptive extension, failed-fit substitution or retries.

Select one rate per arm/task by mean validation MSE across seeds; reporting split
never selects rates. Report all runs and selected mean/median/sample variance,
paired seed ratios, parameter/buffer counts, fit wall time and allocated peaks.
These small-model timings/peaks are diagnostic, not a production VRAM benchmark.

Screen gates: full GELU and full SwiGLU must each reduce reporting MSE versus zero
by at least20% on every task/seed (otherwise that task is inconclusive). On every
qualified task/seed learned readout must be <=1.01 times BOTH full controls,
<=1.01 times budget GELU, and on nonlinear tasks improve both fixed-shape and
affine controls by at least5%. All tasks must qualify/pass for promotion. Any
failure closes only this initialization/optimizer/budget recipe, not all readouts.

Preflight: exact CPU FP64 linear-target witness and GPU readout/input/shape/bias
gradients versus CPU FP64 native forward autograd, relative<=1e-4 per family.
One GPU backward and one CPU backward, no optimizer updates. Stop if this fails.
Audit all saved states/counter300/dataset hashes, independently score validation
and reporting splits using explicit CPU formulas and tolerance1e-5. Audit no
backwards/GPU. Freeze all sources first; preserve prior receipts and defaults.
