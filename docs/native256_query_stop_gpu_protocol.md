# Full-size STOP objective preflight

Run only after the six CPU model/objective tests and three new combined-objective
tests pass. Preserve frozen STOP source/protocol/test bytes and all V2/native
files. No model/checkpoint migration or production training in this preflight.

New composition helper retains original global assistant CE+.05 query+.02 source
and adds .02 balanced original EOS/nonEOS CE to the final STOP head with detached
selected hidden states. Capture the existing selected CE rows to avoid another
transformer forward. Test exact coefficient0 previous-objective Adam parity,
auxiliary-only gradient routing, microbatch normalization and serialized resume.

Use genuine read-only original corpus/cache packets and full21-layer fresh47
STOP model. Compare CPU/GPU all-parameter clipped gradients(<2e-5 absolute), loss
components(<2e-5) and preclip norm(relative<2e-5), then GPU global/micro gradients.
Original norm audit motivates relative norm checking; do not weaken absolute
gradient/loss gates. Verify every initial tensor remains exact, optimizer empty.

Stress the actual B8/memory39/raw512 mixed packet with genuine auxiliary targets,
plus all-source packet padded with invalid memory slots. Check invalid padding
loss invariance. Retain two moment-sized buffers and one parameter-sized scratch
buffer; measure peak allocated/reserved. They approximate storage; no actual Adam
step/scratch execution is claimed.

Alternate STOP coefficient0/.02 on the SAME fresh STOP weights and packets;
record three forward/backward/clip body samples after warmup. This measures
auxiliary-objective overhead, NOT old-reader vs new-reader inference speed or
end-to-end training. Also alternate cached token decode between the old V2 and
new STOP model with separate own fresh initializations, identical actual input,
capacity/batch/phase. Require identical cache byte counts/attributes. Different
fresh architectures are not identical trained outputs; no quality claim from
timing. One GPU worker, FP32/noTF32, no reserved student TEST.

Bind source/CPU receipts and prior comparison protected guards before and after
the zero-update run. Preserve failure evidence and use a new output/version for
any repair. A passing gate permits preparation of a NEW bounded paired training
pilot; it does not establish retrieval quality or final fresh two-epoch parity.
