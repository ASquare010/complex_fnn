# Actual source-aware capacity input selection

18 October2026. The full-query numerical gate passed its objective/relative-norm/
all-gradient/micro checks, then stopped before capacity: there are no original
source-bearing raw512 segments with39 old blocks. Preserve that zero-update
failure. Maximum source and maximum general-language history are different cases.

Use one genuine original raw512/memory39 general-language segment plus seven
genuine source-bearing raw512 segments in the same batch. Production batching
pads shorter memory histories and preserves each row's original valid/visible
counts. This is a genuine mixed original-data packet with the maximum batch
attention shape and substantial auxiliary work.

Also stress eight genuine source-bearing raw512 rows, explicitly append invalid
zero memory columns up to39 and retain every original visible count/valid flag.
Compare its complete LM+query+source objective against the unpadded original
packet before measuring. Report this as invalid-padding capacity stress, not
eight genuinely39-block source histories. No target/example is optimized or
added to the training sampler. Both capacity checks must leave weights exact,
Adam empty and all gradients finite, with two moment-sized buffers. Record three
warmed body samples and peak allocation/reservation, excluding Adam step/scratch,
encoder, IO/eval/save. Neither result alone establishes end-to-end throughput.
