# H122: offline requested-payload reanalysis

All four GPU cases completed once (12 backwards). The original analyzer
failed its exact allocated-byte check. Inspection shows event `size` describes
requested bytes, whereas snapshot block `size` includes allocator rounding and
possibly an unsplit remainder. Also, `_snapshot()` returns the history before
its own snapshot event, so saved marker indices need +1 to identify that event.

Preserve the original plan/analyzer/log and reject its exact allocated-peak
attribution gate. A separate offline pass may reconstruct **requested payload**
using initial `requested_size`, alloc/free-request events and final active
`requested_size`. Do not relabel this as allocated VRAM or relax the old gate.
Verify exact requested endpoint accounting and marker-event correspondence;
show the gap to physical allocation explicitly. No GPU work is authorized by
this reanalysis; the completed 12 backwards are not repeated.

A first inspection also encountered a transient JSON integer-decoding error;
a fresh read of all four saved snapshots succeeded. Its failed log remains.
This does not establish the cause of the recurrent native/runtime problems.
