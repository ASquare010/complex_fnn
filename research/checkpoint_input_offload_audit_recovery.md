# H121 audit startup recovery

The first independent audit exited with a native Windows access violation
while importing SymPy in `common.py:3`. The traceback precedes the PyTorch
import, model construction and `audit.run()`. No `replay.pt`, `replay.json` or
aggregate audit output was created. All 248 completed full-model probe
backwards and six qualification backwards remain frozen and are not repeated.

Preserve the original audit log/exit and all 16 saved gradients. Freeze this
note, a new wrapper and the original evidence before launching a fresh
process. The wrapper imports and calls the **unchanged** original `audit.run`.
There is no claimed fix for the recurrent native import problem. Complete
only the eight previously allocated replay backwards. No tolerance, model,
training recipe, sampling, compute allocation or scientific source changes.
Total if successful remains262 backwards, zero optimizer updates and24 new
gradient artifacts. Do not overwrite or hide the failed startup.
