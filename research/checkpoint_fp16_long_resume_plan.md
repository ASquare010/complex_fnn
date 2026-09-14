# H160 resume protocol — 2026-09-13

User explicitly resumed the goal. The previous cleanup was progress; research
remained paused until this authorization. All frozen H160 dependencies still hash
correctly. Preserve the original three completed runs and the interrupted case03.

Restart only interrupted case03 from its original step-zero source in
`results/checkpoint_fp16_long_v1/restart/case03`. This provides the prescribed
800 continuous updates; continuing step200 would splice timing across sessions.
Never overwrite or move original case03, its logs, step200 checkpoint or history.
Then run previously unstarted cases04-11 using the original worker and protocol.
No completed run is repeated. The original 207 recorded partial updates and its
initial gradient probe remain outside the accepted12-run comparison, with their
compute recorded separately. One unrecorded in-flight update cannot be excluded.

This adds7,200 updates (nine runs). Accepted experiment remains9,600 updates and
9,636 backwards including original preflight/audit; recorded total including the
interrupted attempt is at least9,807 updates and9,844 backwards. These are separate
accounting scopes, not claims that the discarded work never happened.

Original training loop, checkpoint codec, initialization, batch order, numerical
caps, memory/quality/timing gates and native audit stay unchanged. Recover only
case03's output/telemetry/exit-file lookup in new aggregation and analysis files.
Finalization verifies H159 README/current-state hashes against the preserved H160
before-copies because pause/cleanup/resume legitimately changed the live documents.
Keep the original source files and audit receipts unchanged.

All output paths are fresh and exclusive; stop on any failed process, never repeat
completed work automatically. Freeze new recovery scripts, this plan, original
protocol, completed cases and interrupted artifacts before execution. Require20GiB
free (expected additional output below4GiB). No raw-history regeneration after the
cleanup. Research is active, with bounded resource use.

WikiText2 seed113 has its FP16 run before the pause and its restarted native control
after the pause. Report that cross-session timing limitation even if every original
numeric gate passes; do not call it a temporally adjacent pairing or normalize
clocks. Other seed pairs remain from the same uninterrupted session. Quality and
memory checks still use matched states/data/configuration. A passing 800-update
study is not terminal convergence, unrelated-domain generality or a breakthrough.
