# H068 recovery - unchanged fitting screen after interrupted launch

The first launch passed four harness checks, generated its data and entered
plain_teacher/plain/seed17/rate.001. The tool returned `aborted` during a goal
continuation. Two subsequent OS inspections found neither worker PID 13892 nor
the launcher. There are zero completed training/checkpoint/report files, an
empty fitting log, and one empty cell directory. The original RUNNING record is
preserved with a separate [interruption record](../results/token_activation_fit_v1/interruption.json).
Exit code and number of executed updates are unknown: between zero and 300.
This is incomplete execution, not evidence of a model failure or success.

The [original plan](token_activation_fit_plan.md) forbids automatic retries.
This explicit, one-time recovery exception is based on verified process absence
and zero observed endpoints, not a scientific outcome. It changes no definition,
optimizer, seed, rate, data, gate or budget. Preserve the entire first attempt.
Start one fresh complete 252-cell attempt under results/token_activation_fit_v2,
using the original frozen study module with only its output ROOT overridden.
Pin the original source, plan, four-test result, data hash and interruption record.
The generated tensor data and sampler hashes must match the original attempt.
No additional recovery follows automatically if this attempt fails.

Use a hidden detached coordinator so a tool observation interruption does not
kill the GPU worker. Record PID, command, timestamps, logs, return code and source
fidelity. One GPU worker at a time. Do not repeat the unchanged harness tests.
A successful recovery has 75,600 known completed updates and 19,353,600 training
example presentations; total work including the incomplete attempt is bounded by
75,600..75,900 updates and 19,353,600..19,430,400 example presentations.
All original scientific gates and interpretation limits remain unchanged.
