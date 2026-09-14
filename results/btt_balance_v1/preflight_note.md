# H164 pre-protocol execution note

The first CPU numerical-check command returned exit1 with no Python diagnostic
captured. Its cause is unknown. No check.json or GPU training existed afterward.
One bounded retry of unchanged check.py, with faulthandler and a fresh unused
pycache prefix, completed successfully. The second command's full log and exit0
are retained. Do not attribute the original failure to a proven import/root cause.

Both normalized BTT variants match explicit dense multiplication and pass CPU
float64 gradcheck. Parameter counts match the plan. No model learning or GPU
profiling occurred before the protocol was frozen.
