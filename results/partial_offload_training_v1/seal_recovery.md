# H137 CPU-only sealing recovery

prepare_audit exited3221225477 inside the standard compact JSON encoder while
assembling result.json. No aggregate or audit manifest was written. All sixteen
training cases completed with exit0 and their files remain unchanged; no GPU
process remained. Permit one CPU-only recovery: stream the already serialized
case JSON files into an aggregate array, preserving their numerical text exactly.
Use the normal compact writer only for the small hash manifest. Preserve the
original source/log/exit. No optimizer update, backward or evaluation is repeated.
