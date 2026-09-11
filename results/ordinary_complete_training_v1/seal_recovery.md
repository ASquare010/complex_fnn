# CPU-only audit-seal recovery

The audit-preparation process exited3221225477 inside Python's indented JSON
encoder. Neither result.json nor audit_protocol.json was written. All twelve
scientific runs remain complete. Permit one CPU-only retry using compact JSON
encoding (indent=None), which selects the standard accelerated encoder without
changing numerical values or audit content. Preserve the failed source/log.
No GPU run, optimizer update or evaluation is repeated.
