# H135 bounded import recovery

Cases00-06 completed (210 updates). Case07 failed importing torch.jit.frontend:
SystemError: Negative size passed to PyBytes_FromStringAndSize, exit1.
Its case directory has no scientific files and no GPU process remained.
The sequential driver stopped; cases08-11 were never launched.

Preserve all completed cases and original failure logs. Permit one new process
for unfinished case07, then cases08-11, in the original order with identical
worker source, PYTHONMALLOC=pymalloc, policy and configuration. Recovery logs
use a recovery_ prefix. Stop at any further failure. No completed scientific
work is repeated. This is operational recovery, not diagnosis or changed gates.

Automatic approval review could not complete the first recovery setup request.
Read-only inspection verified neither recovery source nor plan had been created;
no retry process started. A subsequent setup request is permitted.
