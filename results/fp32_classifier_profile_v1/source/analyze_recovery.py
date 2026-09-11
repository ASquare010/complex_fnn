"""Rerun unchanged postprocessing with a stack watchdog after an unexplained stall."""

import faulthandler
import runpy

faulthandler.dump_traceback_later(10, repeat=True)
try:
    runpy.run_module("results.fp32_classifier_profile_v1.source.analyze", run_name="__main__")
finally:
    faulthandler.cancel_dump_traceback_later()
