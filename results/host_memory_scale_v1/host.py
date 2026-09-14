"""Windows process memory counters, in bytes; no GPU or optional dependencies."""

import ctypes
import gc
import json
from ctypes import wintypes
from pathlib import Path


class Counters(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
        (name, ctypes.c_size_t)
        for name in (
            "PeakWorkingSetSize",
            "WorkingSetSize",
            "QuotaPeakPagedPoolUsage",
            "QuotaPagedPoolUsage",
            "QuotaPeakNonPagedPoolUsage",
            "QuotaNonPagedPoolUsage",
            "PagefileUsage",
            "PeakPagefileUsage",
            "PrivateUsage",
        )
    ]


kernel = ctypes.WinDLL("kernel32", use_last_error=True)
psapi = ctypes.WinDLL("psapi", use_last_error=True)
kernel.GetCurrentProcess.argtypes = []
kernel.GetCurrentProcess.restype = wintypes.HANDLE
psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
psapi.GetProcessMemoryInfo.restype = wintypes.BOOL


def snapshot():
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(
        kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    value = {name: int(getattr(counters, name)) for name, _ in Counters._fields_}
    assert value["cb"] == 80
    assert value["PeakWorkingSetSize"] >= value["WorkingSetSize"] > 0
    assert value["PeakPagefileUsage"] >= value["PrivateUsage"] > 0
    return value


if __name__ == "__main__":
    before = snapshot()
    buffer = ctypes.create_string_buffer(64 * 2**20)
    ctypes.memset(buffer, 1, len(buffer))
    during = snapshot()
    assert during["WorkingSetSize"] - before["WorkingSetSize"] >= 32 * 2**20
    assert during["PrivateUsage"] - before["PrivateUsage"] >= 32 * 2**20
    del buffer
    gc.collect()
    after = snapshot()
    assert after["PeakWorkingSetSize"] >= during["PeakWorkingSetSize"]
    assert after["PeakPagefileUsage"] >= during["PeakPagefileUsage"]
    output = dict(
        before=before,
        during=during,
        after=after,
        touched_allocation_bytes=64 * 2**20,
        passed=True,
        gpu_work=False,
    )
    with Path("results/host_memory_scale_v1/api_check.json").open("x") as stream:
        json.dump(output, stream, indent=2)
    print("Windows memory counter allocation test passed")
