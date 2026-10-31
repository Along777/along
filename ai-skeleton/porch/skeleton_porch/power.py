"""Keep Windows awake while the porch runs, and report whether we are on mains power."""
from __future__ import annotations

import ctypes
import sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def keep_awake() -> bool:
    """Ask Windows not to sleep while this thread lives. Call from the main thread.

    This does not override closing the lid: set "When I close the lid" to "Do nothing"
    (plugged in), or keep the lid cracked open.
    """
    if sys.platform != "win32":
        return False
    return bool(ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))


def allow_sleep() -> None:
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


class _SystemPowerStatus(ctypes.Structure):
    _fields_ = [("ACLineStatus", ctypes.c_byte), ("BatteryFlag", ctypes.c_byte),
                ("BatteryLifePercent", ctypes.c_byte), ("SystemStatusFlag", ctypes.c_byte),
                ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]


def power_status() -> tuple[bool | None, int | None]:
    """(on AC power?, battery percent) -- None when unknown."""
    if sys.platform != "win32":
        return None, None
    s = _SystemPowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
        return None, None
    ac = {0: False, 1: True}.get(s.ACLineStatus & 0xFF)
    pct = s.BatteryLifePercent & 0xFF
    return ac, (None if pct == 255 else pct)


class _MemoryStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def ram_free_mb() -> int | None:
    """Free physical memory in MB (None when unknown)."""
    if sys.platform != "win32":
        return None
    m = _MemoryStatus()
    m.dwLength = ctypes.sizeof(_MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        return None
    return round(m.ullAvailPhys / 2**20)
