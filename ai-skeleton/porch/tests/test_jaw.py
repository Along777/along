import numpy as np

from skeleton_porch.config import Jaw
from skeleton_porch.jaw import RELAX, JawDriver

CFG = Jaw(closed_deg=90, open_deg=60, delay_s=0.2, gate_db=-45, full_db=-18, relax_after_s=0.4)


def block(dbfs: float, n: int = 480) -> np.ndarray:
    return np.full(n, 10 ** (dbfs / 20), np.float32)


def run(driver: JawDriver, t0: float, t1: float, dt: float = 0.025) -> list:
    out, t = [], t0
    while t < t1:
        out.append((t, driver.tick(t)))
        t += dt
    return out


def test_silent_jaw_stays_relaxed():
    d = JawDriver(CFG)
    assert all(v is None for _, v in run(d, 0, 2))


def test_loud_voice_opens_after_the_delay_then_relaxes():
    d = JawDriver(CFG)
    for i in range(50):  # 1 s of loud speech heard from t=1.0 to t=2.0
        d.feed(1.0 + i * 0.02, block(-15))
    d.feed(2.0, block(-90))  # then silence
    ticks = dict(run(d, 0.5, 3.0))
    before = [v for t, v in ticks.items() if t < 1.15]
    assert all(v is None for v in before)  # nothing before the voice (+ delay) arrives
    during = [v for t, v in ticks.items() if 1.5 < t < 2.1]
    assert during and min(during) <= 62  # wide open: close to open_deg=60
    after = [v for t, v in ticks.items() if t > 2.8]
    assert RELAX in [v for t, v in ticks.items() if 2.2 < t <= 2.8]  # lets go once speech is over
    assert all(v is None for v in after)


def test_quiet_voice_opens_partly():
    d = JawDriver(CFG)
    for i in range(50):
        d.feed(i * 0.02, block(-31.5))  # halfway between gate and full
    angles = [v for t, v in run(d, 0.5, 1.0) if isinstance(v, int) and v != RELAX]
    assert angles and 70 <= angles[-1] <= 80  # about half open between 90 and 60


def test_delay_lines_up_with_bluetooth():
    d = JawDriver(CFG)
    d.feed(1.0, block(-15))
    d.feed(1.02, block(-90))
    assert d.level_at(1.01) > -20 and d.level_at(0.99) == -120.0
