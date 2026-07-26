"""Smoke-check the public article HTML."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
html_path = ROOT / "index.html"
html = html_path.read_text(encoding="utf-8")

checks = [
    "Diet soda did not give you cancer",
    "If you only read one block",
    "What I learned",
    "Grok 4.5",
    "I built",
    "First super prompt",
    "twice as diabetic",
    "cancer_c1_hazard_vs_risk.png",
    "cancer_c5_cox_forest.png",
    "myth_m5_smd_loveplot.png",
    "myth_m2_hba1c_violin.png",
    "outputs/figures/",
    "I am not claiming",
    "Why I bothered with models",
    "What held",
    "The ladder",
    "How to read this",
    "Scoreboard",
    "Not medical advice",
]
failed: list[str] = []
for c in checks:
    ok = c in html
    print(("OK" if ok else "MISS"), c)
    if not ok:
        failed.append(c)

if "\u2014" in html:
    print("MISS em dash present")
    failed.append("emdash")
else:
    print("OK no em dash")

if "[Missing figure" in html:
    print("MISS has missing figure placeholders")
    failed.append("missing figures")

imgs = re.findall(r'src="([^"]+)"', html)
print("img_count", len(imgs))
print("size_kb", html_path.stat().st_size // 1024)
missing = [s for s in imgs if s.startswith("outputs/") and not (ROOT / s).exists()]
print("missing_files", missing)
if missing:
    failed.extend(missing)

sys.exit(1 if failed else 0)
