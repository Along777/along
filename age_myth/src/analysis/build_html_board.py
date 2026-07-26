"""One-page HTML myth-bust board with claims + embedded figures."""
from __future__ import annotations

import base64
import json
from pathlib import Path

from src.analysis.ladder import BULLET
from src.paths import OUTPUTS, ensure_dirs

FIGS = OUTPUTS / "figures" / "definitive"
OUT = OUTPUTS / "reports" / "myth_bust_board.html"


def img_b64(path: Path) -> str:
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> None:
    ensure_dirs()
    claims = json.loads((BULLET / "claims.json").read_text(encoding="utf-8"))
    a = claims["claim_A"]
    a2 = claims["claim_A2"]
    b = claims["claim_B"]["delta_e65"]
    c = claims["claim_C"]
    gates = claims["gates"]

    def fig(name: str, caption: str) -> str:
        p = FIGS / name
        b64 = img_b64(p)
        if not b64:
            return f"<p><em>Missing {name}</em></p>"
        return f'''
        <figure>
          <img src="data:image/png;base64,{b64}" alt="{name}" style="max-width:100%;height:auto;border:1px solid #ddd;border-radius:6px;"/>
          <figcaption style="font-size:0.9rem;color:#444;margin-top:0.4rem;">{caption}</figcaption>
        </figure>'''

    a15 = a2.get("ages", {}).get("15", {})
    a30 = a2.get("ages", {}).get("30", {})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Age Myth Bust — Bulletproof Board</title>
<style>
  body {{ font-family: system-ui, Segoe UI, sans-serif; margin: 0; background: #0f1419; color: #e7ecf1; }}
  .wrap {{ max-width: 1100px; margin: 0 auto; padding: 1.5rem; }}
  h1 {{ font-size: 1.75rem; margin-bottom: 0.25rem; }}
  .sub {{ color: #9aa7b5; margin-bottom: 1.5rem; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1rem; margin: 1.5rem 0; }}
  .card {{ background: #1a2332; border-radius: 10px; padding: 1rem 1.1rem; border: 1px solid #2a3a4f; }}
  .card h3 {{ margin: 0 0 0.5rem; font-size: 1rem; color: #7dd3fc; }}
  .big {{ font-size: 1.6rem; font-weight: 700; color: #86efac; }}
  .pass {{ color: #86efac; font-weight: 600; }}
  .fail {{ color: #fca5a5; font-weight: 600; }}
  .grid {{ display: grid; gap: 1.5rem; }}
  .scope {{ background: #1a2332; padding: 1rem; border-radius: 8px; border-left: 4px solid #fbbf24; }}
  ul {{ line-height: 1.5; }}
  a {{ color: #7dd3fc; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Did people in the past only live to 30–35?</h1>
  <p class="sub">Bulletproof board · HMD public summary + HLD mid-adult ladder · cluster-bootstrapped · de-duplicated · gates {"PASS" if gates.get("all_pass") else "FAIL"}</p>

  <div class="scope">
    <strong>Scope:</strong> High-quality vital-registration populations (HMD public summary), not all premodern humans.
    Period life expectancy ≠ lived cohort lifespan. Conditional ages require surviving to that age.
  </div>

  <div class="cards">
    <div class="card">
      <h3>Claim A — not “died at 30”</h3>
      <div class="big">{a["exp_death_65"]["median"]:.1f}</div>
      <p>Median expected age if alive at 65 when e0&lt;40<br/>
      (cluster 95% CI {a["exp_death_65"]["ci_low"]:.1f}–{a["exp_death_65"]["ci_high"]:.1f})<br/>
      Share ≥70: <strong>{100*a["share_exp65_ge_70"]:.0f}%</strong> · n={a["n_country_years"]}<br/>
      Median S(0→65)={a["median_s_to_65"]:.2f} · IMR={a["median_imr"]:.0f}/1000<br/>
      <span class="{"pass" if gates["A"]["pass"] else "fail"}">Gate A: {"PASS" if gates["A"]["pass"] else "FAIL"}</span></p>
    </div>
    <div class="card">
      <h3>Claim A2 — mid-adult (age 15)</h3>
      <div class="big">{(a15.get("median") or 0):.1f}</div>
      <p>Median expected age if alive at 15 when e0&lt;40<br/>
      Quality: {a2.get("quality_used")}<br/>
      Share age|15 ≥50: {100*(a15.get("share_ge_50") or 0):.0f}% · n={a15.get("n_rows")}<br/>
      Age|30 median: {(a30.get("median") or float("nan")):.1f}<br/>
      <span class="{"pass" if gates["A2"]["pass"] else "fail"}">Gate A2: {"PASS" if gates["A2"]["pass"] else "FAIL"}</span></p>
    </div>
    <div class="card">
      <h3>Claim B — adults improved</h3>
      <div class="big">+{b["mean"]:.1f}y</div>
      <p>Mean rise in e65 pre-1900 → post-2000<br/>
      Cluster 95% CI {b["ci_low"]:.1f}–{b["ci_high"]:.1f}<br/>
      n countries={b["n"]}<br/>
      <span class="{"pass" if gates["B"]["pass"] else "fail"}">Gate B: {"PASS" if gates["B"]["pass"] else "FAIL"}</span></p>
    </div>
    <div class="card">
      <h3>Claim C — infant mechanism</h3>
      <div class="big">{c["median_within_corr_e0_imr"]:.2f}</div>
      <p>Median within-country corr(e0, IMR)<br/>
      First-diff β(ΔIMR)={c["models"]["M3_first_diff"]["beta_dimr"]:.3f}<br/>
      Associational only — not causal<br/>
      <span class="{"pass" if gates["C"]["pass"] else "fail"}">Gate C: {"PASS" if gates["C"]["pass"] else "FAIL"}</span></p>
    </div>
  </div>

  <div class="grid">
    {fig("D1_sweden_storyboard.png", "Sweden long run: e0, age|65, IMR")}
    {fig("D2_age_ladder_when_e0_under_40.png", "Age ladder when e0&lt;40 — hole-free multi-age")}
    {fig("D4_low_e0_distributions.png", "Distributions under the myth threshold")}
    {fig("D6_survival_composition.png", "Early death composition")}
    {fig("D5_mythA_forest.png", "Every country: age|65 still high when e0 low")}
    {fig("D10_mythB_dumbbell.png", "Adult remaining LE rose")}
  </div>

  <h2>What we are not claiming</h2>
  <ul>
    <li>That everyone reached 65 (or 15) historically — survival shares are low.</li>
    <li>That this describes all global premodern populations outside VR data.</li>
    <li>That regressions identify causal effects of infant mortality.</li>
    <li>That period e0 equals the lifespan of a real birth cohort.</li>
  </ul>

  <p class="sub">Regenerate: <code>python -m src.analysis.run_next_level</code> · Full writeup: MYTH_BUST_BULLETPROOF.md</p>
</div>
</body>
</html>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
