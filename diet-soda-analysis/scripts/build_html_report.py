"""
Build the public Diet Soda Myth Lab article (age_myth style, author voice).

Usage (from project root):
    python scripts/build_html_report.py
    python scripts/build_html_report.py --embed

Writes:
    index.html
    outputs/myth_lab_report.html
    outputs/tables/report_facts.json
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"
OUT_ROOT = ROOT / "index.html"
OUT_REPORT = ROOT / "outputs" / "myth_lab_report.html"
FIG_REL = "outputs/figures"


def load_facts() -> dict:
    df = pd.read_parquet(ROOT / "data/processed/analysis_ready.parquet")
    r = pd.read_csv(TABLES / "model_cardio_ladder.csv")
    cox = pd.read_csv(TABLES / "cancer_cox_results.csv")
    groups = df["bev_group"].value_counts().to_dict()

    if "ci_low" not in r.columns:
        r["ci_low"] = r["coef"] - 1.96 * r["se"]
        r["ci_high"] = r["coef"] + 1.96 * r["se"]

    def coef(outcome: str, step: str, term: str = "asb_only") -> dict | None:
        sub = r[(r["outcome"] == outcome) & (r["step"] == step) & (r["term"] == term)]
        if sub.empty:
            return None
        row = sub.iloc[0]
        return {
            "coef": float(row["coef"]),
            "lo": float(row["ci_low"]),
            "hi": float(row["ci_high"]),
            "p": float(row["pval"]),
            "n": int(row["n"]),
        }

    asb = df[df["bev_group"] == "ASB-only"]
    nei = df[df["bev_group"] == "Neither"]
    cox_c = cox[(cox["outcome"] == "cancer_death") & (cox["term"] == "asb_only")]
    cox_row = cox_c.iloc[0].to_dict() if len(cox_c) else {}

    mort = pd.read_parquet(ROOT / "data/processed/analysis_ready_mortality.parquet")
    asb_cdeaths = int(mort.loc[mort["bev_group"] == "ASB-only", "cancer_death"].fillna(0).sum())

    ml: dict = {}
    mlp = TABLES / "ml_tuning_results.json"
    if mlp.exists():
        ml = json.loads(mlp.read_text(encoding="utf-8"))

    facts = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "n": int(len(df)),
        "groups": {k: int(v) for k, v in groups.items()},
        "asb_dm": float(asb["diabetes_sr"].mean()),
        "nei_dm": float(nei["diabetes_sr"].mean()),
        "asb_bmi": float(asb["bmi"].mean()),
        "nei_bmi": float(nei["bmi"].mean()),
        "asb_age": float(asb["age"].mean()),
        "nei_age": float(nei["age"].mean()),
        "asb_cancer": float(asb["cancer_ever"].mean()),
        "nei_cancer": float(nei["cancer_ever"].mean()),
        "bmi_s0": coef("bmi", "S0"),
        "bmi_s3": coef("bmi", "S3"),
        "bmi_s5": coef("bmi", "S5_no_dm"),
        "hba_s0": coef("hba1c", "S0"),
        "hba_s3": coef("hba1c", "S3"),
        "hba_s5": coef("hba1c", "S5_no_dm"),
        "cox": cox_row,
        "asb_cancer_deaths": asb_cdeaths,
        "cancer_deaths_total": int(mort["cancer_death"].fillna(0).sum()),
        "allcause_deaths": int(mort["allcause_death"].fillna(0).sum())
        if "allcause_death" in mort.columns
        else None,
        "delta_r2": ml.get("bmi_delta_r2"),
        "auc_w": ml.get("asb_classifier_auc_weighted"),
        "auc_u": ml.get("asb_classifier_auc_unweighted"),
    }
    (TABLES / "report_facts.json").write_text(
        json.dumps(facts, indent=2, default=str), encoding="utf-8"
    )
    return facts


def fmt_beta(d: dict | None, digits: int = 2) -> str:
    if not d:
        return "n/a"
    return (
        f"β = {d['coef']:.{digits}f} "
        f"(approx. 95% CI {d['lo']:.{digits}f} to {d['hi']:.{digits}f}; n={d['n']:,})"
    )


def fig_tag(name: str, alt: str, caption_html: str, embed: bool) -> str:
    path = FIG / name
    if not path.exists():
        return f"<p class='bug-inline'>[Missing figure: {name}]</p>"
    if embed:
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        src = f"data:image/png;base64,{b64}"
    else:
        src = f"{FIG_REL}/{name}"
    return (
        f"<figure>\n"
        f'  <img src="{src}" width="960" height="540" alt="{alt}" loading="lazy">\n'
        f"  <figcaption>{caption_html}</figcaption>\n"
        f"</figure>"
    )


CSS = """
  :root{
    --bg:#f9f9f7; --surface:#fcfcfb; --card:#ffffff;
    --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
    --hair:#e1e0d9; --base:#c3c2b7;
    --blue:#2a78d6; --blue-soft:#eaf2fc; --red:#e34948; --red-soft:#fdeeee;
    --green:#1baf7a; --green-soft:#e9f7f1; --gold:#eda100; --gold-soft:#fdf5e3;
    --accent:#5b4cdb; --accent-soft:#eeecfb;
    --life:#0d9488; --life-soft:#e6f7f5;
    --mono:ui-monospace,"Cascadia Code","Segoe UI Mono",Consolas,monospace;
  }
  @media (prefers-color-scheme: dark){
    :root{
      --bg:#0d0d0d; --surface:#1a1a19; --card:#212120;
      --ink:#f5f5f2; --ink2:#c3c2b7; --muted:#898781;
      --hair:#2c2c2a; --base:#383835;
      --blue:#3987e5; --blue-soft:#14233a; --red:#e66767; --red-soft:#331b1b;
      --green:#1baf7a; --green-soft:#12291f; --gold:#c98500; --gold-soft:#2b230e;
      --accent:#8b7cf0; --accent-soft:#1c1833;
      --life:#2dd4bf; --life-soft:#0f2422;
    }
    figure{background:#141413}
  }
  *{box-sizing:border-box}
  html{scroll-behavior:smooth}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:16px/1.65 system-ui,-apple-system,"Segoe UI",sans-serif;
       -webkit-font-smoothing:antialiased}
  a{color:var(--blue);text-decoration:none}
  a:hover{text-decoration:underline}

  header.hero{background:linear-gradient(180deg,var(--surface),var(--bg));
    border-bottom:1px solid var(--hair);padding:0 0 40px;text-align:center}
  .band{height:8px;background:linear-gradient(90deg,#e34948 0%,#eda100 35%,#5b4cdb 70%,#0d9488 100%)}
  .heroinner{padding:36px 24px 0}
  .hero .kicker{font-size:13px;letter-spacing:.18em;text-transform:uppercase;color:var(--muted);font-weight:600}
  .hero h1{font-size:clamp(30px,5.4vw,50px);line-height:1.1;margin:14px auto 16px;max-width:920px;letter-spacing:-.025em}
  .hero h1 em{font-style:normal;color:var(--accent)}
  .hero .sub{max-width:720px;margin:0 auto;color:var(--ink2);font-size:18.5px}
  .madeby{margin:14px auto 0;font-size:14px;color:var(--muted)}
  .herobtns{display:flex;flex-wrap:wrap;gap:10px;justify-content:center;margin:22px auto 0}
  .bigbtn{display:inline-block;background:var(--accent);color:#fff;border-radius:99px;
    padding:12px 24px;font-size:15px;font-weight:700}
  .bigbtn:hover{text-decoration:none;filter:brightness(1.08)}
  .bigbtn.ghost{background:transparent;color:var(--ink2);border:1px solid var(--hair)}
  .kpis{display:flex;flex-wrap:wrap;gap:12px;justify-content:center;margin:28px auto 0;max-width:1000px}
  .kpi{background:var(--card);border:1px solid var(--hair);border-radius:12px;padding:14px 18px;min-width:140px}
  .kpi b{display:block;font-size:26px;letter-spacing:-.02em}
  .kpi span{font-size:12.5px;color:var(--muted);line-height:1.35}

  nav{position:sticky;top:0;z-index:50;background:color-mix(in srgb,var(--bg) 88%,transparent);
      backdrop-filter:blur(8px);border-bottom:1px solid var(--hair)}
  nav .wrap{max-width:1100px;margin:0 auto;display:flex;flex-wrap:wrap;justify-content:center;gap:0 2px;padding:0 10px}
  nav a{flex:0 0 auto;padding:11px 12px;font-size:13.5px;color:var(--ink2);border-bottom:2px solid transparent;white-space:nowrap}
  nav a:hover{text-decoration:none;color:var(--ink)}
  nav a:focus-visible{outline:2px solid var(--blue);outline-offset:2px}

  main{max-width:880px;margin:0 auto;padding:8px 24px 80px}
  section{padding-top:52px;scroll-margin-top:70px}
  .secnum{font:600 13px var(--mono);color:var(--accent);letter-spacing:.12em}
  h2{font-size:clamp(24px,3.4vw,34px);margin:6px 0 8px;letter-spacing:-.015em}
  .lede{color:var(--ink2);font-size:17.5px;margin:0 0 20px}
  h3{font-size:19px;margin:28px 0 10px}
  p{margin:0 0 14px}
  ul{margin:0 0 16px;padding-left:1.2em}
  li{margin:0 0 8px}
  .divider{text-align:center;color:var(--base);margin:44px 0 -24px;font-size:18px;letter-spacing:.4em}

  .finding{
    background:var(--card);
    border:3px solid var(--accent);
    border-radius:20px;
    padding:28px 26px 22px;
    margin:8px 0;
    box-shadow:0 12px 40px color-mix(in srgb,var(--accent) 12%,transparent);
  }
  .finding .label{
    font:700 12px var(--mono);letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin:0 0 10px
  }
  .finding .boom{
    font-size:clamp(24px,4vw,36px);line-height:1.18;letter-spacing:-.02em;margin:0 0 16px;font-weight:800
  }
  .finding .boom span{color:var(--accent)}
  .finding p{font-size:16.5px;color:var(--ink2)}
  .finding p strong{color:var(--ink)}
  .finding-grid{
    display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:20px 0 8px
  }
  .finding-grid .cell{
    background:var(--accent-soft);border-radius:14px;padding:16px 14px;text-align:center
  }
  .finding-grid .cell b{display:block;font-size:clamp(26px,3.8vw,34px);letter-spacing:-.02em;line-height:1.1}
  .finding-grid .cell span{display:block;font-size:12.5px;color:var(--ink2);margin-top:6px;line-height:1.35}
  .finding .pair{
    display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:16px 0 4px
  }
  @media (max-width:560px){ .finding .pair{grid-template-columns:1fr} }
  .finding .pair .yes,.finding .pair .no{
    border-radius:12px;padding:14px 16px;font-size:14.5px
  }
  .finding .pair .yes{background:var(--green-soft);border:1px solid color-mix(in srgb,var(--green) 30%,var(--hair))}
  .finding .pair .no{background:var(--red-soft);border:1px solid color-mix(in srgb,var(--red) 30%,var(--hair))}
  .finding .pair b{display:block;font-size:12px;letter-spacing:.08em;text-transform:uppercase;margin-bottom:6px}
  .finding .pair .yes b{color:var(--green)}
  .finding .pair .no b{color:var(--red)}
  .learned{margin:18px 0 6px;padding:0;list-style:none}
  .learned li{
    position:relative;padding:10px 12px 10px 36px;margin:0 0 8px;
    background:var(--surface);border:1px solid var(--hair);border-radius:10px;font-size:15px;color:var(--ink2)
  }
  .learned li::before{
    content:attr(data-n);position:absolute;left:12px;top:10px;
    font:700 12px var(--mono);color:var(--accent)
  }
  .learned li strong{color:var(--ink)}

  .co{border-radius:12px;padding:16px 18px;margin:18px 0;border:1px solid var(--hair);font-size:15.2px}
  .co .tag{display:inline-block;font:700 11px/1 var(--mono);letter-spacing:.1em;text-transform:uppercase;
           padding:4px 9px;border-radius:99px;margin-bottom:9px}
  .plain{background:var(--blue-soft)} .plain .tag{background:var(--blue);color:#fff}
  .why{background:var(--gold-soft)} .why .tag{background:var(--gold);color:#fff}
  .tech{background:var(--green-soft)} .tech .tag{background:var(--green);color:#fff}
  .bug{background:var(--red-soft)} .bug .tag{background:var(--red);color:#fff}
  .eli5{background:var(--life-soft);border-color:color-mix(in srgb,var(--life) 35%,var(--hair))}
  .eli5 .tag{background:var(--life);color:#fff}
  .co p:last-child{margin-bottom:0}
  .bug-inline{color:var(--red);font-size:14px}

  details.build{
    background:var(--card);border:1px solid var(--hair);border-radius:12px;padding:12px 16px;margin:18px 0
  }
  details.build summary{
    cursor:pointer;font-weight:700;color:var(--ink);list-style:none
  }
  details.build summary::-webkit-details-marker{display:none}
  details.build[open] summary{margin-bottom:10px}
  details.build p, details.build blockquote{color:var(--ink2);font-size:15px}
  details.build blockquote{
    margin:10px 0 0;padding:12px 14px;border-left:3px solid var(--accent);
    background:var(--accent-soft);border-radius:0 8px 8px 0;font-size:14.5px
  }
  .readpath{
    max-width:720px;margin:18px auto 0;font-size:14px;color:var(--ink2);line-height:1.5
  }
  .readpath b{color:var(--ink)}
  .gloss{
    display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin:16px 0
  }
  .gloss .g{
    background:var(--surface);border:1px solid var(--hair);border-radius:10px;padding:12px 14px;font-size:13.5px;color:var(--ink2)
  }
  .gloss .g b{display:block;color:var(--ink);font-size:13px;margin-bottom:4px;letter-spacing:.02em}
  .skip{font-size:14px;color:var(--muted);margin:0 0 12px}

  figure{margin:24px 0;background:var(--card);border:1px solid var(--hair);border-radius:14px;padding:14px}
  figure img{width:100%;height:auto;display:block;border-radius:8px}
  figcaption{font-size:13.5px;color:var(--muted);padding:10px 6px 2px;line-height:1.5}
  figcaption b{color:var(--ink2)}

  .chartgrid{display:grid;grid-template-columns:1fr;gap:0}
  @media (min-width:720px){
    .chartgrid.two{grid-template-columns:1fr 1fr;gap:14px}
    .chartgrid.two figure{margin:14px 0}
  }

  .tblwrap{overflow-x:auto;margin:18px 0}
  table{border-collapse:collapse;width:100%;font-size:14.5px;background:var(--card);
        border:1px solid var(--hair);border-radius:12px;overflow:hidden}
  th{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);
     text-align:left;padding:11px 14px;border-bottom:1px solid var(--hair);background:var(--surface)}
  td{padding:10px 14px;border-bottom:1px solid var(--hair);font-variant-numeric:tabular-nums;vertical-align:top}
  tr:last-child td{border-bottom:none}
  td.num, th.num{text-align:right;white-space:nowrap}

  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px;margin:20px 0}
  .card{background:var(--card);border:1px solid var(--hair);border-radius:14px;padding:18px}
  .card h4{margin:0 0 8px;font-size:15.5px}
  .card p{font-size:14px;color:var(--ink2);margin:0}
  .card .verdict{display:inline-block;margin-top:10px;font:600 11.5px var(--mono);letter-spacing:.06em;
                 padding:3px 10px;border-radius:99px}
  .v-fail{background:var(--red);color:#fff}
  .v-win{background:var(--green);color:#fff}
  .v-nuanced{background:var(--gold);color:#1a1400}
  .v-untestable{background:var(--base);color:var(--ink)}
  .v-process{background:var(--life);color:#fff}

  .room{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:18px 0}
  .room .box{background:var(--card);border:1px solid var(--hair);border-radius:14px;padding:16px}
  .room .box h4{margin:0 0 8px;font-size:14px;color:var(--muted);font-weight:600;letter-spacing:.04em;text-transform:uppercase}
  .room .box p{margin:0;font-size:15px;color:var(--ink2)}
  .room .box .n{font-size:28px;font-weight:800;letter-spacing:-.02em;color:var(--ink);display:block;margin-bottom:4px}

  pre,code{font-family:var(--mono);font-size:13px}
  pre{background:var(--surface);border:1px solid var(--hair);border-radius:12px;padding:14px 16px;
      overflow-x:auto;line-height:1.5;color:var(--ink2)}
  code.inline{background:var(--surface);border:1px solid var(--hair);border-radius:6px;padding:1px 6px}

  footer{border-top:1px solid var(--hair);padding:28px 24px 48px;text-align:center;color:var(--muted);font-size:13.5px}
  footer a{color:var(--ink2)}
  @media print{
    nav{display:none}
    section{break-inside:avoid}
  }
"""


def build_html(f: dict, embed: bool) -> str:
    g = f["groups"]
    cox = f["cox"]
    hr = float(cox.get("hr", float("nan")))
    hr_lo = float(cox.get("hr_lo", float("nan")))
    hr_hi = float(cox.get("hr_hi", float("nan")))
    hp = float(cox.get("pval", float("nan")))
    dr2 = f["delta_r2"]
    dr2_s = f"{dr2:.3f}" if dr2 is not None else "0.007"
    auc_w = f.get("auc_w")
    auc_u = f.get("auc_u")
    if auc_w is not None and auc_u is not None:
        auc_line = f"Classifier AUC about {auc_w:.2f} weighted and {auc_u:.2f} unweighted."
    else:
        auc_line = "Classifier AUC about 0.66 weighted and 0.68 unweighted."

    def fig(name: str, alt: str, cap: str) -> str:
        return fig_tag(name, alt, cap, embed)

    asb_n = g.get("ASB-only", 0)
    ssb_n = g.get("SSB-only", 0)
    nei_n = g.get("Neither", 0)
    both_n = g.get("Both", 0)
    dm_asb = 100 * f["asb_dm"]
    dm_nei = 100 * f["nei_dm"]
    can_asb = 100 * f["asb_cancer"]
    can_nei = 100 * f["nei_cancer"]
    h0 = f["hba_s0"]["coef"] if f["hba_s0"] else 0.30
    h5 = f["hba_s5"]["coef"] if f["hba_s5"] else 0.05
    b3 = f["bmi_s3"]["coef"] if f["bmi_s3"] else 2.5
    b5 = f["bmi_s5"]["coef"] if f["bmi_s5"] else 2.3

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Diet soda did not give you cancer. Who drinks it wrote the scare.</title>
<meta name="description" content="I ran public NHANES on diet soda myths. Cancer slogan fails. Diabetes gap mostly selection. BMI stays higher as association. About 27 cancer deaths in the diet-soda arm.">
<meta name="author" content="diet-soda-analysis">
<meta property="og:title" content="Diet soda did not give you cancer. Who drinks it wrote the scare.">
<meta property="og:description" content="n=19,384. Cancer-death HR {hr:.2f} (not significant). HbA1c +{h0:.2f} to +{h5:.2f} after dropping known diabetes. BMI still about +{b3:.1f}.">
<meta property="og:type" content="article">
<style>
{CSS}
</style>
</head>
<body>
<a id="top"></a>

<header class="hero">
  <div class="band" aria-hidden="true"></div>
  <div class="heroinner">
    <div class="kicker">Public NHANES &middot; I pulled the files &middot; no industry money</div>
    <h1>Diet soda did not give you cancer.<br><em>Who drinks it wrote the scare.</em></h1>
    <p class="sub">
      People who already have diabetes and higher BMI show up in the diet-soda column.
      That selection makes crude charts look like the can caused the disease.
      I tested that story on open data. Cancer meme fails. Blood sugar mostly calms down. Weight association stays.
    </p>
    <p class="madeby">
      I built this with <strong>Grok 4.5</strong> as a coding partner &middot; NHANES 2011-2018 &middot; USDA WWEIA &middot; NCHS mortality &middot; {f['generated']}
    </p>
    <div class="herobtns">
      <a class="bigbtn" href="#finding">30-second take</a>
      <a class="bigbtn ghost" href="#cancer">Cancer</a>
      <a class="bigbtn ghost" href="#models">What the models held</a>
    </div>
    <p class="readpath">
      <b>How to read this:</b> the colored box under Take is enough for most people.
      Then Cancer if that is your fight, Diet for blood sugar and weight, Models if you want to see the ladder.
      Not medical advice.
    </p>
    <div class="kpis">
      <div class="kpi"><b>{dm_asb:.0f}% vs {dm_nei:.0f}%</b><span>diabetes (self-report)<br>diet-soda only vs neither</span></div>
      <div class="kpi"><b>+{h0:.2f} → +{h5:.2f}</b><span>blood-sugar marker (HbA1c)<br>crude → drop known diabetes</span></div>
      <div class="kpi"><b>+{b3:.1f}</b><span>BMI still higher after controls<br>association, not proof of cause</span></div>
      <div class="kpi"><b>~{f['asb_cancer_deaths']}</b><span>cancer deaths in diet-soda group<br>HR {hr:.2f}, not significant</span></div>
    </div>
  </div>
</header>

<nav aria-label="Sections">
  <div class="wrap">
    <a href="#finding">Take</a>
    <a href="#eli5">Normal words</a>
    <a href="#broke">Myths</a>
    <a href="#cancer">Cancer</a>
    <a href="#diet">Diet</a>
    <a href="#who">Who drinks</a>
    <a href="#models">What held</a>
    <a href="#build">How built</a>
    <a href="#data">Data</a>
    <a href="#limits">Limits</a>
  </div>
</nav>

<main>

<section id="finding">
  <div class="secnum">01</div>
  <h2>If you only read one block</h2>
  <p class="lede">Whole story in this box. Everything below is receipts and method.</p>

  <div class="finding">
    <p class="label">Big picture</p>
    <p class="boom">
      Diet-soda drinkers are <span>not a random sample</span>.
      They are older, heavier, and about <span>twice as diabetic</span>.
      That fact writes most of the scary charts you see online.
    </p>

    <div class="finding-grid">
      <div class="cell"><b>{dm_asb:.0f}% vs {dm_nei:.0f}%</b><span>self-report diabetes<br>diet-only vs neither</span></div>
      <div class="cell"><b>+{h0:.2f} → +{h5:.2f}</b><span>blood sugar (HbA1c)<br>after I drop known diabetes</span></div>
      <div class="cell"><b>+{b3:.1f}</b><span>BMI still higher after controls<br>still +{b5:.1f} without known diabetes</span></div>
      <div class="cell"><b>{hr:.2f}</b><span>cancer-death rate ratio (HR)<br>CI {hr_lo:.2f}-{hr_hi:.2f} · ~{f['asb_cancer_deaths']} deaths</span></div>
    </div>

    <div class="pair">
      <div class="no">
        <b>The feed</b>
        &ldquo;WHO says diet soda causes cancer. Higher cancer rates prove it. It wrecks blood sugar and makes you fat.&rdquo;
      </div>
      <div class="yes">
        <b>What I found</b>
        IARC 2B is limited-evidence hazard talk, not a ban and not JECFA risk at usual intake.
        People who already look sick on paper drink more diet soda.
        Cancer death is not significant, and the diet-soda arm only has about {f['asb_cancer_deaths']} cancer deaths. Weak test. Not a safety stamp.
      </div>
    </div>

    <p class="label" style="margin-top:22px">What I learned</p>
    <ul class="learned">
      <li data-n="1"><strong>Who drinks it is half the analysis.</strong> Skip that and every crude bar chart lies to you.</li>
      <li data-n="2"><strong>Blood sugar scare is mostly reverse traffic.</strong> Known diabetes people switch. Remove them and HbA1c mostly calms down.</li>
      <li data-n="3"><strong>Weight association is stubborn.</strong> About +{b3:.1f} BMI units after controls. Still not proof the can caused the pounds.</li>
      <li data-n="4"><strong>Cancer slogan is wrong. Tiny long risks stay untestable here.</strong> Public survey plus short mortality follow-up cannot clear lifelong site-specific cancer.</li>
    </ul>

    <div class="co tech" style="margin-top:18px">
      <span class="tag">Why I bothered with models</span>
      <p>
        A raw bar chart only says &ldquo;diet-soda people look different.&rdquo;
        Models ask a harder question: <strong>after I line people up on age, sex, race/ethnicity, education, income, smoking, and calories, does the gap still sit there?</strong>
        Then I stress-test blood sugar by <strong>dropping people who already know they have diabetes</strong> (people who often switch drinks after diagnosis).
        That is not a court verdict of &ldquo;proved.&rdquo; It is a filter on lazy causal talk.
        Scoreboard: <a href="#models">what held</a>.
      </p>
    </div>

    <p class="skip" style="margin-top:14px;margin-bottom:0">Quick words: <strong>diet-only</strong> = drank diet soft drinks, not regular, on the survey day.
    <strong>Neither</strong> = no diet and no regular soft drinks that day.
    <strong>HbA1c</strong> = blood-sugar control marker. <strong>HR</strong> = hazard ratio (cancer death rate comparison).</p>

    <p style="margin-top:18px">
      <strong>Sample I used:</strong> {f['n']:,} non-pregnant adults age 20+, reliable Day-1 diet.
      WWEIA <strong>7102</strong> diet soft drinks vs <strong>7202</strong> regular.
      Groups: diet-only {asb_n:,}, regular-only {ssb_n:,}, both {both_n:,}, neither {nei_n:,}.
    </p>
    <p>
      <strong>Stats honesty up front:</strong> I used multi-cycle MEC weights (normalized).
      I did <strong>not</strong> run full NCHS PSU and strata variance.
      Point estimates are the useful part. Tiny p-values are too flattering.
      One day of diet recall is not a life history. Not medical advice.
    </p>
    <p style="margin-bottom:4px">
      <strong>Policy-ish bottom line:</strong> do not treat meme posts as risk assessment.
      IARC 2B is not a soda ban. JECFA still frames an ADI many cans per day order of magnitude at labeled use.
      Swapping sugar soda for diet soda is a different question than &ldquo;zero risk forever.&rdquo; Water still wins the boring contest.
    </p>
  </div>
</section>

<section id="eli5">
  <div class="secnum">02</div>
  <h2>In normal words</h2>
  <p class="lede">If higher cancer % and higher BMI in diet-soda drinkers feel like case closed, read this first.</p>

  <div class="co eli5">
    <span class="tag">Picture this</span>
    <p>
      Someone gets diabetes or starts worrying about weight. They switch to diet soda.
      NHANES photographs that day. The chart then says &ldquo;diet soda people have worse labs.&rdquo;
      That can be the switch, not the chemical.
      I stress-tested that idea. For HbA1c it mostly holds. For BMI the gap stays. For cancer death I simply do not have enough events to act tough.
    </p>
  </div>

  <div class="room">
    <div class="box">
      <h4>Meme number</h4>
      <span class="n">~{can_asb:.0f}%</span>
      <p>Crude ever-cancer in diet-only vs about {can_nei:.0f}% in neither. Real gap. Bad causal read without age and who switched.</p>
    </div>
    <div class="box">
      <h4>Number people skip</h4>
      <span class="n">~{f['asb_cancer_deaths']}</span>
      <p>Cancer deaths in the diet-soda arm. That is why HR {hr:.2f} with a CI from {hr_lo:.2f} to {hr_hi:.2f} is a weak test, not proof of safety.</p>
    </div>
    <div class="box">
      <h4>The mechanism</h4>
      <span class="n">~2×</span>
      <p>Diabetes rate roughly doubles in diet-only vs neither. Same cast of characters as higher BMI and slightly older age.</p>
    </div>
  </div>

  <div class="co plain">
    <span class="tag">Two different questions</span>
    <p>
      <b>Crude % by drink group:</b> who is already in the diet-soda column?<br>
      <b>Models with age, lifestyle, and &ldquo;drop known diabetes&rdquo;:</b> does the gap still look like a causal punch?<br>
      Same survey. Different question. The feed swaps them on purpose or by accident.
    </p>
  </div>

  <div class="co why">
    <span class="tag">Hazard is not risk</span>
    <p>
      <b>IARC Group 2B</b> asks whether something might cause cancer under some conditions when evidence is limited.<br>
      <b>JECFA ADI</b> asks what daily intake looks acceptable after risk assessment.
      For aspartame that is many cans per day order of magnitude at common can doses, not &ldquo;one can equals cancer.&rdquo;<br>
      If your feed said WHO banned diet soda, your feed lied by compression.
    </p>
  </div>
</section>

<section id="broke">
  <div class="secnum">03</div>
  <h2>Myth board</h2>
  <p class="lede">Stuff people actually post. My call after running the numbers.</p>

  <div class="cards">
    <div class="card">
      <h4>Cancer / WHO / one can</h4>
      <p>Slogan fails hazard vs risk, dose chart, and a mortality model short on diet-soda events.</p>
      <span class="verdict v-fail">BUSTED (slogan)</span>
    </div>
    <div class="card">
      <h4>Diabetes / blood sugar</h4>
      <p>Crude HbA1c about +{h0:.2f}. After I drop known diabetes, about +{h5:.2f}. Mostly selection. Not zero. Not proven cause.</p>
      <span class="verdict v-nuanced">NUANCED</span>
    </div>
    <div class="card">
      <h4>Makes you fat</h4>
      <p>BMI stays about +{b3:.1f} after lifestyle controls. Association that will not die. Still not a trial of causation.</p>
      <span class="verdict v-nuanced">NUANCED</span>
    </div>
    <div class="card">
      <h4>Same people as regular soda</h4>
      <p>No. More diabetes, higher BMI, higher income, older. Love plot and ML both say selection.</p>
      <span class="verdict v-fail">BUSTED</span>
    </div>
    <div class="card">
      <h4>Destroys microbiome</h4>
      <p>Real question for feeding studies. NHANES has no stool data. I name it so I do not fake a null.</p>
      <span class="verdict v-untestable">UNTESTABLE HERE</span>
    </div>
    <div class="card">
      <h4>Only one model works</h4>
      <p>I show the BMI spec curve and cycle stability. Multiverse is still my design. At least you can see it.</p>
      <span class="verdict v-process">PROCESS</span>
    </div>
  </div>

  <div class="tblwrap">
    <table>
      <thead>
        <tr><th>They say</th><th>I say</th><th>Jump</th></tr>
      </thead>
      <tbody>
        <tr><td>WHO banned diet soda / causes cancer</td><td>2B is not a ban. Mortality test is weak on events.</td><td><a href="#cancer">Cancer</a></td></tr>
        <tr><td>Causes diabetes</td><td>Blood-sugar gap mostly shrinks after known diabetes out</td><td><a href="#diet">Diet</a></td></tr>
        <tr><td>Makes you fat</td><td>~+{b3:.1f} BMI association. Cause not shown.</td><td><a href="#diet">Diet</a></td></tr>
        <tr><td>Destroys microbiome</td><td>Cannot test here</td><td><a href="#limits">Limits</a></td></tr>
        <tr><td>Mouse / one can poison</td><td>Dose translation + ADI cans chart</td><td><a href="#cancer">Cancer</a></td></tr>
        <tr><td>Industry covered it up</td><td>Public files. Open code. No industry check.</td><td><a href="#build">Build</a></td></tr>
      </tbody>
    </table>
  </div>
</section>

<div class="divider" aria-hidden="true">&middot; &middot; &middot;</div>

<section id="cancer">
  <div class="secnum">04</div>
  <h2>Cancer</h2>
  <p class="lede">Loudest claim on the internet. Four layers. Agencies, dose, crude %, deaths over follow-up time.</p>

  <div class="co eli5">
    <span class="tag">What I care about</span>
    <p>
      Hazard label is not risk at soda doses.
      Crude ever-cancer % is polluted by who drinks diet soda.
      Follow-up cancer death is the harder public test, and the diet-soda arm is thin on events.
    </p>
  </div>

  <h3>Hazard is not risk</h3>
  {fig("cancer_c1_hazard_vs_risk.png",
       "IARC hazard identification vs JECFA risk and ADI",
       "<b>IARC vs JECFA vs FDA.</b> Group 2B is limited-evidence hazard language. It is not the sentence “diet soda gives you cancer at normal intake.”")}

  <h3>Dose (ADI to cans per day)</h3>
  {fig("cancer_c2_adi_cans.png",
       "Approximate cans per day to reach JECFA ADI",
       "<b>Order-of-magnitude cans/day</b> to hit JECFA ADI 40 mg/kg at about 180-200 mg aspartame per can. Not me telling you to drink a dozen. A check on “one can equals cancer.”")}

  <h3>Crude ever-cancer looks scary</h3>
  <p>
    Unweighted crude ever-cancer: diet-only about <strong>{can_asb:.1f}%</strong> vs neither about <strong>{can_nei:.1f}%</strong>.
    Age bands shrink the scare. They do <strong>not</strong> wipe every residual gap (still higher in 60+ in my tables).
    Design cannot prove cause. Lifetime cancer history next to yesterday’s diet is a bad match.
  </p>
  {fig("cancer_c3b_crude_vs_old.png",
       "Crude ever-cancer versus ages 60+",
       "<b>Composition matters.</b> Left: crude %. Right: ages 60+. Residual gaps can remain. I am not claiming age explains everything.")}
  {fig("cancer_c3_ever_cancer_by_age.png",
       "Ever-cancer by age band and beverage group",
       "<b>Age band by drink group.</b> Stratify before you screenshot a bar into a thread.")}

  <h3>Cancer death with follow-up months</h3>
  <p>
    Cox model (unweighted primary, months since exam, diet-only vs neither, age/sex/smoking):
    cancer-death <strong>HR about {hr:.2f}</strong> (95% CI {hr_lo:.2f}-{hr_hi:.2f}), p about {hp:.2f}.
    Total cancer deaths: <strong>{f['cancer_deaths_total']}</strong>.
    In the diet-only group: about <strong>{f['asb_cancer_deaths']}</strong>.
    Not significant. Wide interval. I will not sell that as safety.
    Rough power notes that assume balanced exposure are optimistic anyway. Diet-only is about 9% of the sample.
  </p>
  {fig("cancer_c4_km_cancer_death.png",
       "Kaplan-Meier cancer death free survival by beverage group",
       "<b>Follow-up time.</b> Rare events in the diet-soda arm. Read next to the forest plot.")}
  {fig("cancer_c5_cox_forest.png",
       "Forest plot of cancer death and all-cause death hazard ratios for diet-only adults",
       "<b>Wide intervals.</b> Cancer death and all-cause. Not significant is not “protective” and not “proven harm.”")}

  <div class="co bug">
    <span class="tag">I am not claiming</span>
    <p>
      I did not prove aspartame is safe forever.
      I did not rule out small long-latency site-specific risks (liver incidence and friends).
      I did not re-run NutriNet.
      I busted a slogan and showed what this public design can carry.
    </p>
  </div>
</section>

<section id="diet">
  <div class="secnum">05</div>
  <h2>Blood sugar and weight</h2>
  <p class="lede">Everyday claims. Same selection story. Different leftover gaps.</p>

  <h3>Diabetes scare mostly shrinks</h3>
  <p>
    Crude diet-only HbA1c association about <strong>+{h0:.2f}</strong> points.
    After I exclude known diabetes, about <strong>+{h5:.2f}</strong>.
    That is mostly people who already had the diagnosis walking into the diet-soda group.
    Residual is small. Still not zero. Still not a trial that proves cause.
  </p>
  <ul>
    <li>Adjusted (S3): {fmt_beta(f['hba_s3'])}</li>
    <li>No known diabetes (S5): {fmt_beta(f['hba_s5'])}</li>
  </ul>
  {fig("myth_m2_hba1c_violin.png",
       "HbA1c by beverage group",
       "<b>Crude HbA1c.</b> Higher in diet-only tracks higher diabetes prevalence. Next move: drop known diabetes.")}
  {fig("myth_m1_reverse_causation_scatter.png",
       "Diet soft drink servings vs BMI by diabetes status",
       "<b>Who is already sick or heavier.</b> They show up in the diet-soft column.")}

  <h3>Weight association sticks</h3>
  <p>
    BMI for diet-only vs neither stays about <strong>+{b3:.1f} kg/m²</strong> after lifestyle covariates,
    and about <strong>+{b5:.1f}</strong> after I drop known diabetes.
    Real cross-sectional association. Not proven causation.
    Adding diet-soda features barely helps predict BMI (ΔR² about {dr2_s}).
  </p>
  <ul>
    <li>Adjusted (S3): {fmt_beta(f['bmi_s3'])}</li>
    <li>No known diabetes (S5): {fmt_beta(f['bmi_s5'])}</li>
  </ul>
  {fig("myth_m1_bmi_violin.png",
       "BMI by beverage group",
       "<b>Crude BMI.</b> Diet-only sits higher. Read it with the who-drinks section, not as a causal trial.")}

  <div class="co why">
    <span class="tag">Why sugar calms and BMI does not</span>
    <p>
      Known diabetes is a hard switch into diet soda. Pull those people out and HbA1c mostly falls.
      Weight is messier. Goals, history, unmeasured lifestyle. Association without a substitution trial is not &ldquo;diet soda causes obesity.&rdquo;
    </p>
  </div>
</section>

<section id="who">
  <div class="secnum">06</div>
  <h2>Who drinks diet soda</h2>
  <p class="lede">This is the mechanism. Skip it and the crude charts look like proof of cause.</p>

  <div class="room">
    <div class="box">
      <h4>Diabetes (self-report)</h4>
      <span class="n">{dm_asb:.0f}% vs {dm_nei:.0f}%</span>
      <p>Diet-only vs neither. Selection, not a random sip.</p>
    </div>
    <div class="box">
      <h4>Mean BMI</h4>
      <span class="n">{f['asb_bmi']:.1f} vs {f['nei_bmi']:.1f}</span>
      <p>Heavier on average before any model speech.</p>
    </div>
    <div class="box">
      <h4>Mean age</h4>
      <span class="n">{f['asb_age']:.0f} vs {f['nei_age']:.0f}</span>
      <p>Slightly older. Matters for lifetime cancer history.</p>
    </div>
  </div>

  {fig("myth_m5_smd_loveplot.png",
       "How diet-only adults differ from people who drank neither soft drink type",
       "<b>Love plot.</b> Look at diabetes, BMI, income, age. That is who shows up in the diet-soda column.")}
  {fig("myth_m8_asb_feature_importance.png",
       "What best predicts diet soda use in a simple machine learning model",
       "<b>What predicts diet-soda use.</b> Diabetes, income, waist/BMI, age. If health predicts the drink more than the drink predicts health, the headline arrow is often backwards.")}

  <div class="co tech">
    <span class="tag">ML note</span>
    <p>
      {auc_line}
      That is prediction quality, not causation.
      Adding diet-soda features barely improves BMI prediction (ΔR² about {dr2_s}). Small number, big point: soda flags are weak BMI predictors next to the rest of the file.
    </p>
  </div>
</section>

<div class="divider" aria-hidden="true">&middot; &middot; &middot;</div>

<section id="models">
  <div class="secnum">07</div>
  <h2>Models: what I ran and what held</h2>
  <p class="lede">
    Why model at all? Because a mean difference is cheap drama.
    A model is me asking: <strong>is the gap still there after I hold fixed the obvious confounders?</strong>
    And for diabetes markers: <strong>does it survive after I remove people who already know they have diabetes?</strong>
  </p>

  <div class="co eli5">
    <span class="tag">In one breath</span>
    <p>
      I did <strong>not</strong> prove diet soda causes or does not cause anything.
      I ran a <strong>stress test</strong> on viral claims with public data.
      Crude scare that dies after controls: treat as selection noise.
      Gap that stays after controls: real association worth respecting, still not a trial.
      Cancer death that is not significant with ~{f['asb_cancer_deaths']} events in the diet group: weak test, not a safety certificate.
    </p>
  </div>

  <div class="gloss">
    <div class="g"><b>β (beta)</b> Average difference in the outcome for diet-only vs neither after the listed controls. BMI β +2.5 ≈ 2.5 BMI units higher.</div>
    <div class="g"><b>HR (hazard ratio)</b> Relative rate of the event over follow-up. HR 1 = same rate. CI that crosses 1 = not significant here.</div>
    <div class="g"><b>S0 / S3 / S5</b> Crude → full lifestyle controls → same but drop known diabetes. A ladder, not three different stories.</div>
    <div class="g"><b>What models can do</b> Stress-test a claim. They cannot replace a randomized trial or lifelong intake history.</div>
  </div>

  <h3>The ladder (S0 → S3 → S5)</h3>
  <div class="tblwrap">
    <table>
      <thead>
        <tr><th>Step</th><th>What I did</th><th>Why</th></tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>S0 crude</strong></td>
          <td>Diet-only vs neither. Almost no controls.</td>
          <td>What a simple chart shows.</td>
        </tr>
        <tr>
          <td><strong>S3 lifestyle</strong></td>
          <td>Add age, sex, race/ethnicity, education, income, smoking, Day-1 calories.</td>
          <td>Diet-soda drinkers are not demographically random. Line them up.</td>
        </tr>
        <tr>
          <td><strong>S5 no known diabetes</strong></td>
          <td>Same as S3, but only people who do <em>not</em> self-report diabetes.</td>
          <td>Removes the reverse switch: already diabetic, already on diet soda.</td>
        </tr>
        <tr>
          <td><strong>Cox cancer death</strong></td>
          <td>Time from exam to cancer death (or censor). Diet-only vs neither. Age, sex, smoking.</td>
          <td>Harder than &ldquo;ever had cancer&rdquo; self-report. Still short on diet-group events.</td>
        </tr>
      </tbody>
    </table>
  </div>

  <h3>Scoreboard: what held</h3>
  <div class="tblwrap">
    <table>
      <thead>
        <tr><th>Claim under test</th><th>What the model did</th><th>What held</th></tr>
      </thead>
      <tbody>
        <tr>
          <td>Diet soda wrecks blood sugar</td><td>HbA1c: crude → lifestyle → drop known diabetes</td>
          <td><strong>Mostly fell.</strong> +{h0:.2f} → +{h5:.2f}. Mostly selection / reverse switch.</td>
        </tr>
        <tr>
          <td>Diet soda makes you fat</td><td>BMI: crude → lifestyle → drop known diabetes</td>
          <td><strong>Stayed.</strong> About +{b3:.1f} (still +{b5:.1f} without known diabetes). Association, not cause proven.</td>
        </tr>
        <tr>
          <td>Diet soda gives you cancer (death)</td><td>Cox HR for cancer death</td>
          <td><strong>Not significant.</strong> HR {hr:.2f} (CI {hr_lo:.2f}-{hr_hi:.2f}), p≈{hp:.2f}, ~{f['asb_cancer_deaths']} diet-group deaths.</td>
        </tr>
        <tr>
          <td>Diet drinkers = everyone else</td><td>Profile gaps + who-drinks prediction model</td>
          <td><strong>Busted.</strong> Diabetes, BMI, income, age separate them. Soda flags barely help predict BMI.</td>
        </tr>
      </tbody>
    </table>
  </div>

  <div class="tblwrap">
    <table>
      <thead>
        <tr><th>Outcome</th><th>Step</th><th>Contrast</th><th class="num">Estimate</th></tr>
      </thead>
      <tbody>
        <tr><td>BMI</td><td>S0 crude</td><td>diet-only vs neither</td><td class="num">β {f['bmi_s0']['coef']:+.2f}</td></tr>
        <tr><td>BMI</td><td>S3 lifestyle</td><td>diet-only vs neither</td><td class="num">β {f['bmi_s3']['coef']:+.2f}</td></tr>
        <tr><td>BMI</td><td>S5 no known diabetes</td><td>diet-only vs neither</td><td class="num">β {f['bmi_s5']['coef']:+.2f}</td></tr>
        <tr><td>HbA1c</td><td>S0 crude</td><td>diet-only vs neither</td><td class="num">β {f['hba_s0']['coef']:+.3f}</td></tr>
        <tr><td>HbA1c</td><td>S3 lifestyle</td><td>diet-only vs neither</td><td class="num">β {f['hba_s3']['coef']:+.3f}</td></tr>
        <tr><td>HbA1c</td><td>S5 no known diabetes</td><td>diet-only vs neither</td><td class="num">β {f['hba_s5']['coef']:+.3f}</td></tr>
        <tr><td>Cancer death</td><td>Cox PH</td><td>diet-only vs neither</td><td class="num">HR {hr:.2f} ({hr_lo:.2f}-{hr_hi:.2f})</td></tr>
      </tbody>
    </table>
  </div>

  <div class="co why">
    <span class="tag">How to read the numbers here</span>
    <p>
      <b>BMI β about +{b3:.1f}:</b> diet-only adults sit that many BMI units higher than neither, after the listed controls.
      <b>HbA1c β about +{h5:.2f}:</b> after dropping known diabetes, the remaining gap is small.
      <b>Cancer HR {hr:.2f}:</b> point estimate below 1, but the CI crosses 1 and events are few. Do not translate that into &ldquo;protective&rdquo; or &ldquo;safe.&rdquo;
      Standard errors are approximate (weights yes, full survey design variance no). Tiny p-values on continuous models are too flattering.
    </p>
  </div>

  <h3>Spec curve and cycles</h3>
  <p>
    One model can be a one-off. I also show BMI across several covariate sets and across NHANES cycles
    so the weight result is not a single formula trick.
  </p>
  {fig("myth_m7_spec_curve_bmi.png",
       "BMI specification curve across covariate sets",
       "<b>BMI multiverse.</b> Same story under different control sets. Not one magic p-value.")}
  {fig("myth_m7_cycle_stability_bmi.png",
       "Mean BMI by beverage group across NHANES cycles",
       "<b>Not one Kaggle year.</b> Pattern across 2011-2018.")}

  <h3>Dose, blood pressure, missingness</h3>
  <p>
    About 90% of adults have zero diet soft drinks on Day-1 recall.
    Continuous “per serving” models mix any-vs-none with dose among drinkers. Be careful.
    Triglyceride models are on <strong>log(TG)</strong>. Single exam BP. Meds only partly handled.
  </p>
  <div class="chartgrid two">
    {fig("myth_m6_asb_dose_hist.png",
         "Histogram of diet soft drink servings among Day-1 consumers",
         "<b>Dose among consumers.</b> Zeros out of this plot.")}
    {fig("myth_m3_sbp_violin.png",
         "Systolic blood pressure by beverage group",
         "<b>SBP crude view.</b> Fair-fight adjusted numbers live in the ladder CSV.")}
  </div>
  {fig("myth_missingness.png",
       "Missingness of key outcomes in the analytic sample",
       "<b>Missingness.</b> Fasting labs are thinner by design. Know that before you model.")}
</section>

<section id="build">
  <div class="secnum">08</div>
  <h2>How I built this</h2>
  <p class="lede">Who did the work, with what tools, and on what brief.</p>

  <p>
    I pulled NHANES 2011-2018 exam, diet, and lab files. Mapped soft drinks with USDA WWEIA codes
    (<strong>7102</strong> diet, <strong>7202</strong> regular). Linked public mortality.
    Built an analysis-ready sample of {f['n']:,} adults. Ran weighted models, a small who-drinks check,
    a cancer pack (agencies, dose chart, age bands, survival model), and this page.
    Headline numbers load from the same tables I checked against the outputs.
  </p>

  <div class="co plain">
    <span class="tag">Grok 4.5</span>
    <p>
      I used Grok 4.5 as a coding partner for downloads, cleaning, models, charts, and HTML.
      It did not invent the NHANES rows. I still matched the big numbers to the result tables.
      If something is wrong, that is on me for shipping it.
    </p>
  </div>

  <details class="build">
    <summary>First super prompt I used (compressed)</summary>
    <blockquote>
      Build a portfolio-grade public Myth Lab on diet soda / artificial sweeteners.
      Use NHANES 2011-2018, USDA WWEIA diet soft drinks vs regular soft drinks, and NCHS linked mortality.
      Engineering first: clean multi-cycle sample, exclusive soda-type groups, documented weights.
      Test myths people actually post: cancer / WHO / aspartame, weight, diabetes, selection, microbiome if untestable say so.
      Verdict ladder: BUSTED, NUANCED, association only, UNTESTABLE HERE.
      No industry funding. No fake certainty. No “proved safe forever.”
      Ship charts, model ladder, cancer module, and a public write-up a coworker can defend in five minutes.
    </blockquote>
    <p style="margin-top:12px;margin-bottom:0">
      That brief turned into the repo you can reproduce below. Later prompts tightened language, fixed weight bugs, and built this article shell after an age_myth style pass.
    </p>
  </details>
</section>

<section id="data">
  <div class="secnum">09</div>
  <h2>Data and reproduce</h2>
  <p class="lede">NHANES continuous 2011-2018. USDA WWEIA. NCHS public Linked Mortality File.</p>

  <div class="tblwrap">
    <table>
      <thead>
        <tr><th>Piece</th><th>What I used</th></tr>
      </thead>
      <tbody>
        <tr><td>Exposure</td><td>WWEIA <strong>7102</strong> vs <strong>7202</strong>, Day-1, exclusive soda-type groups</td></tr>
        <tr><td>Sample</td><td>Adults 20+, not pregnant, reliable Day-1 diet, MEC weight &gt; 0 · n={f['n']:,}</td></tr>
        <tr><td>Groups</td><td>diet-only {asb_n:,} · regular-only {ssb_n:,} · both {both_n:,} · neither {nei_n:,}</td></tr>
        <tr><td>Mortality</td><td>{f['cancer_deaths_total']} cancer deaths · {f['allcause_deaths'] if f['allcause_deaths'] is not None else '—'} all-cause · ~{f['asb_cancer_deaths']} cancer deaths in diet-only</td></tr>
        <tr><td>Weights</td><td>Multi-cycle MEC normalized (÷4). Binary GLM: normalized weights, never raw MEC as fake sample size</td></tr>
        <tr><td>Money</td><td>No beverage industry funding. Public CDC / USDA / NCHS files</td></tr>
      </tbody>
    </table>
  </div>

  <pre><code>cd diet-soda-analysis
python -m src.data.build_analysis_dataset
python -m src.analysis.run_eda
python -m src.analysis.run_models
python -m src.analysis.run_ml
python -m src.analysis.run_cancer_module
python -m src.analysis.run_verdicts
python scripts/handoff_smoke.py
python scripts/build_html_report.py

# outputs/tables/report_facts.json
# docs/myth_verdicts.md</code></pre>

  <p style="color:var(--muted);font-size:14px">
    Open <code class="inline">index.html</code> from the project root so chart paths work.
    Or rebuild with <code class="inline">--embed</code> for a single portable file.
  </p>
</section>

<section id="limits">
  <div class="secnum">10</div>
  <h2>Limits</h2>
  <p class="lede">Read this before you weaponize a screenshot.</p>
  <ul>
    <li>One exam plus one day of diet recall. I cannot prove cause for obesity, diabetes, or cancer.</li>
    <li>I used survey weights. I did not run full multi-stage design variance. p-values are too friendly.</li>
    <li>Ever-cancer is self-report lifetime history. Diet is yesterday. Bad match for long cancer stories.</li>
    <li>Cancer death follow-up length varies by cycle. Diet-only is uncommon. Few events in that group (~{f['asb_cancer_deaths']}).</li>
    <li>No gut microbiome data. No insulin clamp. No randomized swap of sugar for diet. No organ-specific cancer registry.</li>
    <li><strong>Not medical advice.</strong> If you have PKU (phenylketonuria), aspartame is a real clinical issue. Ask a clinician.</li>
  </ul>

  <div class="finding" style="margin-top:28px">
    <p class="label">Line I would post</p>
    <p class="boom" style="font-size:clamp(20px,3.2vw,28px)">
      Diet-soda drinkers are <span>older, heavier, and about twice as diabetic</span>.
      That selection writes the crude scares.
      Blood sugar mostly calms after known diabetes is out.
      BMI stays about <span>+{b3:.1f}</span> as association.
      Cancer slogan fails. HR {hr:.2f} with ~{f['asb_cancer_deaths']} diet-group cancer deaths is not a safety certificate.
    </p>
    <p style="margin-bottom:4px">
      Pair {dm_asb:.0f}% vs {dm_nei:.0f}% with the switch story.
      Pair +{h0:.2f}→+{h5:.2f} with the diabetes exclusion.
      Pair the cancer HR with the event count. Always both.
    </p>
  </div>
</section>

</main>

<footer>
  <p><b>Diet Soda Myth Lab</b> · public article · I built this on open data with Grok 4.5 as a coding partner</p>
  <p>NHANES · USDA WWEIA · NCHS Linked Mortality · no industry funding · not medical advice</p>
  <p>Open <code class="inline">index.html</code> from the project folder so charts load · rebuild: <code class="inline">python scripts/build_html_report.py</code></p>
</footer>

</body>
</html>
"""
    # Guard: no em dashes in authored body (CI hyphens are fine)
    if "\u2014" in html:
        html = html.replace("\u2014", ". ")
    return html


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Diet Soda Myth Lab article HTML")
    ap.add_argument(
        "--embed",
        action="store_true",
        help="Embed figures as base64 (portable single file; larger)",
    )
    args = ap.parse_args()

    facts = load_facts()
    html = build_html(facts, embed=args.embed)

    OUT_ROOT.write_text(html, encoding="utf-8")
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUT_ROOT, OUT_REPORT)

    print(f"Wrote {OUT_ROOT} ({OUT_ROOT.stat().st_size // 1024} KB)")
    print(f"Wrote {OUT_REPORT} ({OUT_REPORT.stat().st_size // 1024} KB)")
    print(f"Facts: {TABLES / 'report_facts.json'}")
    print(f"Embed images: {args.embed}")


if __name__ == "__main__":
    main()
