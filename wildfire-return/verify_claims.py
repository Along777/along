from __future__ import annotations

"""Neither article is allowed to disagree with the computation.

Round 1: every load-bearing number in index.html must appear exactly as the
results JSONs format it. Round 2 extends the same contract to data.html (fed by
data_audit.json), adds per-page structural checks (figures exist, zero JS, no
external stylesheets, cross-links present), and verifies the first_shot.html
freeze against the SHA-256 recorded in FIRST_SHOT.md.

Round 3 extends the contract to modeling.html; Round 5 adds retired-string
checks (numbers the red team superseded may not reappear), widens the orphan
check to every generated figure, and verifies all 11 manifest entries.

first_shot.html is deliberately NOT scanned for claims -- it is a frozen
exhibit of what Round 1 said before the audit existed. (Its figure src
attributes are read for existence/orphan checks; the file is never modified.)

Exit 0 = everything verified. Exit 1 = a page and the science disagree.
Run (offline):  python verify_claims.py
"""

import json
import re
import sys

import wildfire as wf

INDEX = (wf.ROOT / "index.html").read_text(encoding="utf-8")
DATA_HTML = (wf.ROOT / "data.html").read_text(encoding="utf-8")
_MODELING_PATH = wf.ROOT / "modeling.html"
MODELING_HTML = _MODELING_PATH.read_text(encoding="utf-8") if _MODELING_PATH.exists() else None


def index_needles() -> list[tuple[str, str]]:
    trend = json.loads((wf.DATA / "trend_lab_results.json").read_text())
    ml = json.loads((wf.DATA / "ml_lab_results.json").read_text())
    eda = json.loads((wf.DATA / "eda_results.json").read_text())
    maps = json.loads((wf.DATA / "maps_summary.json").read_text())
    tubbs = json.loads((wf.DATA / "tubbs_record.json").read_text())["curated"]
    modeling = json.loads((wf.DATA / "modeling_results.json").read_text())["claims"]
    rt_full = json.loads((wf.DATA / "redteam_results.json").read_text())
    rt = rt_full["claims"]
    gz = json.loads((wf.DATA / "generalization.json").read_text())["claims"]
    tc, mc = trend["claims"], ml["claims"]
    return [
        # ---- Round-5b: the final report carries the reviewed modeling numbers
        ("SOR AP (final report)",
         f"{rt['sor_test_ap']:.3f} &plusmn; {rt['sor_seed_sd']:.4f}"),
        ("normalized lift (final report)", f"{rt['lift_point']:.1f}&times;"),
        ("lift CI (final report)", f"[{rt['lift_lo']:.1f}&ndash;{rt['lift_hi']:.1f}]"),
        ("precision at 1% (final report)",
         f"<b>{100 * rt['sor_precision_at_1pct']:.1f}%</b> escalate"),
        ("published BSS (final report)",
         f"<b>&minus;{abs(rt['published_round1_bss']):.2f}</b>"),
        ("rebuilt BSS (final report)", f"+{rt['sor_bss']:.3f}"),
        ("Tubbs OOS percentile (final report)",
         f"{rt['tubbs_oos_percentile']:.0f}th percentile</b>"),
        ("Tubbs calibrated risk (final report)",
         f"<b>{100 * rt['tubbs_calibrated_risk']:.1f}%</b> against a 2.1% base rate"),
        ("Tubbs OOS reference size", f"{rt_full['tubbs_v2']['n_reference']:,}"),
        ("leave-2020-out (final report)",
         f"test score is <b>{rt['leave_2020_out_ap']:.3f}</b>"),
        # ---- Round-6 generalization, condensed into the final report
        ("train AP (final report)", f"<b>{gz['train_ap']:.3f}</b> on the training years"),
        ("val AP (final report)", f"<b>{gz['val_ap']:.3f}</b> on the"),
        ("val AUC (final report)", f"<b>{gz['val_auc']:.3f}</b> and <b>{gz['test_auc']:.3f}</b>"),
        ("shuffle null (final report)", f"AP <b>{gz['null_ap']:.4f}</b>"),
        ("missing negatives (final report)",
         f"<b>{gz['missing_negatives_millions']:.1f} million</b> are the negative class"),
        ("MTBS leak (final report)", f"takes it to <b>{modeling['leak_mtbs_ap']:.3f}</b>"),
        ("red-team findings (final report)", f"<b>{rt['findings_total']} findings</b>"),
        ("CA 2020 burned acres", f"{tc['ca_2020_acres_m']:.2f}M acres"),
        ("CONUS acres multiple", f"{round(tc['conus_acres_x_change'], 1):.1f}&times;"),
        ("CA season days/decade", f"+{tc['ca_season_days_per_decade']:.1f} days per decade"),
        ("top-decile ERC early", f"{100 * tc['top_decile_erc_share_early']:.1f}%"),
        ("top-decile ERC late", f"{100 * tc['top_decile_erc_share_late']:.1f}%"),
        ("human ignition share", f"{100 * tc['human_share_count']:.1f}%"),
        ("natural West acres share", f"{100 * tc['natural_share_acres_west']:.1f}%"),
        ("FL large-fire IRR", f"{tc['fl_large_irr_decade']:.2f}/decade"),
        ("West large-fire IRR", f"{tc['west_large_irr_decade']:.2f}/decade"),
        ("class-G slope", f"+{trend['class_G_national']['theil_slope']:.1f}"),
        ("national counts MK p", f"{trend['trends']['national_all_sizes']['mk_p']:.3f}"),
        ("CA large share of acres", f"{100 * eda['ca_large_share_of_acres']:.0f}%"),
        ("corpus row count", f"{eda['corpus_rows']:,}"),
        ("T1 CA macro-F1", f"{mc['t1_ca_test_macro_f1']:.3f}"),
        ("T1 climatology", f"{mc['t1_ca_climatology_macro_f1']:.3f}"),
        ("T2 macro-F1", f"{mc['t2_ca_test_macro_f1']:.3f}"),
        ("T3 Brier", f"{mc['t3_brier']:.3f}"),
        ("museum campfire acc", f"{100 * mc['museum_campfire_acc']:.1f}%"),
        ("museum campfire majority", f"{100 * mc['museum_campfire_majority']:.1f}%"),
        ("label shuffle F1", f"{mc['label_shuffle_f1']:.3f}"),
        ("random CV F1", f"{mc['protocol_random_cv']:.3f}"),
        ("T1 excluded share", f"{100 * mc['t1_label_excluded_share']:.1f}%"),
        ("Tubbs acres", f"{maps['tubbs']['acres']:,.0f} acres"),
        ("FL large-fire count", str(maps["fl_large_n"])),
        ("FL spring share", f"{100 * maps['fl_spring_share']:.0f}%"),
        ("Tubbs FOD_ID", str(tubbs["fod_id"])),
        ("Tubbs discovery date", tubbs["discovery_date"][:10]),
        ("Tubbs rmin", f"{tubbs['rmin']:.1f}%"),
        ("Tubbs vpd", f"{tubbs['vpd']:.2f} kPa"),
        ("test-era label exclusion",
         f"{100 * eda['modeling_handoff']['ca_label_by_era']['test']['class_missing']:.1f}%"),
        ("train-era label exclusion",
         f"{100 * eda['modeling_handoff']['ca_label_by_era']['train']['class_missing']:.1f}%"),
    ]


def data_needles() -> list[tuple[str, str]]:
    audit = json.loads((wf.DATA / "data_audit.json").read_text())
    c = audit["claims"]
    ak = audit["cause_forensics"]["by_state"]["AK"]
    h = json.loads((wf.DATA / "eda_results.json").read_text())["modeling_handoff"]
    return [
        ("review: test-era exclusion", f"{100 * h['ca_label_by_era']['test']['class_missing']:.1f}%"),
        ("review: train-era exclusion", f"{100 * h['ca_label_by_era']['train']['class_missing']:.1f}%"),
        ("review: 7-class test missing", f"{100 * h['ca_label_by_era']['test']['group_missing']:.1f}%"),
        ("review: CA cache tier-1 dups", f"{h['ca_tier1_dup_rows_in_cache']} tier-1 rows"),
        ("review: FL cache tier-1 dups", f"{h['fl_tier1_dup_rows_in_cache']} in FL"),
        ("rows audited", f"{c['rows_audited']:,}"),
        ("columns audited", f"{c['columns_audited']} columns"),
        ("FOD_ID collisions", f"<b>{c['fod_id_dup_rows']}</b> collisions"),
        ("tier-1 groups", f"{c['t1_dup_groups']} groups"),
        ("tier-1 rows", f"{c['t1_dup_rows']} rows"),
        ("tier-2 rows", f"{c['t2_dup_rows']:,}"),
        ("state agreement", f"{100 * c['state_agreement_rate']:.1f}% agreement"),
        ("size-class mismatches", f"disagree {c['size_class_violations']} times"),
        ("long burns", f"{c['burn_gt400']} fires ran past 400 days"),
        ("whole-degree coords", f"{c['whole_degree_coords']:,}"),
        ("low-precision share", f"{100 * c['coord_low_precision_share']:.1f}% of fires"),
        ("CA general-missing", f"{100 * c['ca_general_missing']:.1f}%"),
        ("FL general-missing", f"{100 * c['fl_general_missing']:.1f}%"),
        ("CA class-missing", f"{100 * c['ca_class_missing']:.1f}%"),
        ("FL class-missing", f"{100 * c['fl_class_missing']:.1f}%"),
        ("AK fire count", f"{ak['n']:,} fires"),
        ("non-CONUS rows", f"{audit['checks']['range_lat']['violations']:,}"),
        ("rows changed", f"{c['rows_changed_by_cleaning_v2']} rows"),
    ]


def modeling_needles() -> list[tuple[str, str]]:
    """Round-3 modeling claims + Round-4 red-team claims, formatted exactly as
    modeling.html prints them. The Round-4 re-key made the tuned HGB the
    system of record; LightGBM's numbers remain as reported-not-selected."""
    m = json.loads((wf.DATA / "modeling_results.json").read_text())
    c = m["claims"]
    rt_full = json.loads((wf.DATA / "redteam_results.json").read_text())
    rt = rt_full["claims"]
    cal = rt_full["calibration_v2"]["hgb"]
    tk = cal["topk"]
    fz = rt_full["fair_zoo"]["families"]
    g = json.loads((wf.DATA / "generalization.json").read_text())
    gz, gg = g["claims"], g["generalization"]
    return [
        # ---- Round-6 generalization section
        ("train AP", f'<td class="num"><b>{gz["train_ap"]:.3f}</b></td>'),
        ("train AUC", f'<td class="num"><b>{gz["train_auc"]:.3f}</b></td>'),
        ("val AUC", f'<td class="num"><b>{gz["val_auc"]:.3f}</b></td>'),
        ("test AUC (SoR)", f'<td class="num"><b>{gz["test_auc"]:.3f}</b></td>'),
        ("train lift", f'{gg["train"]["lift"]:.1f}&times;'),
        ("train-test AP gap", f"a difference of <b>{gz['train_test_ap_gap']:.3f}</b>"),
        ("held-out AUC agreement",
         f"they score <b>{gz['val_auc']:.3f}</b> and <b>{gz['test_auc']:.3f}</b> ROC-AUC"),
        ("shuffle null AP", f"AP <b>{gz['null_ap']:.4f}</b> against a base rate"),
        ("shuffle null AUC", f"ROC-AUC <b>{gz['null_auc']:.3f}</b>"),
        ("cell-day universe", f"<b>{gz['cell_days_millions']:.1f} million</b> cell-days"),
        ("missing negatives", f"<b>{gz['missing_negatives_millions']:.1f} million</b> cell-days"),
        ("ignition rate per cell-day",
         f"<b>{100 * gz['ignition_rate_per_cell_day']:.1f}%</b> of those cell-days"),
        ("days with fire", f"<b>{gz['days_with_fire_pct']:.1f}%</b> of the"),
        ("features engineered", f"{c['n_features_engineered']} engineered features"),
        ("LightGBM zoo AP", f"{c['test_ap']:.3f}"),
        ("AP interval", f"[{c['test_ap_lo']:.3f}, {c['test_ap_hi']:.3f}]"),
        ("test prevalence", f"{100 * c['prevalence_test']:.2f}%"),
        ("Round-1 BSS replica", f"{c['round1_bss']:+.2f}"),
        ("precision at 1%", f"{100 * c['precision_at_1pct']:.1f}%"),
        ("recall ceiling", f"{100 * c['ceiling_recall_at_1pct']:.1f}%"),
        ("tuning gain", f"{c['tuning_gain_ap']:+.4f}"),
        ("feature gain", f"{c['ablation_gain_ap']:+.4f}"),
        ("leak AP honest", f"{c['leak_honest_ap']:.3f}"),
        ("leak AP with MTBS id", f"{c['leak_mtbs_ap']:.3f}"),
        ("CA to FL transfer", f"{c['transfer_ca_to_fl_ap']:.3f}"),
        ("Tubbs mixed-sample percentile", f"{c['tubbs_percentile']:.0f}th"),
        # ---- Round-4 red team (the re-keyed headline + experiments)
        ("SOR seed AP", f"Seed sd {rt['sor_seed_sd']:.4f} (HGB)"),
        ("SOR seed mean AP", f"{rt['sor_test_ap']:.3f}"),
        ("SOR calibrated BSS", f"{rt['sor_bss']:+.3f}"),
        ("no-geography AP", f"{rt['no_geography_ap']:.3f}"),
        # formatted from the full-precision value: the claims block's 0.1265
        # re-rounds to 0.127, a double-rounding trap the page must not inherit
        ("yesterday-knowledge AP",
         f"AP <b>{rt_full['experiments']['yesterday_knowledge']['ap']:.3f}</b>"),
        ("same-day feature delta", f"{rt['same_day_delta']:+.4f}"),
        ("dup-flags delta", f"{rt['dup_delta']:+.4f}"),
        ("leave-2020-out AP", f"<b>{rt['leave_2020_out_ap']:.3f}</b>. That is the number"),
        ("2020 positive share", f"{100 * rt['share_positives_2020']:.0f}%"),
        ("normalized test lift", f"{rt['lift_point']:.1f}&times;"),
        ("low-pop tercile lift", f"{rt['low_pop_lift']:.1f}&times;"),
        ("Tubbs OOS percentile", f"{rt['tubbs_oos_percentile']:.0f}th"),
        ("published Round-1 BSS", f"{rt['published_round1_bss']:+.2f}".replace("-", "-")),
        ("NB-t earned null", f"p = {rt['nb_t_optuna_p']:.2f}"),
        # ---- Round-5 coverage: the red-team claims that were printed but unguarded
        ("LGBM seed AP", f"LightGBM's {rt['lgbm_test_ap']:.3f} &plusmn;"),
        ("LGBM seed sd", f"{rt['lgbm_seed_sd']:.4f} is reported, not selected"),
        ("SOR calibrated slope", f"slope {rt['sor_slope']:.2f}"),
        ("SOR precision at 1pct", f"{100 * rt['sor_precision_at_1pct']:.1f}%</b> escalate"),
        ("tail reliability",
         f"the model says {100 * rt['tail_predicted']:.1f}% against an observed "
         f"{100 * rt['tail_observed']:.1f}%"),
        ("Tubbs calibrated risk",
         f"calibrated escalation risk of <b>{100 * rt['tubbs_calibrated_risk']:.1f}%</b>"),
        ("lift CI", f"[{rt['lift_lo']:.1f}&ndash;{rt['lift_hi']:.1f}]"),
        ("high-pop tercile lift", f"{rt['high_pop_lift']:.1f}&times;"),
        ("embargoed history delta", f"&Delta;AP {rt['embargo_delta']:+.4f}"),
        ("red-team findings", f"<b>{rt['findings_total']} findings</b>"),
        ("gated touches",
         f"<b>{rt_full['test_touch_accounting']['gated_selection_touches']}</b> (one per zoo family)"),
        ("isotonic ECE", f"to <b>{cal['test']['ece']:.4f}</b>"),
        ("isotonic intercept", f"intercept from 0.267 to {cal['test']['intercept']:.3f}"),
        ("isotonic AP cost", f"(0.124 &rarr; {cal['test']['ap']:.3f} on the seed-42 fit)"),
        ("topk 0.5% row (isotonic)",
         f"{100 * tk[0]['precision']:.1f}%</td><td class=\"num\">{100 * tk[0]['recall']:.1f}%"),
        ("topk 5% row (isotonic)",
         f"{100 * tk[3]['precision']:.1f}%</td><td class=\"num\">{100 * tk[3]['recall']:.1f}%"),
        ("fair zoo: random forest",
         f"random forest: {fz['random_forest']['ap']:.3f} (was {fz['random_forest']['round3_ap']:.3f}), "
         f"BSS {fz['random_forest']['bss']:+.2f}"),
        ("fair zoo: extra trees",
         f"extra trees: {fz['extra_trees']['ap']:.3f} (was {fz['extra_trees']['round3_ap']:.3f}), "
         f"BSS {fz['extra_trees']['bss']:+.2f}"),
        ("fair zoo: ridge logistic",
         f"ridge logistic: {fz['ridge_logistic']['ap']:.3f} (was {fz['ridge_logistic']['round3_ap']:.3f}), "
         f"BSS {fz['ridge_logistic']['bss']:+.2f}"),
    ]


# Strings the Round-4/5 reviews RETIRED. They must never reappear on the pages
# listed here; index.html joins this table when the Round-5b final report lands.
# "0.131" is handled separately (label-proximity) because the number legitimately
# remains as LightGBM's reported-not-selected score and as an unrelated CI bound.
RETIRED: dict[str, list[tuple[str, str]]] = {
    "modeling.html": [
        ("old normalized lift", "6.3x"),
        ("test-selected sigmoid BSS", "+0.054"),
        ("sigmoid presented as chosen", "sigmoid map fitted"),
        ("champion framing", "The champion is"),
        ("pre-amendment protocol", "touched once per model"),
        ("old tuner-spread verdict", "within seed noise</b> (seed sd is"),
        ("mixed-sample Tubbs headline", "96th percentile</b> among all California ignitions"),
        ("wrong family count", "Seven model families"),
        ("wrong ablation baseline", "30-column baseline"),
    ],
    "data.html": [
        ("two-page verifier claim", "BOTH pages"),
    ],
    "index.html": [
        ("old MK p", "p=0.539"),
        ("old human share", "84.5%"),
        ("old cause macro-F1", "0.870"),
        ("old giant-fire slope", "+4 giant fires"),
        ("the false calibration claim", "its probabilities are calibrated ("),
        ("the mislabeled photo caption", "rebuilt after moving in"),
        ("stale cache size (entity)", "~24&nbsp;MB"),
        ("stale cache size (plain)", "~24 MB"),
    ],
}


def retired_strings(page_name: str, html: str) -> list[str]:
    problems = [f"{page_name} RETIRED STRING PRESENT: {label} -- '{s}'"
                for label, s in RETIRED.get(page_name, []) if s in html]
    if page_name == "modeling.html":
        # 0.131 may appear only as LightGBM's labeled score or inside the
        # defaults' CI [0.091, 0.131] -- never as an unlabeled headline.
        for i in [m.start() for m in re.finditer(r"0\.131", html)]:
            ctx = html[max(0, i - 90):i + 90]
            if not ("reported, not selected" in ctx or "lightgbm tuned" in ctx
                    or "[0.091, 0.131]" in ctx):
                problems.append(f"{page_name}: unlabeled 0.131 at offset {i} -- the re-keyed "
                                f"headline is 0.126; 0.131 must carry its label")
    return problems


def structural(page_name: str, html: str, must_link: list[str]) -> list[str]:
    problems = []
    for src in re.findall(r'src="(figures/[^"]+)"', html):
        if not (wf.ROOT / src).exists():
            problems.append(f"{page_name}: IMAGE MISSING {src}")
    if re.search(r"<script", html, re.I):
        problems.append(f"{page_name}: SCRIPT TAG found -- contractually zero-JS")
    if re.search(r"<link[^>]+stylesheet", html, re.I):
        problems.append(f"{page_name}: EXTERNAL STYLESHEET found")
    if re.search(r"&#10696;[A-Z0-9_]+&#10697;", html):
        problems.append(f"{page_name}: unfilled placeholder tokens remain")
    # House style since Round 6: no em dashes. Entity or literal, both fail.
    n_dash = len(re.findall(r"&mdash;|—", html))
    if n_dash:
        problems.append(f"{page_name}: {n_dash} em dash(es) present; house style forbids them")
    for link in must_link:
        if f'href="{link}"' not in html:
            problems.append(f"{page_name}: missing cross-link to {link}")
    return problems


def orphan_figures(pages: list[tuple[str, str]]) -> list[str]:
    """Every generated figure must be shown by some page (first_shot.html's
    references count -- it is read here for src attributes only, never scanned
    for claims and never modified)."""
    referenced = set()
    for _, html in pages:
        referenced.update(re.findall(r'src="(figures/[^"]+)"', html))
    first_shot = (wf.ROOT / "first_shot.html").read_text(encoding="utf-8")
    referenced.update(re.findall(r'src="(figures/[^"]+)"', first_shot))
    problems = [f"first_shot.html IMAGE MISSING {src}"
                for src in re.findall(r'src="(figures/[^"]+)"', first_shot)
                if not (wf.ROOT / src).exists()]
    for p in sorted(wf.FIG.glob("*.png")):
        rel = f"figures/{p.name}"
        if rel not in referenced:
            problems.append(f"ORPHAN FIGURE: {rel} is generated but no page shows it")
    return problems


def check_freeze() -> list[str]:
    md = (wf.ROOT / "FIRST_SHOT.md").read_text(encoding="utf-8")
    m = re.search(r"sha256\(first_shot\.html\)\s*=\s*([0-9a-f]{64})", md)
    if not m:
        return ["FIRST_SHOT.md: recorded sha256 not found"]
    actual = wf.sha256_of(wf.ROOT / "first_shot.html")
    if actual != m.group(1):
        return [f"FREEZE BROKEN: first_shot.html sha256 {actual[:12]}... != recorded {m.group(1)[:12]}..."]
    return []


def main() -> int:
    failures: list[str] = []
    counts: dict[str, int] = {}

    # Page table: name -> (html, needle builder, required cross-links).
    # first_shot.html is deliberately absent -- it is a frozen exhibit, checked
    # by hash only, never scanned for claims that have since been corrected.
    pages: list[tuple] = [
        ("index.html", INDEX, index_needles, ["data.html"]),
        ("data.html", DATA_HTML, data_needles, ["index.html"]),
    ]
    if MODELING_HTML is not None:
        pages.append(("modeling.html", MODELING_HTML, modeling_needles,
                      ["index.html", "data.html"]))
        pages[0] = ("index.html", INDEX, index_needles, ["data.html", "modeling.html"])
        pages[1] = ("data.html", DATA_HTML, data_needles, ["index.html", "modeling.html"])

    for name, html, needle_fn, links in pages:
        try:
            needles = needle_fn()
        except FileNotFoundError as e:
            failures.append(f"{name}: results JSON missing ({e})")
            continue
        except KeyError as e:
            failures.append(f"{name}: results JSON has no claim {e}")
            continue
        counts[name] = len(needles)
        for label, needle in needles:
            if needle not in html:
                failures.append(f"{name} CLAIM MISSING: {label} -- expected '{needle}'")
        failures += structural(name, html, links)
        failures += retired_strings(name, html)

    trend = json.loads((wf.DATA / "trend_lab_results.json").read_text())
    chip = {"CONFIRMED": "v-win", "SPLIT": "v-split", "REJECTED": "v-loss"}
    for hid, h in trend["hypotheses"].items():
        want = f'class="verdict {chip[h["verdict"]]}">{h["verdict"]}<'
        if want not in INDEX:
            failures.append(f"index.html VERDICT MISMATCH: {hid} should render {h['verdict']}")

    failures += orphan_figures([(n, h) for n, h, _, _ in pages])
    failures += check_freeze()

    try:
        wf.verify_manifest("national_annual.csv", "national_monthly.csv", "ca_fires.parquet",
                           "fl_fires.parquet", "conus_grid.parquet", "tubbs_record.json",
                           "reduce_report.json", "geo/us_states_20m.json", "recent_annual.csv",
                           "recent_annual_meta.json", "dup_flags.parquet",
                           strict=True)
    except SystemExit as e:
        failures.append(f"MANIFEST: {e}")

    if failures:
        print("\n".join(failures))
        print(f"\n{len(failures)} failure(s).")
        return 1
    summary = " + ".join(f"{n} in {p}" for p, n in counts.items())
    n_retired = sum(len(v) for v in RETIRED.values())
    print(f"{summary}; {len(trend['hypotheses'])} verdict chips; {n_retired} retired strings "
          f"absent; all figures present and referenced; zero JS on every page; cross-links "
          f"intact; first_shot freeze intact; all 11 manifest entries match.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
