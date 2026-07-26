from __future__ import annotations

"""Streaming validation primitives for the full-corpus data audit.

Hand-rolled on purpose. pandera and great-expectations validate one in-memory
DataFrame; this audit accumulates cross-chunk, cross-file aggregates (per-state
missingness matrices, per-year vocabularies, global ID uniqueness, per-year
duplicate buckets) that those frameworks don't provide -- we would be writing
the accumulator layer either way, and a plain dict is the exact contract
verify_claims.py consumes. Zero new dependencies, no pandas-3 compat risk.

Protocol: every check implements update(df, year) called once per chunk, and
finalize() -> JSON-ready dict. Statuses are "pass" / "info" / "warn" -- the
audit REPORTS, it never gates. Examples are capped at 5 per check.
"""

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

MAX_EXAMPLES = 5

SENTINEL_STRINGS = ["", "-9999", "-9999.0", "-999", "-999.0", "32767", "32767.0",
                    "NA", "N/A", "nan", "NULL", "None"]

# LatLong_State carries full names ("California"), not USPS codes.
STATE_NAME_TO_USPS = {
    "Alabama": "AL", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
    "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC", "Florida": "FL",
    "Georgia": "GA", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
    "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN",
    "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY", "Alaska": "AK", "Hawaii": "HI",
    "Puerto Rico": "PR",
}


def _status(rate: float, warn_at: float) -> str:
    if rate == 0:
        return "pass"
    return "warn" if rate >= warn_at else "info"


class NumericRangeCheck:
    """Out-of-range scan on a string column coerced in-pass (audit reads raw text)."""

    def __init__(self, check_id: str, col: str, lo: float, hi: float, warn_at: float = 0.001):
        self.check_id, self.col, self.lo, self.hi, self.warn_at = check_id, col, lo, hi, warn_at
        self.evaluated = 0
        self.violations = 0
        self.by_year: Counter = Counter()
        self.examples: list = []

    def update(self, df: pd.DataFrame, year: int) -> None:
        v = pd.to_numeric(df[self.col], errors="coerce")
        ok = v.notna()
        bad = ok & ~v.between(self.lo, self.hi)
        self.evaluated += int(ok.sum())
        n_bad = int(bad.sum())
        if n_bad:
            self.violations += n_bad
            self.by_year[year] += n_bad
            for fid, val in zip(df.loc[bad, "fod_id"].head(MAX_EXAMPLES),
                                v[bad].head(MAX_EXAMPLES)):
                if len(self.examples) < MAX_EXAMPLES:
                    self.examples.append({"fod_id": int(fid), "value": float(val)})

    def finalize(self) -> dict:
        rate = self.violations / self.evaluated if self.evaluated else 0.0
        return {"id": self.check_id, "desc": f"{self.col} in [{self.lo}, {self.hi}]",
                "status": _status(rate, self.warn_at), "evaluated": self.evaluated,
                "violations": self.violations, "rate": round(rate, 6),
                "by_year": {str(k): v for k, v in sorted(self.by_year.items())},
                "examples": self.examples}


class SentinelScan:
    """Pre-coercion sentinel-string census + suspicious exact zeros. Report-only."""

    def __init__(self, cols: list[str], zero_suspect_cols: list[str]):
        self.cols = cols
        self.zero_suspect = zero_suspect_cols
        self.hits: dict[str, Counter] = defaultdict(Counter)
        self.zeros: Counter = Counter()
        self.evaluated = 0

    def update(self, df: pd.DataFrame, year: int) -> None:
        self.evaluated += len(df)
        for col in self.cols:
            s = df[col]
            present = s[s.notna()].astype(str).str.strip()
            if len(present) == 0:
                continue
            counts = present[present.isin(SENTINEL_STRINGS)].value_counts()
            for k, v in counts.items():
                self.hits[col][k or "<empty>"] += int(v)
        for col in self.zero_suspect:
            v = pd.to_numeric(df[col], errors="coerce")
            self.zeros[col] += int((v == 0).sum())

    def finalize(self) -> dict:
        found = {c: dict(cnt) for c, cnt in self.hits.items() if cnt}
        return {"id": "sentinel_scan", "desc": "sentinel strings beyond the handled {-9999,-999}",
                "status": "info" if found else "pass", "evaluated": self.evaluated,
                "violations": int(sum(sum(c.values()) for c in self.hits.values())),
                "rate": None, "sentinels_found": found,
                "suspicious_exact_zeros": dict(self.zeros),
                "notes": "zeros are physically plausible for some columns; report-only"}


class VocabPerYear:
    """Label-set stability of a categorical column across the 29 years."""

    def __init__(self, col: str):
        self.col = col
        self.per_year: dict[int, set] = defaultdict(set)

    def update(self, df: pd.DataFrame, year: int) -> None:
        s = df[self.col]
        self.per_year[year].update(s[s.notna()].astype(str).unique().tolist())

    def finalize(self) -> dict:
        years = sorted(self.per_year)
        all_labels = sorted(set().union(*self.per_year.values())) if years else []
        spans = {lab: [min(y for y in years if lab in self.per_year[y]),
                       max(y for y in years if lab in self.per_year[y])] for lab in all_labels}
        stable = all(self.per_year[y] == self.per_year[years[0]] for y in years) if years else True
        return {"id": f"vocab_{self.col}", "desc": f"label vocabulary of {self.col} across years",
                "status": "pass" if stable else "info", "labels_all": all_labels,
                "label_year_span": spans, "stable": stable}


class UniqueCheck:
    """Global uniqueness of an int64 id column (numpy accumulation, ~18 MB total)."""

    def __init__(self, col: str = "fod_id"):
        self.col = col
        self.parts: list[np.ndarray] = []

    def update(self, df: pd.DataFrame, year: int) -> None:
        self.parts.append(df[self.col].to_numpy(dtype="int64", copy=True))

    def finalize(self) -> dict:
        ids = np.concatenate(self.parts) if self.parts else np.array([], dtype="int64")
        uniq, counts = np.unique(ids, return_counts=True)
        dup_mask = counts > 1
        dup_rows = int((counts[dup_mask] - 1).sum())
        examples = [{"fod_id": int(u), "occurrences": int(c)}
                    for u, c in zip(uniq[dup_mask][:MAX_EXAMPLES], counts[dup_mask][:MAX_EXAMPLES])]
        return {"id": f"unique_{self.col}", "desc": f"{self.col} is globally unique",
                "status": "pass" if dup_rows == 0 else "warn", "evaluated": int(ids.size),
                "violations": dup_rows, "rate": round(dup_rows / ids.size, 8) if ids.size else 0.0,
                "duplicate_ids": int(dup_mask.sum()), "examples": examples}


class MissingnessMatrix:
    """Not-null counts by state and by year for the curated + audit columns."""

    def __init__(self, cols: list[str]):
        self.cols = cols
        self.corpus_notna: Counter = Counter()
        self.corpus_total = 0
        self.by_state_notna: dict[str, Counter] = defaultdict(Counter)
        self.by_state_total: Counter = Counter()
        self.by_year_notna: dict[int, Counter] = defaultdict(Counter)
        self.by_year_total: Counter = Counter()

    def update(self, df: pd.DataFrame, year: int) -> None:
        n = len(df)
        self.corpus_total += n
        self.by_year_total[year] += n
        notna = df[self.cols].notna().sum()
        for col, v in notna.items():
            self.corpus_notna[col] += int(v)
            self.by_year_notna[year][col] += int(v)
        for state, g in df.groupby("state", dropna=False):
            st = str(state)
            self.by_state_total[st] += len(g)
            for col, v in g[self.cols].notna().sum().items():
                self.by_state_notna[st][col] += int(v)

    def finalize(self) -> dict:
        def share(notna: int, total: int) -> float:
            return round(1 - notna / total, 4) if total else 1.0
        return {
            "columns": self.cols,
            "corpus": {c: share(self.corpus_notna[c], self.corpus_total) for c in self.cols},
            "by_year": {str(y): {c: share(cnt[c], self.by_year_total[y]) for c in self.cols}
                        for y, cnt in sorted(self.by_year_notna.items())},
            "by_state": {st: {c: share(cnt[c], self.by_state_total[st]) for c in self.cols}
                         for st, cnt in sorted(self.by_state_notna.items())},
            "rows_total": self.corpus_total,
        }


class CrossFieldRule:
    """Generic cross-field rule: fn(df) -> (evaluated_mask, violation_mask, detail_series|None)."""

    def __init__(self, check_id: str, desc: str, fn, warn_at: float = 0.001, notes: str = ""):
        self.check_id, self.desc, self.fn, self.warn_at, self.notes = check_id, desc, fn, warn_at, notes
        self.evaluated = 0
        self.violations = 0
        self.by_year: Counter = Counter()
        self.by_year_eval: Counter = Counter()
        self.detail: Counter = Counter()
        self.examples: list = []

    def update(self, df: pd.DataFrame, year: int) -> None:
        ok, bad, detail = self.fn(df)
        self.evaluated += int(ok.sum())
        self.by_year_eval[year] += int(ok.sum())
        n_bad = int(bad.sum())
        if n_bad:
            self.violations += n_bad
            self.by_year[year] += n_bad
            if detail is not None:
                for k, v in detail[bad].value_counts().head(20).items():
                    self.detail[str(k)] += int(v)
            for fid in df.loc[bad, "fod_id"].head(MAX_EXAMPLES):
                if len(self.examples) < MAX_EXAMPLES:
                    self.examples.append({"fod_id": int(fid), "year": year})

    def finalize(self) -> dict:
        rate = self.violations / self.evaluated if self.evaluated else 0.0
        out = {"id": self.check_id, "desc": self.desc, "status": _status(rate, self.warn_at),
               "evaluated": self.evaluated, "violations": self.violations,
               "rate": round(rate, 6), "by_year": {str(k): v for k, v in sorted(self.by_year.items())},
               "by_year_eval": {str(k): v for k, v in sorted(self.by_year_eval.items())},
               "examples": self.examples}
        if self.detail:
            out["detail_top"] = dict(self.detail.most_common(20))
        if self.notes:
            out["notes"] = self.notes
        return out


class NearDupScan:
    """Two-tier near-duplicate candidate scan, scoped per year (state freed at year end).

    Tier 1 (headline): same discovery date, coords rounded to 3 dp (~110 m),
    fire_size >= 10 acres (class C+), sizes within 10% relative tolerance,
    different FOD_ID. Tier 2 (ceiling, quarantined): same key, exact size, any
    size -- dominated by legitimate batch-reported small burns.
    Disposition is pre-registered: FLAG ONLY, never dropped.
    """

    RULE_TEXT = ("tier1: same discovery date + coords rounded 3dp (~110m) + fire_size >= 10 ac "
                 "+ sizes within 10% relative tolerance + different FOD_ID; "
                 "tier2 (ceiling): same key + exact size match, any size. Flag-only.")

    def __init__(self):
        self.buckets: dict[tuple, list] = defaultdict(list)
        self.t1_groups = 0
        self.t1_rows = 0
        self.t1_by_year: Counter = Counter()
        self.t1_system_pairs: Counter = Counter()
        self.t2_groups = 0
        self.t2_rows = 0
        self.t2_by_year: Counter = Counter()
        self.examples: list = []
        # flagged FOD_IDs, kept so the runner can write the dup_flags sidecar
        # (modeling joins on it for sensitivity checks; rows are never dropped)
        self.t1_ids: list[int] = []
        self.t2_ids: list[int] = []

    def update(self, df: pd.DataFrame, year: int) -> None:
        lat = pd.to_numeric(df["lat_raw"], errors="coerce").round(3)
        lon = pd.to_numeric(df["lon_raw"], errors="coerce").round(3)
        sub = pd.DataFrame({"date": df["discovery_date_str"], "lat": lat, "lon": lon,
                            "fod_id": df["fod_id"], "size": pd.to_numeric(df["fire_size"], errors="coerce"),
                            "source": df["source"]}).dropna(subset=["date", "lat", "lon", "size"])
        for key, fid, size, source in zip(zip(sub["date"], sub["lat"], sub["lon"]),
                                          sub["fod_id"], sub["size"], sub["source"]):
            self.buckets[key].append((int(fid), float(size), str(source)))

    def year_end(self, year: int) -> None:
        for key, rows in self.buckets.items():
            if len(rows) < 2:
                continue
            # tier 2: exact-size subgroups
            by_size: dict[float, list] = defaultdict(list)
            for r in rows:
                by_size[r[1]].append(r)
            for size, grp in by_size.items():
                if len(grp) > 1:
                    self.t2_groups += 1
                    self.t2_rows += len(grp)
                    self.t2_by_year[year] += len(grp)
                    self.t2_ids.extend(r[0] for r in grp)
            # tier 1: class-C+ rows, 10% relative size tolerance, pairwise
            big = [r for r in rows if r[1] >= 10.0]
            if len(big) >= 2:
                flagged = set()
                pair_types = set()
                for i in range(len(big)):
                    for j in range(i + 1, len(big)):
                        a, b = big[i], big[j]
                        if a[0] == b[0]:
                            continue
                        if abs(a[1] - b[1]) / max(a[1], b[1]) <= 0.10:
                            flagged.update((a[0], b[0]))
                            pair = tuple(sorted((a[2], b[2])))
                            pair_types.add(pair)
                if flagged:
                    self.t1_groups += 1
                    self.t1_rows += len(flagged)
                    self.t1_by_year[year] += len(flagged)
                    self.t1_ids.extend(flagged)
                    for pair in pair_types:
                        kind = "within-system" if pair[0] == pair[1] else "cross-system"
                        self.t1_system_pairs[f"{kind}: {pair[0]} x {pair[1]}"] += 1
                    if len(self.examples) < MAX_EXAMPLES:
                        self.examples.append({"year": year, "date": key[0],
                                              "lat": key[1], "lon": key[2],
                                              "fod_ids": sorted(flagged),
                                              "sizes": [r[1] for r in big if r[0] in flagged]})
        self.buckets.clear()

    def finalize(self) -> dict:
        return {"id": "near_duplicates", "desc": "near-duplicate candidate scan (flag-only)",
                "status": "info" if self.t1_rows else "pass",
                "rule_text": self.RULE_TEXT,
                "tier1": {"groups": self.t1_groups, "rows": self.t1_rows,
                          "by_year": {str(k): v for k, v in sorted(self.t1_by_year.items())},
                          "by_system_pair": dict(self.t1_system_pairs.most_common(15)),
                          "examples": self.examples},
                "tier2": {"groups": self.t2_groups, "rows": self.t2_rows,
                          "by_year": {str(k): v for k, v in sorted(self.t2_by_year.items())},
                          "notes": "ceiling, not a dup count: batch-reported small burns at "
                                   "shared coordinates are expected and legitimate"}}
