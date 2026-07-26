# The First Shot

**[`first_shot.html`](first_shot.html)** is the *Return to Fire* article exactly as it stood at the
end of its first build session — July 25, 2026 — before the data audit (Round 2) existed. It has
never been edited since, and it never will be. This file explains why it's preserved.

## What happened

One prompt. Aaron pointed at his 2020 wildfire projects — hand-written R and Python from the very
start of his data science career, built after the Tubbs Fire took his family's home — and asked for
a "100x genius mode" rebuild: better data, better modeling, better conclusions, honoring what the
originals did well.

From that single prompt, the session:

1. **Read the old work and took it seriously** — the 3-part trilogy (R/ggplot EDA, leaflet maps,
   sklearn random forests), extracting its goals and intent rather than just its flaws.
2. **Criticized it constructively, with receipts** — the silent Julian-date bug that made every
   fire January 1970, the 94.7% accuracy that was mostly a majority class plus memorized
   coordinates, the climate conclusion resting on one trend line — while cataloguing what a
   self-taught beginner got *right* in 2020: a hard non-toy dataset, interpretation after every
   figure, an Albers projection, and one unprompted flash of leakage awareness.
3. **Rebuilt everything for 2026** — FPA FOD-Attributes (2.3M fires × 308 columns, 4.68 GB raw →
   24 MB of hash-locked caches), negative-binomial trend models with honest verdicts (including a
   REJECTED one), a museum-piece replica of the 2020 model beside temporally-validated honest
   models, static Albers maps with the Tubbs Fire starred, and a 16-section article in the house
   style — every number machine-verified against the results JSONs by `verify_claims.py`.
4. **Found the row.** The 2020 dataset ended in 2015; the fire that started everything wasn't in
   it. The new data reaches 2020. FOD_ID 400015986 — TUBBS, October 8, 2017, 36,807 acres, minimum
   humidity 6.3%, ERC above the 90th local percentile — the data caught up to the story.

Plan → fetch → reduce → four labs → article → verifier: roughly six hours end to end.

## Why it's preserved

Aaron's reaction, verbatim: *"This is very cool. I love the constructive criticism and what I did
well. all one prompt, so save this trail!"* — the thing worth keeping wasn't just the article, but
the *treatment*: the 2020 work honored as a starting line and stress-tested like a peer's, not torn
down.

It's also the **before-picture**. Round 2 puts on the data-engineer hat and audits the corpus
itself (missingness geography, duplicates, coordinate quality, cause-label forensics — see
[`data.html`](data.html)). If that audit ever changes a cache, the live article's numbers update
through the verifier loop — and this frozen copy is the honest record of what the first shot
claimed before anyone looked harder.

## Freeze integrity

```
sha256(first_shot.html) = 106afcc848e6c4c8ce528f64e4bda145386667046319a4f696550e01757f8a4a
```

Frozen 2026-07-25. If the hash above no longer matches the file, the exhibit has been tampered
with and should be restored from git history.

Links: [the live article](index.html) · [the data audit](data.html) · [README](README.md) ·
[the 2020 originals, also unedited](https://along777.github.io/along/projects/wildfiresp1.html)
