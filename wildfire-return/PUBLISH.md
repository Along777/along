# Publishing Return to Fire

Everything in this folder is finished and verified. This file is the checklist for
the commit, written so it can be executed from a fresh session with no context.

## 1. Confirm the build is green

```powershell
cd wildfire-return
.venv\Scripts\python.exe verify_claims.py
```

Expected, and it must exit 0:

```
48 in index.html + 23 in data.html + 59 in modeling.html; 6 verdict chips;
18 retired strings absent; all figures present and referenced; zero JS on
every page; cross-links intact; first_shot freeze intact; all 11 manifest
entries match.
```

That single command checks 130 published numbers against the results JSONs, confirms
no retired number has crept back, re-hashes the frozen `first_shot.html`, and
re-checks all 11 cache hashes.

## 2. Stage the right things

**Do not run `git add .` from the repository root.** `Aaron Long Resume.pdf` sits
untracked there and would be swept into a public commit.

The folder and the portfolio card must go in together, otherwise the card on the
homepage links to a 404:

```bash
git add wildfire-return index.html
```

Then check what is staged before committing:

```bash
git status --short
```

Expect **94 files, about 38.3 MB**: four HTML pages, 18 Python files, four markdown
docs, 43 figures, and the hash-locked caches in `data/`. The 4.68 GB of raw CSVs and
the `.venv` are gitignored and must not appear. The root `index.html` change is the
portfolio card and shows as one modified file.

## 3. Commit

```bash
git commit -m "Return to Fire: six-round AI rebuild of the 2020 wildfire trilogy"
```

## 4. Push

```bash
git push origin master
```

GitHub Pages serves from `master`. After the push:

- the article goes live at https://along777.github.io/along/wildfire-return/
- the seven in-page GitHub links (README, DATA_DICTIONARY, examples, tree roots)
  begin resolving; they 404 until this commit exists
- the homepage card at https://along777.github.io/along/ links to the live page

## Things to know before it is public

- `figures/story_house_before.jpg`, `story_house_after.jpg` and
  `story_house_rebuild.jpg` are family photographs of the house in Santa Rosa.
  They are central to the piece and ship deliberately.
- `first_shot.html` is a frozen exhibit. Its SHA-256 is recorded in `FIRST_SHOT.md`
  and re-checked by `verify_claims.py` on every run. Never edit it. If a future
  round needs to change it, the honest move is a new page, not an edit.
- `data/ca_fires.parquet` is 22.9 MB. It is well under GitHub's limits, but every
  future cache regeneration adds a new blob of that size to history permanently.
- The 21 figures `first_shot.html` displays live in the shared `figures/` directory
  and are not covered by its hash. The freeze guarantees the page, not the images.

## Regenerating anything

The full pipeline, in order, is in [README.md](README.md) under "Follow along".
Nothing on any page is hand-entered: `verify_claims.py` fails the build if a
published number and its results JSON disagree.
