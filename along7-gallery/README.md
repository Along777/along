# A Database With Good Lighting

The build story of [**along7gallery.xyz**](https://along7gallery.xyz): a personal NFT art collection
turned into a permanent, self-hosted museum. 3,735 works by 839 artists across six blockchains, every
original file rescued from the source the artist declared, every ownership history replayed from the
ledger, and 99 automated checks standing between the database and the public. Built over eight weeks
in the summer of 2026 by one data scientist and an AI.

**Article:** https://along777.github.io/along/along7-gallery/

This folder is the write-up plus the art: the master banner of the collection and the fifteen works
in it, each one placed with its artist's knowledge. The production pipeline, the database, and the
curation layer stay private; the method and the numbers are all here.

## The problem, in one paragraph

An NFT does not store the art. The token is a permanent line on a blockchain; the picture it points to
is a guest on IPFS, Arweave, an artist's web host, or a marketplace cache, and any of those can go dark.
When the pipeline first pulled the original file for every piece in the collection, twelve were already
gone and nine survived only as a marketplace thumbnail. That was the whole argument for the project,
found inside the collection that was supposed to be safe.

## The process, in rounds

| When | What happened | The receipt |
|---|---|---|
| **Jul 9 to 10 · day one** | Empty folder, an OpenSea key, one prompt. 893 NFTs across 339 contracts catalogued in the first hour; a random 25-work pilot proved `tokenURI → metadata → original file` across IPFS, Arweave, and artist servers; 18,110 on-chain events from three sources; 893 originals (1,078 files, 15.4 GB); live on Cloudflare R2 by hour fourteen | The pilot review caught the marketplace serving AVIF image stubs labeled as video (14 purged). A hostile review under two hats found 17 holes; all fixed before the full push |
| **Jul 11 to 22 · the wiki run** | Artist rooms, a timeline of true collecting dates traced through the hot wallet, marketplace forensics from transaction receipts, a globe of artist locations, sixteen stories, the collector's X archive mined for what artists said at the time | The audit harness grew from 21 checks to 74. Two full reviews written as reports, a truth audit and an end-of-day "good night review," before the next feature was allowed in |
| **Jul 17 to 18 · the vault upgrade** | 987 → 1,708 works, one chain → four, 286 → 420 artists. A chain-first recount recovered works the marketplace API had silently dropped. First 3-2-1 backup | A cleanup script deleted 890 legitimate files; the audit failed the build (2,255 broken references) and the day-old backup restored everything in minutes. True orphans: 70 files, 33 MB |
| **Jul 21 to 22 · hardening** | `default-src 'self'` CSP, artist-authored interactive works sandboxed at the edge, JSON-LD on every page, a CC0 dataset, llms.txt, robots.txt welcoming AI crawlers on purpose | 80 checks. A "max auditor" pass walked 2,979 pages, found zero defects, and said so |
| **Aug 13 to 14 · the TV** | A `/tv` page that plays the whole collection (films stream a 1080p transcode tier, gifs loop, live-code pieces run in sandboxed iframes) plus a local tool for a Samsung Frame's Art Mode | 358 transcodes, 2.35 GB, 0 corrupt. Adversarial review caught 12 real bugs before the first deploy and 4 before the second |
| **Aug 26 to 29 · Tezos** | The other half of the collection from a second chain: 1,958 works, ~40 GB, via TzKT and objkt. 476 creators got rooms; cross-chain identities folded by handle, website, and name. A complete Spanish mirror built, audited to 0 FAIL, and parked the same day | Public IPFS gateways throttled the bulk run after ~270 files and denylisted whole artists. Slower waves, gateway rotation, and objkt's CDN as a last door: 0 failed downloads |
| **Sep 2 · fresh eyes** | A new model (Claude Fable 5.1) reviewed five fronts with no memory of having written them. New `factcheck` stage re-verifies chain facts into a ledger; prose numbers became build-time tokens; build time 916 s → ~100 s | 26 bios and every collection-scale count in the essays were stale. An Etherscan rate-limit response had been indistinguishable from a reverted call. "160 duplicate events" were 3. 99 checks, 0 FAIL |

## The headlines

| Question | Answer, with the number |
|---|---|
| How much is archived from the artist's own source? | **99.5%** of 3,735 works. IPFS 3,251 files · artist servers 651 · Arweave 326 · fully on-chain 22 · marketplace cache only **127** (labeled as such) |
| How is a file verified? | Type from **magic bytes**, never the Content-Type header; stored with **SHA-256**; sub-20 KB CDN stubs posing as originals fail the audit |
| How is provenance known? | **80,161** transfer and sale events from two independent sources; **3,652** verified mint records; **3,003** acquisitions traced to the settling marketplace via transaction receipts |
| Can it show the bidding? | **740** bids across **215** auctions, read from calldata (SuperRare, Transient, and one sealed-bid drop: 202 bids from 73 collectors). Houses that bid off-chain leave no trail, and the site says so |
| Does the vault still hold everything? | Custody check on all 3,735 works, offline from the ledger: **0** have left. Multi-edition tokens use a net-balance test (the naive version raised 1,249 false alarms) |
| Is authorship proven? | By `getcontractcreation` (deployer == artist wallet), never by name match: **462** artist contracts, **0** mismatches |
| What does it cost to host? | Static site on Cloudflare R2, ~64 GB, about **a dollar a month**, zero egress fees |
| What gates a deploy? | **99** named audit checks with **0** failures allowed, plus a grep that must find the private wallet **0** times in the built site |

## Mental model

```
py -m vault sync                 # inventory the wallet (captures per-token collection)
py -m vault metadata --all       # each token's own metadata document        [OpenSea / tokenURI]
py -m vault artists --all        # collections + artist accounts
py -m vault provenance --all     # events: marketplace index + chain ledger  [OpenSea + Etherscan / TzKT]
py -m vault enrich               # derive collected_at / collected_from / sales / classes
py -m vault origin               # mint + acquire venue                       (MUST run after enrich)
py -m vault market               # receipt forensics: which marketplace really settled it
py -m vault editions             # true per-token edition size from the contract
py -m vault bids --platform auction   # reconstruct the bidding from calldata
py -m vault download --all       # originals: gateway rotation, sha-256, magic bytes
py -m vault site --base-url https://along7gallery.xyz     # 5,925 pages in ~100 s
py -m vault factcheck            # re-verify chain facts into a ledger (custody, ownerOf, deployer, editions, ENS)
py -m vault audit                # ~100 checks; 0 FAIL required to ship
py -m vault deploy               # rclone sync of site/ ONLY, after the privacy grep
```

One SQLite file holds it all: works, contracts, collections, artists, events, sales, market events,
media files, profiles, fact checks, page hashes. The attribution chain is a join
(`nfts → contracts → collections → artists`), a per-token collection field beats the contract-level one
so shared Art Blocks and fx(hash) contracts split into real projects, Tezos always follows the per-token
creator, and a hand-authored `curation/` layer of JSON overrides wins over everything the chain cannot
know. Only the built `site/` folder ever leaves the machine.

## Bugs worth keeping

Each of these is now a numbered gotcha in the project's working notes. The tag says what caught it.

- **`LIMIT 1` returned the wrong wallet.** SQLite walked the primary-key index, alphabetical order. *(pilot review)*
- **A collection named in math-italic Unicode** produced a 260-character slug and broke Windows MAX_PATH. NFKC-normalize; cap every path segment with a hash. *(day one)*
- **The marketplace account API silently drops delisted works** and never re-adds them. Diffing the chain explorer's list against the database found 41 works sitting in the vault the whole time. *(the collector noticed a missing work)*
- **The near-disaster.** An orphan sweep keyed on one table deleted 890 files owned by four others. Audit failed the build; backup restored 2,337 files. *(audit + backup)*
- **The catch-all artist.** A shared Art Blocks deployer "owned" every project it hosted, so one fake artist swallowed 34 works. A diagnostic query found all three deployers with the same flaw. *(the collector knew the artist)*
- **Click one work, get another.** `.masonry .card` (0,2,0) outranked `.hidden-card` (0,1,0); "show more" cards rendered visible while still classed hidden, and the lightbox opened index 0. *(the collector diagnosed it from scroll position)*
- **Consolidated works lost their past.** Auction wins moved hot → vault kept only the internal hop. Backfilling from the chain brought 35 auctions and their bids back. *(the collector remembered the bidding)*
- **A throttled RPC call looked exactly like a revert.** Both returned `None`. A stage could record "no totalSupply" as a fact while rate-limited. Now a typed exception, exponential backoff, and "inconclusive" in the ledger. *(fresh eyes, week eight)*
- **160 duplicate events that weren't.** Partial ERC-1155 transfers and bundle buys. True duplicates: 3. Identity must include quantity and price. *(fresh eyes, then self-corrected)*
- **Prose drift.** Sixteen essays said "seventeen hundred works, four chains" while the census said 3,735 and six. Numbers are now build-time tokens, and a gate parses number words back out of the live prose. *(fresh eyes)*

## How the AI was used

Three Claude models, one human. The AI did the engineering: a working archive and gallery by the end
of day one, then three blockchains' APIs, auctions reconstructed from calldata, three hundred films
transcoded for a TV, a 5,925-page site rendered in about a hundred seconds, a complete Spanish mirror
built and audited in a day. The human did the seeing: which "video" was really a still, which artist
actually made a piece under a shared contract, which acquisition was a private deal, which wallet must
never appear, what stays off the walls (prices), and what the whole thing should sound like. Each of
those decisions is now a curation file or a check, so the machine holds the line the human drew.

The habit that held it together: make the machine prove things. "The map sends you to the wrong piece"
became a script that verifies which work every map point links to. "Are the numbers in the essays
right?" became a parser. Verification is the deliverable, not the chore after it. The same second-pass
pattern as *Return to Fire* and *Heat and Crime*: a fresh-context model at the end found what the model
that wrote the code structurally could not.

What the AI got wrong, honestly: it deleted the 890 files; it blind-stamped a derived category over
hand-curated tiers on every run; it wrote prose that sounded like a machine (the em-dash ban is now an
audit check); it let hand-written numbers go stale for weeks; it fixed a curated fact in the database
three times before fixing it at the seed file.

## The banner and the fifteen

`art/banner.jpg` is the master banner: the gallery's homepage look frozen as one piece, fifteen works
in a five-by-three grid under the along7 brand plate, built from files the artists sent at full
resolution (the print master is 10,368 by 5,185 pixels; the web copy here is 2,400 wide). The fifteen
works sit beside it in `art/` at 1,400 pixels, numbered in grid order, reading left to right, top to
bottom. All fifteen artworks remain the property and copyright of their artists; they appear here
as part of the collection's record, not as anything for sale.

The pipeline code is not in this folder yet. What is public is the method described on the page,
the gallery, and its CC0 dataset of facts.

## Limits, stated plainly

- Foundation and Ninfa settle bidding off-chain or in packed events; their auctions count as won but carry no reconstructed bids.
- Tezos acquisition venues are labels from the indexer, not proven contract addresses; the audit exempts them on purpose.
- "Minter" is a blend (mint-tx sender for most works, first real owner for auction-house mints); a minter mismatch warns, only a timestamp disagreement fails.
- 127 works exist only as marketplace cache copies. Nothing can bring the artist's file back.
- Layer-2 edition sizes may stay null: the free explorer tier blocks log history on Base and Polygon.
- 17 stub bios, 11 unattributed works, one unmerged duplicate profile. Warnings, not failures: decisions, not defects.
- The chain fact-checks carry a date and warn after 30 days. "0 left the vault" is true as of the date on it.

## Layout

```
along7-gallery/
  index.html          the write-up (hand-rolled CSS, theme toggle, no external requests)
  README.md           this file
  art/                banner.jpg + banner-og.jpg (the master banner, web size) and the fifteen works, 01 to 15
```

## Links

- Live gallery: https://along7gallery.xyz
- The vault in numbers (recomputed every build): https://along7gallery.xyz/stats/
- How this was made, the short version on the gallery: https://along7gallery.xyz/archive/
- Open dataset, CC0, facts only, the pictures stay the artists': https://along7gallery.xyz/data/
