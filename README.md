# The Daily Prompt 📰

A daily AI-news newspaper, aggregated by agents and typeset as a 1-bit pixel-art broadsheet.

## How it works

```
config/sources.json ──► pipeline/fetch.py ──► ~90 items/day
                            │  (RSS, HF daily_papers API, HN Algolia API,
                            │   YouTube channel RSS + yt-dlp transcripts)
                            ▼
                        pipeline/compose.py
                            │  editor  (GLM-4.6): picks lead, briefs, papers, quotes
                            │  writer  (Gemma 4 31B): writes broadsheet copy
                            ▼
                        pipeline/render.py ──► templates/*.j2
                            ▼
                        output/YYYY-MM-DD/   (multi-page edition)
                          index.html         front page
                          story-1..4.html    full inside articles (lead + 3)
                          wire.html          everything scanned + arXiv listings
                        output/latest ──► symlink to today's edition
```

Cost per edition: under a cent (models via OpenRouter, key in `.env` — never commit it).

## Run manually

```sh
.venv/bin/python -m pipeline.main            # today's edition
.venv/bin/python -m pipeline.main --no-youtube --hours 48
```

## The news-stand app

`app/index.html` — a small React app (vendored UMD React in `app/vendor/`, no build
step, works offline from `file://`). Timeline of all editions across the top, a
1-bit patterned source-split chart + edition stats, and the selected edition in an
iframe. Data comes from `output/editions.js`, a manifest the renderer regenerates
from each edition's `stats.json` (a `.js` global because `file://` pages can't
`fetch()` local JSON). **This is the bookmark.**

## Daily delivery

A launchd agent builds the paper every morning at 6:47 and posts a macOS notification:

- Plist: `~/Library/LaunchAgents/com.albert.ai-newsletter.plist`
- Logs: `output/build.log`
- Read it: open `app/index.html` (bookmark it); raw editions live in `output/`
- Uninstall: `launchctl bootout gui/$(id -u)/com.albert.ai-newsletter`

## Design

`design/prototype.html` is the approved design reference. Fonts: Jacquard 24
(masthead), Press Start 2P (headlines), VT323 (body), inlined as base64 in
`assets/fonts/`. The lead illustration is drawn at runtime: procedural scene,
4×4 Bayer ordered dithering, 1 bit per pixel, date-seeded so each issue differs.
Light theme is newsprint; dark theme is a green-phosphor CRT.

## Source registry

`config/sources.json` — feeds verified live on 2026-08-07 (deep-research pass).
Notable facts encoded there: Papers With Code is dead (use HF daily_papers);
X/Twitter is never polled directly (smol.ai's AI News summarizes 544 accounts
for us); arXiv feeds are empty on weekends.
