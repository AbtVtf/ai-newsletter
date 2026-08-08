"""Build today's edition of The Daily Prompt.

Usage: .venv/bin/python -m pipeline.main [--hours 28] [--no-youtube]
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import compose, fetch, llm, render

SLUG_LABELS = ["lead", "second", "third", "fourth", "fifth", "sixth"]
COVERED_PATH = Path(__file__).resolve().parent.parent / "output" / "covered.json"


def load_coverage(today_str):
    """Coverage ledger from PREVIOUS days (same-day entries are ignored so a
    rebuild of today's edition doesn't block its own stories)."""
    history = json.loads(COVERED_PATH.read_text()) if COVERED_PATH.exists() else []
    past = [h for h in history if h["date"] != today_str]
    return history, {h["url"] for h in past}, [h["title"] for h in past[-60:]]


def save_coverage(history, covered_items, today_str, now):
    keep_after = (now - timedelta(days=14)).strftime("%Y-%m-%d")
    history = [h for h in history if h["date"] >= keep_after and h["date"] != today_str]
    seen = {h["url"] for h in history}
    for item in covered_items:
        if item["url"] not in seen:
            seen.add(item["url"])
            history.append({"date": today_str, "url": item["url"], "title": item["title"]})
    COVERED_PATH.write_text(json.dumps(history, indent=1))


def _when(iso):
    try:
        return datetime.fromisoformat(iso).strftime("%H:%M")
    except (ValueError, TypeError):
        return "--:--"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=28)
    parser.add_argument("--no-youtube", action="store_true")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    date_display = now.strftime("%A, %B %-d, %Y")

    print("fetching sources...")
    items, stats = fetch.fetch_all(hours=args.hours, include_youtube=not args.no_youtube)
    print(f"  {len(items)} items in window; arXiv new: {stats['arxiv_new']}")
    for err in stats["errors"]:
        print(f"  feed error: {err}", file=sys.stderr)

    today_str = now.strftime("%Y-%m-%d")
    history, covered_urls, covered_titles = load_coverage(today_str)
    before = len(items)
    items = [i for i in items if i["url"] not in covered_urls]
    print(f"  {before - len(items)} already-covered items filtered out")
    if len(items) < 5:
        sys.exit("not enough items to build an edition — check feeds/network")

    print("editor picking stories...")
    picks = compose.editor_pick(items, covered_titles)
    if not picks["lead"]:
        sys.exit("editor returned no lead story")

    if covered_titles:
        drop = compose.dedup_against_history(picks, covered_titles)
        if drop:
            print(f"  dedup judge dropped {len(drop)} re-covered stories")
            for key in ("secondary", "briefs", "videos", "overheard"):
                picks[key] = [i for i in picks[key] if i["id"] not in drop]
            if picks["lead"]["id"] in drop:
                if not picks["secondary"]:
                    sys.exit("dedup left no viable lead story")
                picks["lead"] = picks["secondary"].pop(0)
                print(f"  lead was a repeat — promoted: {picks['lead']['title'][:70]}")

    print(f"  lead: {picks['lead']['title'][:80]}")
    print(f"  reasoning: {picks['reasoning']}")

    # feature stories: lead + secondaries each get a full inside page
    features = [picks["lead"]] + picks["secondary"]
    articles = []
    for idx, item in enumerate(features):
        print(f"writing article {idx + 1}/{len(features)}: {item['title'][:60]}")
        fulltext = compose.enrich(item)
        written = compose.write_article(item, fulltext)
        articles.append({
            "slug": f"story-{idx + 1}",
            "nav_label": SLUG_LABELS[idx] if idx < len(SLUG_LABELS) else f"story {idx + 1}",
            "headline": written.get("headline", item["title"]),
            "deck": written.get("deck", ""),
            "paragraphs": written.get("paragraphs", [item["snippet"]]),
            "kicker": "Front page" if idx == 0 else "Inside",
            "url": item["url"],
            "hn_url": item.get("hn_url"),
            "source": item["source"],
            "original_title": item["title"],
            "published_display": _when(item.get("published")),
        })

    # the forums section must actually represent the forums: guarantee reddit a seat
    reddit_items = sorted((i for i in items if i["source"].startswith("r/")),
                          key=lambda i: -i.get("points", 0))
    if reddit_items and not any(o["source"].startswith("r/") for o in picks["overheard"]):
        picked_ids = {o["id"] for o in picks["overheard"]}
        extras = [r for r in reddit_items if r["id"] not in picked_ids][:2]
        picks["overheard"] = picks["overheard"][:8 - len(extras)] + extras

    print("writer composing front page...")
    copy = compose.write_front(picks, articles[0], date_display)

    briefs = []
    for item, written in zip(picks["briefs"], copy.get("briefs", [])):
        briefs.append({"headline": written.get("headline", item["title"]),
                       "body": written.get("body", item["snippet"]),
                       "url": item["url"], "source": item["source"]})
    overheard = []
    for item, written in zip(picks["overheard"], copy.get("overheard", [])):
        overheard.append({"quote": written.get("quote", item["title"]),
                          "attribution": written.get("attribution", f"— {item['source']}"),
                          "context": written.get("context", item["snippet"]),
                          "url": item.get("hn_url", item["url"])})

    print(f"writing {len(picks['papers'])} paper articles...")
    paper_paragraphs = compose.write_paper_articles(picks["papers"])
    papers = [{"title": p["title"],
               "meta": " · ".join(x for x in (p.get("arxiv_id"),
                                              f"{p.get('upvotes', 0)} upvotes on HF",
                                              p["source"]) if x),
               "paragraphs": paper_paragraphs[i],
               "url": p["url"]} for i, p in enumerate(picks["papers"])]

    print(f"video desk: writing {len(picks['videos'])} video articles...")
    screens = compose.video_desk(picks["videos"])
    for s in screens:
        print(f"  {'transcript' if s['had_transcript'] else 'desc only'}: {s['headline'][:60]}")

    # "on the air" rack: one per channel, core channels first
    rack, seen_channels = [], set()
    for v in sorted((i for i in items if i["category"] == "video"),
                    key=lambda i: i.get("tier") != "core"):
        if v["source"] not in seen_channels:
            seen_channels.add(v["source"])
            rack.append(v)
    videos = rack[:8]

    # the wire: everything scanned, grouped
    group_defs = [("Community", ("community",)), ("Labs & press", ("labs", "press")),
                  ("Papers & research", ("papers",)), ("Newsletters", ("newsletters",)),
                  ("Video", ("video",))]
    groups = []
    for title, cats in group_defs:
        rows = [{"when": _when(i.get("published")), "title": i["title"],
                 "url": i["url"], "snippet": (i.get("snippet") or "")[:110]}
                for i in items if i["category"] in cats]
        if rows:
            groups.append({"title": title, "items": rows})
    arxiv_items = [{"when": _when(p.get("published")), "title": p["title"], "url": p["url"]}
                   for p in stats["arxiv_items"]]

    top_hn = max((i for i in items if i["source"] == "Hacker News"),
                 key=lambda i: i.get("points", 0), default=None)
    ticker = [
        {"label": "ARXIV NEW", "value": str(stats["arxiv_new"])},
        {"label": "ITEMS SCANNED", "value": str(len(items))},
        {"label": "HN TOP", "value": f"{top_hn['points']} PTS" if top_hn else "N/A"},
        {"label": "PAPERS", "value": str(len(papers))},
        {"label": "VIDEOS COVERED", "value": str(len(picks["videos"]))},
    ]

    split_labels = {"community": "Community", "labs": "Labs", "press": "Press",
                    "papers": "Papers", "newsletters": "Newsletters", "video": "Video"}
    source_split = {}
    for item in items:
        label = split_labels.get(item["category"], "Other")
        source_split[label] = source_split.get(label, 0) + 1

    context = {
        "source_split": source_split,
        "date_display": date_display,
        "issue_no": render.issue_number(now),
        "ticker": ticker,
        "articles": articles,
        "copy": copy,
        "briefs": briefs,
        "overheard": overheard,
        "papers": papers,
        "screens": screens,
        "videos": videos,
        "groups": groups,
        "arxiv_items": arxiv_items,
        "arxiv_new": stats["arxiv_new"],
        "items_scanned": len(items),
        "build_time": now.strftime("%H:%M UTC"),
        "cost": f"{llm.spent():.4f}",
        "seed": int(now.strftime("%Y%m%d")),
    }

    print("rendering pages...")
    out_dir = render.write_edition_dir(context, now)

    covered = features + picks["briefs"] + picks["videos"] + picks["papers"] + picks["overheard"]
    save_coverage(history, covered, today_str, now)
    print(f"done: {out_dir}/index.html  (cost ${llm.spent():.4f}; coverage ledger: {len(covered)} stories logged)")


if __name__ == "__main__":
    main()
