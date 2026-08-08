"""Fetch and normalize items from every source in config/sources.json."""

import json
import re
import subprocess
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import feedparser
import requests

ROOT = Path(__file__).resolve().parent.parent
UA = {"User-Agent": "ai-newsletter-bot/0.1 (personal daily digest)"}
# reddit 403s descriptive bot UAs on the public JSON endpoint; a browser UA passes
BROWSER_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
YT_CACHE = ROOT / "config" / "youtube_channels.json"


def load_sources():
    return json.loads((ROOT / "config" / "sources.json").read_text())


def _entry_time(entry):
    for key in ("published_parsed", "updated_parsed"):
        t = entry.get(key)
        if t:
            return datetime.fromtimestamp(time.mktime(t), tz=timezone.utc)
    return None


def _clean(text, limit=280):
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def fetch_feed(source, since):
    """RSS/Atom/YouTube feeds via feedparser."""
    resp = requests.get(source["url"], headers=UA, timeout=30, allow_redirects=True)
    resp.raise_for_status()
    parsed = feedparser.parse(resp.content)
    items = []
    for e in parsed.entries:
        when = _entry_time(e)
        if when is None or when < since:
            continue
        items.append({
            "source": source["name"],
            "category": source.get("category", "misc"),
            "title": _clean(e.get("title"), 200),
            "url": e.get("link", ""),
            "published": when.isoformat(),
            "snippet": _clean(e.get("summary", "")),
        })
    return items


def fetch_hf_daily_papers(source, since):
    """The default listing is not date-sorted; ask for a specific day's batch and
    walk back to the latest weekday batch (weekends/holidays publish nothing)."""
    rows = []
    day = datetime.now(timezone.utc).date()
    for _ in range(4):
        resp = requests.get(f"https://huggingface.co/api/daily_papers?date={day.isoformat()}",
                            headers=UA, timeout=30)
        resp.raise_for_status()
        rows = resp.json()
        if rows:
            break
        day -= timedelta(days=1)
    items = []
    for row in rows:
        paper = row.get("paper", {})
        when = row.get("publishedAt") or paper.get("publishedAt")
        try:
            when_dt = datetime.fromisoformat(when.replace("Z", "+00:00"))
        except (AttributeError, ValueError):
            continue
        items.append({
            "source": source["name"],
            "category": "papers",
            "title": _clean(paper.get("title"), 200),
            "url": f"https://huggingface.co/papers/{paper.get('id', '')}",
            "published": when_dt.isoformat(),
            "snippet": _clean(paper.get("summary", ""), 900),
            "arxiv_id": paper.get("id", ""),
            "upvotes": paper.get("upvotes", 0),
        })
    items.sort(key=lambda i: -i.get("upvotes", 0))
    return items


def fetch_hn(source, since, min_points=50):
    cutoff = int(since.timestamp())
    seen, items = set(), []
    for query in ("AI", "LLM"):
        resp = requests.get(
            "https://hn.algolia.com/api/v1/search",
            params={
                "query": query,
                "tags": "story",
                "numericFilters": f"created_at_i>{cutoff},points>{min_points}",
                "hitsPerPage": 30,
            },
            headers=UA, timeout=30,
        )
        resp.raise_for_status()
        for hit in resp.json().get("hits", []):
            if hit["objectID"] in seen:
                continue
            seen.add(hit["objectID"])
            items.append({
                "source": "Hacker News",
                "category": "community",
                "title": _clean(hit.get("title"), 200),
                "url": hit.get("url") or f"https://news.ycombinator.com/item?id={hit['objectID']}",
                "hn_url": f"https://news.ycombinator.com/item?id={hit['objectID']}",
                "published": datetime.fromtimestamp(hit["created_at_i"], tz=timezone.utc).isoformat(),
                "snippet": f"{hit.get('points', 0)} points, {hit.get('num_comments', 0)} comments",
                "points": hit.get("points", 0),
            })
    items.sort(key=lambda i: -i.get("points", 0))
    return items


def fetch_reddit(source, since):
    """JSON endpoint first (has scores/comments) but reddit blocks it capriciously;
    the .rss endpoint with a browser UA is the reliable fallback."""
    sub = re.search(r"reddit\.com/(r/[^/]+)", source["url"]).group(1)
    try:
        resp = requests.get(f"https://www.reddit.com/{sub}/top.json?t=day&limit=25",
                            headers=BROWSER_UA, timeout=30)
        resp.raise_for_status()
        items = []
        for child in resp.json()["data"]["children"]:
            post = child["data"]
            when = datetime.fromtimestamp(post["created_utc"], tz=timezone.utc)
            if when < since:
                continue
            permalink = f"https://www.reddit.com{post['permalink']}"
            items.append({
                "source": sub,
                "category": "community",
                "title": _clean(post.get("title"), 200),
                "url": post.get("url") or permalink,
                "hn_url": permalink,
                "published": when.isoformat(),
                "snippet": f"{post.get('score', 0)} points, {post.get('num_comments', 0)} comments. "
                           + _clean(post.get("selftext", ""), 150),
                "points": post.get("score", 0),
            })
        items.sort(key=lambda i: -i.get("points", 0))
        return items[:15]
    except (requests.RequestException, KeyError, ValueError):
        pass
    for attempt in range(3):
        resp = requests.get(f"https://www.reddit.com/{sub}/top/.rss?t=day",
                            headers=BROWSER_UA, timeout=30)
        if resp.status_code != 429:
            break
        time.sleep(15 * (attempt + 1))
    resp.raise_for_status()
    items = []
    for e in feedparser.parse(resp.content).entries:
        when = _entry_time(e)
        if when is None or when < since:
            continue
        items.append({
            "source": sub,
            "category": "community",
            "title": _clean(e.get("title"), 200),
            "url": e.get("link", ""),
            "hn_url": e.get("link", ""),
            "published": when.isoformat(),
            "snippet": _clean(e.get("summary", ""), 150),
        })
    return items[:15]


def resolve_youtube_channel(handle):
    cache = json.loads(YT_CACHE.read_text()) if YT_CACHE.exists() else {}
    if handle in cache:
        return cache[handle]
    channel_id = None
    # yt-dlp handles YouTube's consent/bot-check pages; plain requests gets an interstitial
    try:
        proc = subprocess.run(
            [str(ROOT / ".venv" / "bin" / "yt-dlp"), "--skip-download",
             "--playlist-items", "1", "--print", "channel_id",
             f"https://www.youtube.com/{handle}/videos"],
            capture_output=True, text=True, timeout=90,
        )
        match = re.search(r"UC[\w-]{20,}", proc.stdout)
        channel_id = match.group(0) if match else None
    except (subprocess.SubprocessError, OSError):
        pass
    if not channel_id:
        resp = requests.get(f"https://www.youtube.com/{handle}", headers=UA, timeout=30)
        match = re.search(r'"channelId":"(UC[\w-]+)"', resp.text)
        channel_id = match.group(1) if match else None
    if not channel_id:
        return None
    cache[handle] = channel_id
    YT_CACHE.write_text(json.dumps(cache, indent=2))
    return channel_id


def fetch_youtube(source, since):
    channel_id = resolve_youtube_channel(source["handle"])
    if not channel_id:
        return []
    feed_source = dict(source, url=f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}")
    items = fetch_feed(feed_source, since)[:3]  # cap per channel so one prolific uploader can't flood the edition
    for item in items:
        item["category"] = "video"
        item["tier"] = source.get("tier", "core")
    return items


def fetch_transcript(video_url, timeout=90):
    """Auto-generated subtitles via yt-dlp -> plain text. Returns None on any failure."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            subprocess.run(
                [str(ROOT / ".venv" / "bin" / "yt-dlp"), "--skip-download",
                 "--write-auto-subs", "--sub-langs", "en", "--sub-format", "vtt",
                 "-o", f"{tmp}/sub", video_url],
                capture_output=True, timeout=timeout, check=True,
            )
            vtts = list(Path(tmp).glob("*.vtt"))
            if not vtts:
                return None
            lines, seen = [], set()
            for line in vtts[0].read_text().splitlines():
                if "-->" in line or line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
                    continue
                text = re.sub(r"<[^>]+>", "", line).strip()
                if text and text not in seen:
                    seen.add(text)
                    lines.append(text)
            return " ".join(lines)[:8000] or None
        except (subprocess.SubprocessError, OSError):
            return None


def fetch_article_text(url, limit=6000):
    """Crude readability: concatenated <p> contents."""
    try:
        resp = requests.get(url, headers=UA, timeout=30)
        resp.raise_for_status()
        paras = re.findall(r"<p[^>]*>(.*?)</p>", resp.text, re.S)
        text = " ".join(_clean(p, 10000) for p in paras)
        return re.sub(r"\s+", " ", text).strip()[:limit] or None
    except requests.RequestException:
        return None


def fetch_all(hours=28, include_youtube=True):
    """Returns (items, stats). arXiv's firehose is counted for the ticker but not
    fed to the editor — HF Daily Papers is the curated paper source."""
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    config = load_sources()
    items, stats = [], {"arxiv_new": 0, "arxiv_items": [], "errors": []}

    for category in ("papers", "communities", "labs", "press", "newsletters"):
        for source in config.get(category, []):
            if source.get("status") == "candidate" and source.get("type") == "scrape":
                continue  # no fetcher for scrape-only sources yet
            source = dict(source, category="community" if category == "communities" else category)
            try:
                if "arxiv.org" in (source.get("url") or ""):
                    # the arXiv firehose goes to the wire page, not the editor
                    arxiv = fetch_feed(source, since - timedelta(hours=24))
                    stats["arxiv_new"] = len(arxiv)
                    stats["arxiv_items"] = arxiv[:80]
                elif "daily_papers" in (source.get("url") or ""):
                    items.extend(fetch_hf_daily_papers(source, since)[:30])
                elif "hn.algolia.com" in (source.get("url") or ""):
                    items.extend(fetch_hn(source, since)[:30])
                elif "reddit.com" in (source.get("url") or ""):
                    items.extend(fetch_reddit(source, since))
                elif source.get("type") in ("rss", "atom", "json"):
                    items.extend(fetch_feed(source, since))
            except Exception as exc:  # a dead feed must never kill the edition
                stats["errors"].append(f"{source['name']}: {exc}")

    if include_youtube:
        for source in config.get("youtube", []):
            try:
                items.extend(fetch_youtube(dict(source, category="video"), since))
            except Exception as exc:
                stats["errors"].append(f"{source['name']}: {exc}")

    for idx, item in enumerate(items):
        item["id"] = idx
    return items, stats
