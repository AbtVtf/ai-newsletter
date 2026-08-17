"""Render the edition into a multi-page pixel-newspaper directory."""

import json
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT = Path(__file__).resolve().parent.parent
EPOCH = datetime(2026, 8, 7, tzinfo=timezone.utc)  # issue no. 1


def _env():
    return Environment(
        loader=FileSystemLoader(ROOT / "templates"),
        autoescape=select_autoescape(["html", "j2"]),
    )


def _fonts():
    return {
        f"font_{name}": (ROOT / "assets" / "fonts" / f"font_{name}.b64").read_text().strip()
        for name in ("jacquard", "press", "vt323")
    }


def issue_number(now):
    return max(1, (now.date() - EPOCH.date()).days + 1)


def write_edition_dir(context, now):
    """Renders the single-page edition. Returns the edition dir."""
    env, fonts = _env(), _fonts()
    out_dir = ROOT / "output" / now.strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "index.html").write_text(
        env.get_template("edition.html.j2").render(**context, **fonts))

    # retire multi-page leftovers from earlier builds of the same day
    for stale in list(out_dir.glob("story-*.html")) + [out_dir / "wire.html"]:
        if stale.exists():
            stale.unlink()

    articles = context["articles"]
    (out_dir / "articles.json").write_text(json.dumps({
        "date": now.strftime("%Y-%m-%d"),
        "date_display": context["date_display"],
        "issue_no": context["issue_no"],
        "articles": articles,
        "briefs": context["briefs"],
        "papers": context["papers"],
    }, indent=1))
    sections = 1 + (len(articles) - 1) + 1 + len(context["screens"]) + 4
    (out_dir / "stats.json").write_text(json.dumps({
        "date": now.strftime("%Y-%m-%d"),
        "date_display": context["date_display"],
        "issue_no": context["issue_no"],
        "headline": articles[0]["headline"],
        "items_scanned": context["items_scanned"],
        "arxiv_new": context["arxiv_new"],
        "pages": sections,
        "cost": context["cost"],
        "split": context["source_split"],
    }, indent=1))
    write_manifest()

    latest = ROOT / "output" / "latest"
    if latest.is_symlink() or latest.is_file():
        latest.unlink()
    if not latest.exists():
        latest.symlink_to(out_dir.name)
    # retire the old single-file alias if present
    old = ROOT / "output" / "latest.html"
    if old.exists():
        old.unlink()
    return out_dir


def write_manifest():
    """Regenerate output/editions.js from every edition's stats.json.
    A .js global (not .json) because file:// pages can't fetch() local JSON."""
    editions = []
    for stats_file in sorted((ROOT / "output").glob("*/stats.json")):
        if stats_file.parent.is_symlink():  # skip the 'latest' alias
            continue
        try:
            editions.append(json.loads(stats_file.read_text()))
        except json.JSONDecodeError:
            continue
    (ROOT / "output" / "editions.js").write_text(
        "window.EDITIONS = " + json.dumps(editions) + ";\n")
