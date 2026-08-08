"""Editorial brain: the editor model picks the day's stories, the writer model
writes the newspaper copy. Papers keep their real titles/ids — no rewriting."""

import json

from . import fetch, llm

VOICE = """You write for "The Daily Prompt", a daily AI newspaper styled as a
1920s pixel-art broadsheet. Voice: dry wit, precise, a little theatrical in
headlines, never breathless, never fabricates. Every claim must come from the
provided source material. Headlines in the grand old-newspaper register."""


def editor_pick(items, covered_titles=None, max_candidates=150):
    lines = []
    for item in items[:max_candidates]:
        cat = item["category"]
        if item.get("tier") == "adjacent":
            cat += "-adjacent"  # AI-adjacent channel: only pick if squarely about AI
        lines.append(f"[{item['id']}] ({item['source']}/{cat}) {item['title']} — {item['snippet'][:120]}")
    covered_block = ""
    if covered_titles:
        covered_block = ("\nALREADY COVERED in recent editions — do NOT pick candidates that are "
                         "the same story/event as any of these, even from a different outlet, "
                         "unless the candidate reports a significant NEW development:\n"
                         + "\n".join(f"- {t}" for t in covered_titles) + "\n")
    prompt = f"""You are the editor of a daily AI newspaper. Below are today's candidate items.
{covered_block}

Pick, using only these ids (no id twice):
- "lead": the single most significant AI story of the day (prefer news over papers)
- "secondary": exactly 5 more stories strong enough to carry their own full article
- "briefs": exactly 10 items for a "From the labs" wire-brief section (news, not papers)
- "papers": up to 10 research papers (category papers only)
- "videos": up to 6 videos worth a written article — AT MOST ONE PER CHANNEL (skip video-adjacent items unless squarely about AI)
- "overheard": up to 8 community items (Hacker News / Reddit) with strong discussion energy
- "reasoning": one sentence on why the lead leads

IMPORTANT: when several candidates cover the SAME underlying story (e.g. two
outlets reporting one announcement), pick only the best ONE across all
categories — never two articles about the same event.

Return ONLY JSON: {{"lead": id, "secondary": [ids], "briefs": [ids], "papers": [ids], "videos": [ids], "overheard": [ids], "reasoning": "..."}}

Candidates:
{chr(10).join(lines)}"""
    picks = llm.chat_json("editor", prompt, max_tokens=2500)
    by_id = {i["id"]: i for i in items}
    used = set()

    def take(ids, cap):
        out = []
        for i in ids or []:
            if i in by_id and i not in used:
                used.add(i)
                out.append(by_id[i])
            if len(out) == cap:
                break
        return out

    lead = take([picks.get("lead")], 1)
    videos = take(picks.get("videos"), 12)
    seen_channels, diverse_videos = set(), []
    for v in videos:  # hard guarantee: one written video article per channel
        if v["source"] not in seen_channels:
            seen_channels.add(v["source"])
            diverse_videos.append(v)
    return {
        "lead": lead[0] if lead else None,
        "secondary": take(picks.get("secondary"), 5),
        "briefs": take(picks.get("briefs"), 10),
        "papers": take(picks.get("papers"), 10),
        "videos": diverse_videos[:6],
        "overheard": take(picks.get("overheard"), 8),
        "reasoning": picks.get("reasoning", ""),
    }


def dedup_against_history(picks, covered_titles):
    """Backstop for the editor ignoring the covered list: a second model flags
    picked items that re-cover an already-covered event. Returns ids to drop."""
    candidates = [picks["lead"]] + picks["secondary"] + picks["briefs"] + picks["videos"] + picks["overheard"]
    candidates = [c for c in candidates if c]
    listing = "\n".join(f"[{c['id']}] {c['title']}" for c in candidates)
    covered = "\n".join(f"- {t}" for t in covered_titles)
    prompt = f"""Previously published newspaper headlines:
{covered}

Candidate stories picked for today's edition:
{listing}

Which candidates report the SAME event as a previously published headline?
Different outlet or different wording still counts as the same event. A candidate
only survives if it reports a genuinely NEW development beyond what was published.

Return ONLY JSON: {{"drop": [ids of candidates to drop]}} (empty list if none)."""
    try:
        out = llm.chat_json("summarizer", prompt, max_tokens=600)
        return {i for i in out.get("drop", []) if isinstance(i, int)}
    except RuntimeError:
        return set()


def write_paper_articles(papers):
    """A short written article (2-3 paragraphs) per paper, one batched call."""
    if not papers:
        return []
    listing = json.dumps([{"title": p["title"], "abstract": p["snippet"][:800]} for p in papers], indent=1)
    prompt = f"""{VOICE}

Write a short article (2-3 paragraphs) about each research paper below:
what it does, how, and why it matters — so the reader is informed without
opening the paper. Ground strictly in each abstract; no hype, no invention.

{listing}

Return ONLY JSON: {{"articles": [{{"paragraphs": ["...", "..."]}}, ...]}} — exactly {len(papers)} entries, same order."""
    try:
        out = llm.chat_json("summarizer", prompt, max_tokens=8000)
        articles = out.get("articles", [])
    except RuntimeError:
        articles = []
    result = []
    for i, p in enumerate(papers):
        paragraphs = articles[i].get("paragraphs") if i < len(articles) and isinstance(articles[i], dict) else None
        result.append(paragraphs or [p["snippet"][:400]])
    return result


def video_desk(videos):
    """A written article per editor-picked video, from its transcript."""
    desk = []
    for video in videos:
        transcript = fetch.fetch_transcript(video["url"])
        material = (transcript or video["snippet"])[:4500]
        prompt = f"""{VOICE}

A YouTube video is being covered as a written column. Write the piece so a reader
never needs to watch the video.

VIDEO: "{video['title']}" by {video['source']}
{'TRANSCRIPT (auto-captions)' if transcript else 'DESCRIPTION ONLY (no transcript available — keep it short)'}: {material}

Return ONLY JSON:
{{"headline": "broadsheet register, under 10 words",
 "body": ["2 to 4", "paragraphs grounded in the material"]}}"""
        try:
            written = llm.chat_json("summarizer", prompt, max_tokens=1800)
        except RuntimeError:
            written = {}
        desk.append({
            "headline": written.get("headline", video["title"]),
            "body": written.get("body") or [video["snippet"]],
            "source": video["source"],
            "url": video["url"],
            "had_transcript": bool(transcript),
        })
    return desk


def enrich(item):
    """Pull full text (article or YouTube transcript) for a feature story."""
    if not item:
        return None
    if item["category"] == "video":
        return fetch.fetch_transcript(item["url"])
    return fetch.fetch_article_text(item["url"])


def write_article(item, fulltext):
    """Expanded inside-page article for one feature story."""
    prompt = f"""{VOICE}

Write an expanded inside-page article about this story.

STORY: {item['title']} (source: {item['source']}, {item['url']})
MATERIAL: {(fulltext or item['snippet'])[:6000]}

Rules: ground every sentence in the material above. If the material is thin,
write a shorter piece rather than padding or inventing. 4-6 body paragraphs.

Return ONLY JSON:
{{"headline": "broadsheet register, under 12 words",
 "deck": "one-sentence standfirst",
 "paragraphs": ["4 to 6", "body paragraphs"]}}"""
    return llm.chat_json("writer", prompt, max_tokens=2500)


def write_front(picks, lead_article, date_display):
    """Front-page furniture: briefs copy, overheard quips, weather, classifieds.
    Lead headline/deck come from its inside article for consistency."""
    briefs_input = json.dumps(
        [{"title": b["title"], "source": b["source"], "snippet": b["snippet"][:200]} for b in picks["briefs"]],
        indent=1)
    overheard_input = json.dumps(
        [{"title": o["title"], "source": o["source"], "snippet": o["snippet"][:150]} for o in picks["overheard"]],
        indent=1)

    prompt = f"""{VOICE}

Today is {date_display}. The lead story is: "{lead_article['headline']}" — {lead_article['deck']}

Write the remaining front-page furniture. Source material:

WIRE BRIEFS ({len(picks['briefs'])}): {briefs_input}

COMMUNITY ITEMS ({len(picks['overheard'])}): {overheard_input}

Return ONLY JSON with exactly these keys:
{{
 "kicker": "short all-caps-style section eyebrow for the lead",
 "briefs": [{{"headline": "under 10 words", "body": "3-5 sentences that fully inform — the reader should not need to click through; grounded in that brief's snippet"}}, ...exactly {len(picks['briefs'])}, same order as input],
 "overheard": [{{"quote": "a punchy one-line paraphrase of the community item", "attribution": "— source name", "context": "1-2 sentences on what the discussion is about and why it caught fire"}}, ...same order as input],
 "weather": {{"big": "2-4 word forecast, GPU/compute themed, clearly humorous", "detail": "one sentence continuing the joke"}},
 "classifieds": [{{"tag": "For sale|Wanted|Lost|Notice", "body": "one-sentence AI-culture joke"}}, ...exactly 3]
}}"""
    return llm.chat_json("writer", prompt, max_tokens=7000)
