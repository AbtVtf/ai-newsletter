"""Edition article -> short spoken script (~35 s) + platform caption.

The broadsheet copy is 250-340 words of 1920s newspaper voice. That reads
badly out loud and runs two minutes. This rewrites it as plain spoken
narration under a word budget, and writes the social caption/hashtags.

Run with the newsletter venv:
  .venv/bin/python -m video.script_writer --date 2026-08-08 --story story-1
"""

import argparse
import json
import re
from pathlib import Path

from pipeline import llm

ROOT = Path(__file__).resolve().parent.parent

PROMPT = """You write scripts for a 30-40 second vertical news short. An AI \
presenter reads your words aloud to camera. The channel is "The Daily Prompt", \
a daily AI-news show.

Source story (written in an ornate 1920s newspaper voice — do NOT copy that voice):
HEADLINE: {headline}
DECK: {deck}
BODY:
{body}

Write the script. Hard rules:
- {min_words}-{max_words} words total. This is a hard limit.
- Plain modern spoken English. Short sentences. One idea per sentence.
- THE FIRST SENTENCE DECIDES EVERYTHING. Most viewers leave after two \
seconds, which is about five words. So the first sentence must be UNDER 8 \
WORDS and must contain the most alarming or surprising concrete fact in the \
story. A sentence fragment is fine and often better. Do not open with who \
announced it, when it happened, or any background. Name the shocking thing \
first and explain afterwards.
  Bad:  "OpenAI has paused development on its upcoming model, Astra."
  Good: "This AI could break into protected systems."
- No sentence anywhere may exceed 14 words.
- No "welcome back", no "in today's news", no greeting of any kind.
- Numbers and names stay exact. Never invent a fact that is not in the body.
- A CAPABILITY IS NOT AN EVENT. This is the most common way AI news gets \
distorted, so check it every time. If the body says a model "could", "is \
able to", or "demonstrated the ability to" do something in testing, never \
write that it DID it, escaped, got loose, or attacked anyone. If the body \
attributes an event to a different model, lab or product, never transfer it \
to the subject of this story — especially when the body explicitly separates \
them.
  Bad:  "The AI escaped and attacked real systems."
  Good: "In testing, it broke into protected systems on its own."
- Punchy is good. False is not. Do not overstate to make a stronger hook: a \
viewer who stops for a promise you do not keep leaves faster than one who \
never stopped, and the channel only has value if it is right.
- No metaphors, no rhetorical questions, no "let's dive in", no hype adjectives.
- End on what it means or what happens next. Do not say "like and subscribe".
- Text is spoken aloud: no markdown, no emoji, no parentheses, no bullet \
points, no numerals-as-symbols (write "twenty percent", not "20%").
- Every sentence ends with a period, question mark, or exclamation mark.

Also write:
- title: under 70 characters, plain and specific, no clickbait, no emoji.
- overlay: 4-8 words, ALL CAPS, the on-screen headline.
- hook: 3-7 words, ALL CAPS, for the thumbnail. The single most arresting
  fact, as a statement. Not a question, not a tease, not "you won't believe".
  It must be true on its own without the video explaining it.
- kicker: 1-3 words, ALL CAPS, a section label like "MODEL SAFETY" or
  "OPEN WEIGHTS". Sits above the hook on the thumbnail.
- description: 3-4 sentences for the video description. First sentence states
  what happened. The rest give the context a viewer needs. Plain prose, no
  hashtags, no emoji, no calls to action, no links.
- hashtags: exactly 5, space separated, each starting with #, no punctuation.

Respond with ONLY this JSON:
{{"script": "...", "title": "...", "overlay": "...", "hook": "...",
  "kicker": "...", "description": "...", "hashtags": "..."}}"""

# Same fields, but the script already exists and must not change — the audio
# and the viseme timeline were built from it.
METADATA_PROMPT = """Below is a news article and the finished narration script \
for a 30-40 second vertical video made from it. Write the publishing metadata.

ARTICLE HEADLINE: {headline}
ARTICLE BODY:
{body}

NARRATION SCRIPT (already recorded — do not rewrite it):
{script}

Rules: every fact must come from the article. No clickbait, no emoji, no \
calls to action, no invented numbers. A CAPABILITY IS NOT AN EVENT: if the \
article says a model "could" or "demonstrated the ability to" do something \
in testing, never write that it did it, escaped, or attacked anyone, and \
never transfer an event from a different model or lab onto this story.

- title: under 70 characters, plain and specific.
- overlay: 4-8 words, ALL CAPS, the on-screen headline.
- hook: 3-7 words, ALL CAPS, for the thumbnail. The single most arresting \
fact, as a statement. Not a question, not a tease. True on its own.
- kicker: 1-3 words, ALL CAPS, a section label like "MODEL SAFETY".
- description: 3-4 sentences. First sentence states what happened, the rest \
give context. Plain prose, no hashtags, no links.
- hashtags: exactly 5, space separated, each starting with #.

Respond with ONLY this JSON:
{{"title": "...", "overlay": "...", "hook": "...", "kicker": "...",
  "description": "...", "hashtags": "..."}}"""


def word_count(text):
    return len(re.findall(r"[\w']+", text))


def first_sentence_words(script):
    m = re.split(r"(?<=[.!?])\s+", script.strip())
    return word_count(m[0]) if m else 0


def write_script(article, min_words=48, max_words=62, retries=2):
    body = "\n".join(article["paragraphs"])
    prompt = PROMPT.format(headline=article["headline"], deck=article.get("deck", ""),
                           body=body, min_words=min_words, max_words=max_words)
    for attempt in range(retries + 1):
        out = llm.chat_json("writer", prompt, max_tokens=1200)
        script = re.sub(r"\s+", " ", out.get("script", "")).strip()
        n = word_count(script)
        opener = first_sentence_words(script)
        problems = []
        if not (min_words - 10 <= n <= max_words + 12):
            problems.append(f"total was {n} words, need {min_words}-{max_words}")
        if opener > 9:
            problems.append(f"first sentence was {opener} words, need under 8")
        if not problems:
            out["script"] = script
            out["word_count"] = n
            out["opener_words"] = opener
            return out
        print(f"  retrying: {'; '.join(problems)}")
        prompt += "\n\nYour last attempt failed: " + "; ".join(problems) + "."
    out["script"] = script
    out["word_count"] = n
    out["opener_words"] = opener
    return out


def write_metadata(article, script):
    """Publishing fields for a script that already exists. Never touches the
    script itself — the audio and viseme timeline were built from it."""
    prompt = METADATA_PROMPT.format(headline=article["headline"],
                                    body="\n".join(article["paragraphs"]),
                                    script=script)
    return llm.chat_json("writer", prompt, max_tokens=900)


def build_description(meta, article):
    """Assemble the publishable description. The source URL is pasted in from
    the edition data, never asked of the model, so it cannot be invented."""
    parts = [meta["description"].strip(), ""]
    parts.append(f"Source: {article['source']} — {article['url']}")
    parts.append("")
    parts.append("The Daily Prompt is a daily AI-news paper written by agents "
                 "and read by one. New edition every morning.")
    parts.append("")
    parts.append(meta.get("hashtags", "").strip())
    return "\n".join(parts).strip() + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--story", help="slug like story-1; default = all articles")
    ap.add_argument("--min-words", type=int, default=48)
    ap.add_argument("--max-words", type=int, default=62)
    ap.add_argument("--metadata-only", action="store_true",
                    help="refresh title/hook/description for existing scripts")
    args = ap.parse_args()

    edition = ROOT / "output" / args.date
    data = json.loads((edition / "articles.json").read_text())
    articles = [a for a in data["articles"]
                if not args.story or a["slug"] == args.story]
    if not articles:
        raise SystemExit(f"no article matching {args.story} in {edition}")

    out_dir = edition / "shorts"
    out_dir.mkdir(parents=True, exist_ok=True)
    for article in articles:
        work = out_dir / article["slug"]
        work.mkdir(parents=True, exist_ok=True)
        script_path = work / "script.txt"

        if args.metadata_only:
            if not script_path.exists():
                print(f"skipping {article['slug']}: no script yet")
                continue
            print(f"metadata: {article['slug']} — {article['headline'][:55]}")
            script = script_path.read_text().strip()
            out = json.loads((work / "script.json").read_text())
            out.update(write_metadata(article, script))
        else:
            print(f"writing script: {article['slug']} — {article['headline'][:55]}")
            out = write_script(article, args.min_words, args.max_words)

        out.update({"slug": article["slug"], "date": data["date"],
                    "date_display": data["date_display"],
                    "issue_no": data["issue_no"], "source_url": article["url"],
                    "source": article["source"],
                    "source_headline": article["headline"]})
        if not args.metadata_only:
            script_path.write_text(out["script"] + "\n")
            print(f"  {out['word_count']} words -> {script_path}")

        (work / "script.json").write_text(json.dumps(out, indent=1))
        (work / "description.txt").write_text(build_description(out, article))
        print(f"  hook:    {out.get('hook', '')}")
        print(f"  overlay: {out['overlay']}")
    print(f"cost so far: ${llm.spent():.4f}")


if __name__ == "__main__":
    main()
