"""LangGraph node + router functions.

The graph looks like:

    START -> select_sources --Send--> fetch        (one parallel branch per feed)
             fetch* -> normalize -> rank -> build_sections
             build_sections --Send--> summarize_section  (one LLM editor per section)
             or straight to compile_digest when the LLM is off
             compile_digest -> render -> END
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

from langgraph.types import Send

from .config import Settings, load_sources
from .llm import build_llm
from .models import Article, SectionDraft, SectionSpec, SourceConfig
from .sources import canonical_url, fetch_source, truncate
from .state import AgentState

# --------------------------------------------------------------- section map

SECTIONS: dict[str, SectionSpec] = {
    "labs": SectionSpec(
        id="labs",
        title="Labs & Product",
        description="Official announcements from the frontier labs and AI products.",
    ),
    "research": SectionSpec(
        id="research",
        title="Research & Technical",
        description="Papers, deep dives and hands-on technical write-ups.",
    ),
    "community": SectionSpec(
        id="community",
        title="Community: YC & Hacker News",
        description="What the builders are upvoting and shipping this week.",
    ),
    "media": SectionSpec(
        id="media",
        title="Industry & AI News",
        description="The AI press: funding, launches, policy and analysis.",
    ),
}
SECTION_ORDER = ["labs", "research", "community", "media"]

_STOPWORDS = {"the", "a", "an", "of", "for", "and", "to", "in", "on", "with", "is", "by", "how", "why", "what"}
_KW_CACHE: dict[str, re.Pattern[str]] = {}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _settings(state: AgentState) -> Settings:
    return Settings.from_dict(state.get("settings") or {})


# -------------------------------------------------------------- node: select

def select_sources(state: AgentState) -> dict:
    """Apply the --only / --category filters to the registry."""
    settings = _settings(state)
    sources = load_sources(settings.sources_file)
    if settings.only:
        wanted = {s.lower() for s in settings.only}
        sources = [s for s in sources if s.id.lower() in wanted]
    if settings.categories:
        wanted = {c.lower() for c in settings.categories}
        sources = [s for s in sources if s.category.lower() in wanted]
    if not sources:
        raise ValueError(
            "no sources matched the given --only/--category filters "
            f"(only={settings.only}, categories={settings.categories})"
        )
    return {"sources": [s.model_dump() for s in sources]}


def route_fetch(state: AgentState):
    """Dynamic fan-out: one parallel branch per selected source."""
    return [Send("fetch", {"source": src}) for src in state["sources"]]


# ----------------------------------------------------------------- node: fetch

def fetch_node(state: AgentState) -> dict:
    src = SourceConfig.model_validate(state["source"])
    settings = _settings(state)
    log = {"id": src.id, "name": src.name, "category": src.category, "count": 0, "ok": False}
    try:
        entries = fetch_source(src, timeout=settings.timeout)[: settings.per_source]
    except Exception as exc:  # network / parse errors must never kill the run
        log["error"] = str(exc)[:200]
        return {"raw_entries": [], "fetch_log": [log], "errors": [f"{src.id}: {exc}"]}
    log["count"] = len(entries)
    log["ok"] = True
    return {"raw_entries": entries, "fetch_log": [log]}


# --------------------------------------------------------------- node: normalize

def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    pattern = _KW_CACHE.get(keyword)
    if pattern is None:
        if re.fullmatch(r"[a-z0-9]+", keyword):
            pattern = re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)
        else:
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        _KW_CACHE[keyword] = pattern
    return pattern


def _keywords_match(title: str, keywords: list[str]) -> bool:
    return any(_keyword_pattern(k).search(title) for k in keywords)


def _tokens(title: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", title.lower()) if len(w) > 2 and w not in _STOPWORDS}


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def normalize(state: AgentState) -> dict:
    """Filter by time window + keywords, dedupe by URL and near-duplicate titles."""
    settings = _settings(state)
    now = _now()
    cutoff = now - timedelta(hours=settings.hours)
    seen_urls: set[str] = set()
    seen_titles: list[set[str]] = []
    articles: list[Article] = []

    for raw in state.get("raw_entries") or []:
        url = canonical_url(raw.get("url") or "")
        title = re.sub(r"\s+", " ", raw.get("title") or "").strip()
        if not url or url in seen_urls or len(title) < 4:
            continue
        if raw.get("keywords") and not _keywords_match(title, raw["keywords"]):
            continue

        published = _parse_dt(raw.get("published"))
        if published and published < cutoff:
            continue  # outside the time window

        tokens = _tokens(title)
        if tokens and any(len(tokens & other) / len(tokens | other) >= 0.7 for other in seen_titles):
            continue  # same story picked up by several outlets

        seen_urls.add(url)
        if len(tokens) >= 3:
            seen_titles.append(tokens)

        parts = urlsplit(url)
        tags = [t for t in [raw.get("source_tag")] if t]
        articles.append(
            Article(
                id=f"{raw.get('source_id', 'x')}-{hashlib.sha1(url.encode()).hexdigest()[:10]}",
                title=title,
                url=url,
                domain=parts.netloc.removeprefix("www."),
                source_id=raw.get("source_id", "unknown"),
                source_name=raw.get("source_name", "Unknown"),
                category=raw.get("category", "media"),
                weight=float(raw.get("weight", 1.0)),
                published=published.isoformat() if published else None,
                summary=truncate(raw.get("summary") or "", 400),
                tags=tags,
                comments_url=raw.get("comments_url"),
            )
        )

    return {"articles": [a.model_dump(mode="json") for a in articles]}


# ------------------------------------------------------------------ node: rank

def rank(state: AgentState) -> dict:
    """Recency-decayed score on top of the source weight.

    Keeps the global top N but reserves ``per_section`` slots per category so a
    loud source (labs) can never crowd every other section out of the digest.
    """
    settings = _settings(state)
    now = _now()
    scored = []
    for article in state.get("articles") or []:
        item = dict(article)
        published = _parse_dt(item.get("published"))
        age_hours = (now - published).total_seconds() / 3600 if published else settings.hours
        recency = max(0.0, 48.0 - age_hours) * 0.4
        discussion = 2.0 if item.get("comments_url") else 0.0
        item["score"] = round(item["weight"] * 10 + recency + discussion, 2)
        scored.append(item)

    scored.sort(key=lambda a: (-a["score"], a.get("published") or ""))

    kept: list[dict] = []
    kept_ids: set[str] = set()
    by_category: dict[str, list[dict]] = {}
    for item in scored:
        by_category.setdefault(item["category"], []).append(item)
    for items in by_category.values():
        for item in items[: settings.per_section]:
            kept.append(item)
            kept_ids.add(item["id"])
    for item in scored:
        if len(kept) >= settings.top:
            break
        if item["id"] not in kept_ids:
            kept.append(item)
            kept_ids.add(item["id"])

    kept.sort(key=lambda a: (-a["score"], a.get("published") or ""))
    return {"articles": kept[: settings.top]}


# ---------------------------------------------------------- node: build_sections

def build_sections(state: AgentState) -> dict:
    settings = _settings(state)
    groups: dict[str, list[dict]] = {}
    for article in state.get("articles") or []:
        groups.setdefault(article["category"], []).append(article)

    order = SECTION_ORDER + [c for c in groups if c not in SECTION_ORDER]
    sections = []
    for category in order:
        items = groups.get(category) or []
        if not items:
            continue
        spec = SECTIONS.get(
            category,
            SectionSpec(id=category, title=category.title(), description=f"{category.title()} stories."),
        )
        sections.append(
            {
                "id": spec.id,
                "title": spec.title,
                "description": spec.description,
                "articles": items[: settings.per_section],
            }
        )
    return {"sections": sections}


def route_digest(state: AgentState):
    """Fan the sections out to parallel LLM editors, or skip straight to compile."""
    settings = _settings(state)
    llm, _ = build_llm(settings)
    if llm is None or not state.get("sections"):
        return "compile_digest"
    return [Send("summarize_section", {"section": s}) for s in state["sections"]]


# ------------------------------------------------------- node: summarize_section

def _article_payload(articles: list[dict]) -> list[dict]:
    payload = []
    for i, article in enumerate(articles):
        payload.append(
            {
                "i": i,
                "title": article["title"],
                "source": article["source_name"],
                "published": (article.get("published") or "")[:10],
                "summary": truncate(article.get("summary") or "", 300),
            }
        )
    return payload


_SYSTEM_PROMPT = """You edit a daily AI news digest with the tone of a Hacker News front page: \
factual, specific, zero hype, no emojis, always English.

You receive a JSON array of articles for ONE section; every article has an "i" index.
Return a structured object with:
- headline: section headline, max 90 chars, concrete (who did what).
- intro: one sentence, max 220 chars, editorial framing for the section.
- items: the entries worth featuring, each with:
    index: the article's "i" value (never invent indexes, never repeat one)
    tldr: 1-2 factual sentences, max 240 chars, do not restate the title
    tags: 1-3 lowercase topic tags, e.g. "llm", "funding", "agents", "safety"
Skip thin, duplicated or off-topic articles."""


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


def summarize_section(state: AgentState) -> dict:
    """One LLM editor per section (parallel). Never raises: failures degrade."""
    section = state["section"]
    settings = _settings(state)
    section_id = section["id"]
    llm, reason = build_llm(settings)
    if llm is None:
        return {"drafts": [{"section_id": section_id, "ok": False, "reason": reason}]}

    articles = section["articles"]
    human = (
        f"Section: {section['title']}\nPurpose: {section['description']}\n"
        f"Articles:\n{json.dumps(_article_payload(articles), ensure_ascii=False, indent=1)}"
    )
    messages = [("system", _SYSTEM_PROMPT), ("human", human)]

    try:
        draft: SectionDraft = llm.with_structured_output(SectionDraft).invoke(messages)
    except Exception:
        try:
            schema = json.dumps(SectionDraft.model_json_schema())
            raw = llm.invoke(
                messages
                + [("human", f"Return ONLY a JSON object matching this schema, no prose:\n{schema}")]
            )
            draft = SectionDraft.model_validate_json(_extract_json(raw.content if hasattr(raw, "content") else str(raw)))
        except Exception as exc:
            return {"drafts": [{"section_id": section_id, "ok": False, "reason": str(exc)[:300]}]}

    items = [
        {"index": it.index, "tldr": truncate(it.tldr, 300), "tags": [t.lower().strip()[:24] for t in it.tags if t][:3]}
        for it in draft.items
        if 0 <= it.index < len(articles)
    ]
    return {
        "drafts": [
            {
                "section_id": section_id,
                "ok": True,
                "headline": draft.headline.strip()[:120],
                "intro": truncate(draft.intro.strip(), 300),
                "items": items,
            }
        ]
    }


# ------------------------------------------------------------ node: compile_digest

def _fallback_tldr(article: dict) -> str:
    summary = re.sub(r"\s+", " ", article.get("summary") or "").strip()
    return truncate(summary, 240)


def compile_digest(state: AgentState) -> dict:
    """Merge the per-section drafts (or fall back to heuristics) into the digest."""
    settings = _settings(state)
    drafts = {
        d.get("section_id"): d
        for d in state.get("drafts") or []
        if d.get("section_id")
    }
    errors = []
    sections_out = []

    for section in state.get("sections") or []:
        articles = section["articles"]
        draft = drafts.get(section["id"])
        used: set[int] = set()
        items: list[dict] = []
        headline, intro = section["title"], section["description"]
        llm_used = bool(draft and draft.get("ok"))

        if llm_used:
            headline = (draft.get("headline") or headline).strip()
            intro = (draft.get("intro") or intro).strip()
            for entry in draft.get("items") or []:
                index = entry.get("index")
                if not isinstance(index, int) or not (0 <= index < len(articles)) or index in used:
                    continue
                used.add(index)
                item = dict(articles[index])
                item["tldr"] = entry.get("tldr") or _fallback_tldr(item)
                item["tags"] = entry.get("tags") or []
                items.append(item)
        elif draft and draft.get("reason"):
            errors.append(f"digest[{section['id']}]: {draft['reason']}")

        # everything the model did not feature still lands in the feed
        for index, article in enumerate(articles):
            if index in used:
                continue
            item = dict(article)
            item["tldr"] = _fallback_tldr(item)
            item.setdefault("tags", [])
            items.append(item)

        sections_out.append(
            {
                "id": section["id"],
                "title": headline,
                "source_title": section["title"],
                "intro": intro,
                "items": items,
                "llm": llm_used,
            }
        )

    now = _now()
    digest = {
        "title": "AI News Digest",
        "date": now.strftime("%Y-%m-%d"),
        "generated_at": now.isoformat(),
        "model": settings.model or "",
        "llm_used": any(s["llm"] for s in sections_out),
        "sections": sections_out,
    }
    updates: dict = {"digest": digest}
    if errors:
        updates["errors"] = errors
    return updates