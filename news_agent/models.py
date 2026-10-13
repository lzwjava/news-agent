"""Data models: source registry entries, articles and digest structures."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SourceConfig(BaseModel):
    """A single entry from ``sources.yaml``."""

    id: str
    name: str
    url: str
    type: Literal["rss", "html"] = "rss"
    category: str = "media"
    weight: float = 1.0
    home: str | None = None
    max_items: int = 25
    # per-source override for picky bot filters (e.g. Cloudflare fronts)
    user_agent: str | None = None
    # rss keyword pre-filter (title must contain at least one)
    keywords: list[str] | None = None
    # html adapter selectors
    anchor: str | None = None
    title: str | None = None
    date: str | None = None
    tag: str | None = None
    summary: str | None = None
    # optional regex filters applied to the URL path of scraped links
    link_pattern: str | None = None
    link_exclude: str | None = None


class Article(BaseModel):
    """Normalized article, JSON-serializable so it can live in graph state."""

    id: str
    title: str
    url: str
    domain: str
    source_id: str
    source_name: str
    category: str
    weight: float = 1.0
    published: str | None = None  # ISO-8601
    summary: str = ""
    tags: list[str] = Field(default_factory=list)
    comments_url: str | None = None
    score: float = 0.0


class SectionSpec(BaseModel):
    """Static description of a digest section (built from article categories)."""

    id: str
    title: str
    description: str


# ----------------------------------------------------------------- LLM output


class DraftItem(BaseModel):
    """One edited entry. ``index`` refers to the item position sent to the model."""

    index: int = Field(ge=0)
    tldr: str = Field(description="One or two factual sentences, max 240 characters")
    tags: list[str] = Field(default_factory=list, description="1-3 lowercase topic tags")


class SectionDraft(BaseModel):
    """Structured output requested from the LLM for one section."""

    headline: str = Field(description="Punchy section headline, max 90 characters")
    intro: str = Field(description="One sentence of editorial framing, max 220 characters")
    items: list[DraftItem] = Field(description="Edited entries, only indexes that matter")