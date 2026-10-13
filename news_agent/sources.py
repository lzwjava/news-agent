"""Source adapters: RSS/Atom fetching and generic HTML list scraping."""

from __future__ import annotations

import html as html_lib
import re
import time
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import feedparser
import httpx
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

from .models import SourceConfig

USER_AGENT = "news-agent/0.1 (+https://github.com/lzw/news-agent; feed reader)"

TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "yclid", "mc_cid", "mc_eid", "ref", "cmpid", "s",
}

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
# feed descriptions that are navigation, not a summary
_NON_SUMMARIES = {"comments", "comment", "discuss", "discussion", "read more", "continue reading"}


def http_get(url: str, timeout: float = 20.0, retries: int = 2, user_agent: str | None = None) -> httpx.Response:
    """GET with retry/backoff - several feeds (HN, VentureBeat) rate-limit."""
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = httpx.get(
                url,
                headers={"User-Agent": user_agent or USER_AGENT, "Accept": "*/*"},
                timeout=timeout,
                follow_redirects=True,
            )
            if resp.status_code in {419, 425, 429, 500, 502, 503, 504} and attempt < retries:
                last_error = httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}", request=resp.request, response=resp
                )
                time.sleep(1.5 * (attempt + 1) ** 2)
                continue
            resp.raise_for_status()
            return resp
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1) ** 2)
                continue
            raise
    raise last_error if last_error else RuntimeError(f"GET failed: {url}")


def canonical_url(url: str) -> str:
    """Normalize a URL so the same story links dedupe together."""
    parts = urlsplit(url.strip())
    if not parts.scheme:
        return ""
    query = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if k.lower() not in TRACKING_PARAMS
    ]
    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), path, urlencode(query), "")
    )


def strip_html(value: str | None) -> str:
    if not value:
        return ""
    text = html_lib.unescape(value)
    text = _TAG_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def truncate(text: str, limit: int = 400) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(",.;: ") + "…"


def _feed_datetime(entry) -> str | None:
    for key in ("published_parsed", "updated_parsed", "created_parsed"):
        parsed = entry.get(key)
        if parsed:
            try:
                dt = datetime(*parsed[:6], tzinfo=timezone.utc)
                return dt.isoformat()
            except (TypeError, ValueError):
                continue
    for key in ("published", "updated", "date"):
        raw = entry.get(key)
        if raw:
            try:
                return _parse_date(raw)
            except (ValueError, TypeError):
                continue
    return None


def _parse_date(value: str) -> str | None:
    try:
        dt = date_parser.parse(value)
    except (ValueError, TypeError, OverflowError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _base_entry(src: SourceConfig) -> dict:
    return {
        "source_id": src.id,
        "source_name": src.name,
        "category": src.category,
        "weight": src.weight,
        "home": src.home or src.url,
        "keywords": src.keywords,
    }


def fetch_rss(src: SourceConfig, timeout: float = 20.0) -> list[dict]:
    resp = http_get(src.url, timeout=timeout, user_agent=src.user_agent)
    feed = feedparser.parse(resp.content)
    entries: list[dict] = []
    for entry in feed.entries[: src.max_items]:
        link = entry.get("link")
        if not link:
            continue
        title = strip_html(entry.get("title"))
        if not title:
            continue
        item = _base_entry(src)
        summary = truncate(strip_html(entry.get("summary") or entry.get("description")))
        if summary.strip().lower().rstrip(".") in _NON_SUMMARIES:
            summary = ""  # e.g. Hacker News: description is just the "Comments" link
        item.update(
            {
                "title": title,
                "url": urljoin(src.url, link),
                "published": _feed_datetime(entry),
                "summary": summary,
                "comments_url": entry.get("comments"),
            }
        )
        entries.append(item)
    return entries


def fetch_html(src: SourceConfig, timeout: float = 20.0) -> list[dict]:
    """Scrape an index page that has no feed.

    Configured with CSS selectors:
      * ``anchor`` - selector for the link elements
      * ``title``  - selector (inside the link) for the headline
      * ``date``   - selector for the date text
      * ``tag``    - selector for a topic/subject label
      * ``summary``- optional selector for a description
    """
    resp = http_get(src.url, timeout=timeout, user_agent=src.user_agent)
    soup = BeautifulSoup(resp.text, "html.parser")
    anchors = soup.select(src.anchor or "a[href]")
    seen: set[str] = set()
    entries: list[dict] = []

    for anchor in anchors:
        href = anchor.get("href")
        if not href:
            continue
        url = urljoin(src.url, href)
        if not url.startswith("http"):
            continue
        path = urlsplit(url).path
        if src.link_pattern and not re.search(src.link_pattern, path):
            continue
        if src.link_exclude and re.search(src.link_exclude, path):
            continue
        key = canonical_url(url)
        if not key or key in seen:
            continue

        node = anchor.select_one(src.title) if src.title else None
        title = strip_html(node.get_text(" ", strip=True) if node else anchor.get_text(" ", strip=True))
        if len(title) < 4:
            continue

        published = None
        if src.date:
            date_node = anchor.select_one(src.date)
            if date_node:
                published = _parse_date(date_node.get_text(" ", strip=True))

        tag = None
        if src.tag:
            tag_node = anchor.select_one(src.tag)
            if tag_node:
                tag = strip_html(tag_node.get_text(" ", strip=True)) or None

        summary = ""
        if src.summary:
            summary_node = anchor.select_one(src.summary)
            if summary_node:
                summary = strip_html(summary_node.get_text(" ", strip=True))

        seen.add(key)
        item = _base_entry(src)
        item.update(
            {
                "title": title,
                "url": url,
                "published": published,
                "summary": truncate(summary),
                "comments_url": None,
            }
        )
        if tag and tag.lower() != title.lower():
            item["source_tag"] = tag
        entries.append(item)
        if len(entries) >= src.max_items:
            break
    return entries


def fetch_source(src: SourceConfig, timeout: float = 20.0) -> list[dict]:
    """Dispatch to the right adapter."""
    if src.type == "html":
        return fetch_html(src, timeout)
    return fetch_rss(src, timeout)