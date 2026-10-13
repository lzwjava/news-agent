"""Writers: markdown digest, HN-style HTML feed, JSON and RSS output."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

from .config import Settings
from .state import AgentState


def _now() -> datetime:
    return datetime.now(timezone.utc)


def humanize(value: str | None) -> str:
    """ISO timestamp -> '3h ago'."""
    if not value:
        return "date n/a"
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return "date n/a"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    seconds = (_now() - dt).total_seconds()
    if seconds < 0:
        seconds = 0
    for unit, divisor in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= divisor:
            return f"{int(seconds // divisor)}{unit} ago"
    return "just now"


def _stats(state: AgentState) -> dict:
    fetch_log = state.get("fetch_log") or []
    return {
        "sources": len(fetch_log),
        "sources_ok": sum(1 for f in fetch_log if f.get("ok")),
        "fetched": sum(f.get("count", 0) for f in fetch_log),
        "kept": len(state.get("articles") or []),
        "errors": list(state.get("errors") or []),
        "fetch_log": fetch_log,
    }


# ------------------------------------------------------------------- markdown

def render_markdown(state: AgentState, digest: dict, stats: dict) -> str:
    settings = Settings.from_dict(state["settings"])
    lines = [
        f"# {digest['title']} — {digest['date']}",
        "",
        (
            f"> {stats['sources_ok']}/{stats['sources']} sources ok · {stats['fetched']} fetched · "
            f"{stats['kept']} kept (last {settings.hours}h) · generated "
            f"{digest['generated_at'][:19].replace('T', ' ')} UTC"
            + (f" · model `{digest['model']}`" if digest.get("llm_used") and digest.get("model") else "")
        ),
        "",
    ]
    for section in digest["sections"]:
        marker = "" if section["llm"] else " _(unedited)_"
        lines += [f"## {section['title']}{marker}", "", f"_{section['intro']}_", ""]
        for i, item in enumerate(section["items"], 1):
            age = humanize(item.get("published"))
            lines.append(f"{i}. [{item['title']}]({item['url']}) — `{item['source_name']}` · {age}")
            if item.get("tldr"):
                lines.append(f"   {item['tldr']}")
            if item.get("tags"):
                lines.append("   " + " ".join(f"`#{t}`" for t in item["tags"]))
            if item.get("comments_url"):
                lines.append(f"   [discussion]({item['comments_url']})")
        lines.append("")

    lines += ["---", "", "## Source health", ""]
    for entry in stats["fetch_log"]:
        status = f"{entry['count']} items" if entry.get("ok") else f"FAILED — {entry.get('error', 'unknown')}"
        lines.append(f"- **{entry['name']}** (`{entry['id']}`): {status}")
    if stats["errors"]:
        lines += ["", "## Errors", ""] + [f"- {e}" for e in stats["errors"]]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------- HTML

_CSS = """
:root{--bg:#f6f6ef;--card:#fff;--ink:#1a1a1a;--muted:#7a7a72;--accent:#ff6600;
--line:#e6e4d9;--badge:#efece1}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
header.top{background:var(--accent);color:#fff;padding:14px 20px}
header.top .wrap{max-width:920px;margin:0 auto;display:flex;align-items:baseline;gap:12px;flex-wrap:wrap}
header.top h1{margin:0;font-size:19px;letter-spacing:.2px}
header.top .sub{font-size:13px;opacity:.92}
main{max-width:920px;margin:0 auto;padding:22px 20px 60px}
.stats{display:flex;gap:10px;flex-wrap:wrap;margin:0 0 24px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 12px;font-size:13px}
.stat b{font-size:16px;display:block}
section.digest{margin-bottom:34px}
section.digest h2{font-size:20px;margin:0 0 4px}
section.digest .intro{color:var(--muted);margin:0 0 12px;font-size:14px}
.item{display:flex;gap:12px;background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:12px 14px;margin-bottom:10px}
.rank{color:var(--muted);font-variant-numeric:tabular-nums;min-width:26px;text-align:right;padding-top:1px}
.item .main{flex:1;min-width:0}
.title{font-size:16px;font-weight:600}
.title a{color:var(--ink);text-decoration:none}
.title a:hover{color:var(--accent)}
.domain{color:var(--muted);font-weight:400;font-size:13px}
.meta{color:var(--muted);font-size:12.5px;margin-top:3px;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.badge{background:var(--badge);border-radius:4px;padding:1px 7px;color:#5c5c52}
.tldr{margin-top:7px;font-size:14px;color:#33332e}
.tags{margin-top:6px;display:flex;gap:6px;flex-wrap:wrap}
.tag{font-size:11.5px;color:var(--accent);border:1px solid #ffd9bd;background:#fff6f0;
border-radius:20px;padding:1px 8px}
h3.block{font-size:15px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);
border-bottom:1px solid var(--line);padding-bottom:6px;margin:34px 0 14px}
table{width:100%;border-collapse:collapse;font-size:13.5px;background:var(--card);
border:1px solid var(--line);border-radius:10px;overflow:hidden}
td,th{padding:8px 12px;text-align:left;border-bottom:1px solid var(--line)}
th{background:#fbfaf5;color:var(--muted);font-weight:600}
tr:last-child td{border-bottom:none}
.err{color:#b3391a}
footer{color:var(--muted);font-size:12.5px;margin-top:30px}
@media (max-width:640px){.rank{display:none}.item{padding:11px}}
"""


def render_html(state: AgentState, digest: dict, stats: dict) -> str:
    settings = Settings.from_dict(state["settings"])
    esc = html.escape
    out: list[str] = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{esc(digest['title'])} · {esc(digest['date'])}</title>",
        f"<style>{_CSS}</style></head><body>",
        "<header class='top'><div class='wrap'>",
        f"<h1>▲ {esc(digest['title'])}</h1>",
        f"<span class='sub'>{esc(digest['date'])} · built with LangGraph"
        + (f" · {esc(digest['model'])}" if digest.get("llm_used") and digest.get("model") else "")
        + "</span></div></header><main>",
        "<div class='stats'>",
        f"<div class='stat'><b>{stats['kept']}</b>articles kept</div>",
        f"<div class='stat'><b>{stats['fetched']}</b>fetched</div>",
        f"<div class='stat'><b>{stats['sources_ok']}/{stats['sources']}</b>sources ok</div>",
        f"<div class='stat'><b>{settings.hours}h</b>time window</div>",
        "</div>",
    ]

    rank = 0
    for section in digest["sections"]:
        out.append("<section class='digest'>")
        out.append(f"<h2>{esc(section['title'])}</h2>")
        out.append(f"<p class='intro'>{esc(section['intro'])}</p>")
        for item in section["items"]:
            rank += 1
            out.append("<div class='item'>")
            out.append(f"<div class='rank'>{rank}.</div><div class='main'>")
            out.append(
                f"<div class='title'><a href='{esc(item['url'])}' target='_blank' rel='noopener'>"
                f"{esc(item['title'])}</a> <span class='domain'>({esc(item['domain'])})</span></div>"
            )
            meta = [
                f"<span class='badge'>{esc(item['source_name'])}</span>",
                f"<span>{esc(humanize(item.get('published')))}</span>",
                f"<span>▲ {item.get('score', 0):.1f}</span>",
            ]
            if item.get("comments_url"):
                meta.append(f"<a href='{esc(item['comments_url'])}' target='_blank' rel='noopener'>discussion</a>")
            out.append("<div class='meta'>" + "".join(meta) + "</div>")
            if item.get("tldr"):
                out.append(f"<div class='tldr'>{esc(item['tldr'])}</div>")
            if item.get("tags"):
                tags = "".join(f"<span class='tag'>#{esc(t)}</span>" for t in item["tags"])
                out.append(f"<div class='tags'>{tags}</div>")
            out.append("</div></div>")
        out.append("</section>")

    out.append("<h3 class='block'>Source health</h3><table><tr><th>Source</th><th>Category</th><th>Result</th></tr>")
    for entry in stats["fetch_log"]:
        result = (
            f"{entry['count']} items"
            if entry.get("ok")
            else f"<span class='err'>FAILED — {esc(str(entry.get('error', 'unknown')))}</span>"
        )
        out.append(
            f"<tr><td>{esc(entry['name'])}<br><span class='domain'>{esc(entry['id'])}</span></td>"
            f"<td>{esc(entry.get('category', ''))}</td><td>{result}</td></tr>"
        )
    out.append("</table>")

    # full ranked list, everything that made it through filtering
    by_source: dict[str, list[dict]] = {}
    for article in state.get("articles") or []:
        by_source.setdefault(article["source_name"], []).append(article)
    if by_source:
        out.append("<h3 class='block'>All ranked articles</h3><table>"
                   "<tr><th>#</th><th>Title</th><th>Source</th><th>Published</th><th>Score</th></tr>")
        for i, article in enumerate(
            sorted(state["articles"], key=lambda a: -a.get("score", 0)), 1
        ):
            out.append(
                f"<tr><td>{i}</td>"
                f"<td><a href='{esc(article['url'])}' target='_blank' rel='noopener'>{esc(article['title'])}</a></td>"
                f"<td>{esc(article['source_name'])}</td>"
                f"<td>{esc(humanize(article.get('published')))}</td>"
                f"<td>{article.get('score', 0):.1f}</td></tr>"
            )
        out.append("</table>")

    if stats["errors"]:
        out.append("<h3 class='block'>Errors</h3><ul>")
        out += [f"<li class='err'>{esc(e)}</li>" for e in stats["errors"]]
        out.append("</ul>")

    out.append(
        f"<footer>Generated {esc(digest['generated_at'][:19].replace('T', ' '))} UTC · "
        "news-agent (LangGraph) · sources from sources.yaml</footer>"
    )
    out.append("</main></body></html>")
    return "\n".join(out)


# --------------------------------------------------------------- json / rss / xml

def render_json(state: AgentState, digest: dict, stats: dict) -> str:
    settings = Settings.from_dict(state["settings"])
    payload = {
        "meta": {
            "generated_at": digest["generated_at"],
            "date": digest["date"],
            "window_hours": settings.hours,
            "model": digest.get("model") if digest.get("llm_used") else None,
            "stats": {k: v for k, v in stats.items() if k != "fetch_log"},
            "fetch_log": stats["fetch_log"],
        },
        "sections": digest["sections"],
        "articles": state.get("articles") or [],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def render_rss(digest: dict) -> str:
    items = [item for section in digest["sections"] for item in section["items"]]
    out = [
        "<?xml version='1.0' encoding='UTF-8'?>",
        "<rss version='2.0'><channel>",
        f"<title>{xml_escape(digest['title'])}</title>",
        f"<lastBuildDate>{xml_escape(digest['generated_at'])}</lastBuildDate>",
        "<description>AI news digest generated by a LangGraph agent.</description>",
    ]
    for item in items:
        pub = item.get("published") or digest["generated_at"]
        try:
            pub_date = datetime.fromisoformat(pub.replace("Z", "+00:00")).strftime("%a, %d %b %Y %H:%M:%S +0000")
        except ValueError:
            pub_date = pub
        description = item.get("tldr") or item["title"]
        out += [
            "<item>",
            f"<title>{xml_escape(item['title'])}</title>",
            f"<link>{xml_escape(item['url'])}</link>",
            f"<guid isPermaLink='false'>{xml_escape(item['id'])}</guid>",
            f"<pubDate>{xml_escape(pub_date)}</pubDate>",
            f"<source url='{xml_escape(item['url'])}'>{xml_escape(item['source_name'])}</source>",
            f"<description>{xml_escape(description)}</description>",
            "</item>",
        ]
    out += ["</channel></rss>"]
    return "\n".join(out)


# ------------------------------------------------------------------- graph node

def render(state: AgentState) -> dict:
    settings = Settings.from_dict(state["settings"])
    out_dir = Path(settings.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    digest = state.get("digest") or {"sections": [], "date": _now().strftime("%Y-%m-%d"), "generated_at": _now().isoformat(), "title": "AI News Digest", "llm_used": False, "model": ""}
    stats = _stats(state)

    files = {
        "markdown": out_dir / f"digest-{digest['date']}.md",
        "html": out_dir / f"index-{digest['date']}.html",
        "json": out_dir / f"digest-{digest['date']}.json",
        "rss": out_dir / "feed.xml",
        "latest_html": out_dir / "index.html",
        "latest_markdown": out_dir / "digest.md",
    }

    markdown = render_markdown(state, digest, stats)
    page = render_html(state, digest, stats)
    files["markdown"].write_text(markdown, encoding="utf-8")
    files["latest_markdown"].write_text(markdown, encoding="utf-8")
    files["html"].write_text(page, encoding="utf-8")
    files["latest_html"].write_text(page, encoding="utf-8")
    files["json"].write_text(render_json(state, digest, stats), encoding="utf-8")
    files["rss"].write_text(render_rss(digest), encoding="utf-8")

    return {"outputs": {key: str(path) for key, path in files.items()}}