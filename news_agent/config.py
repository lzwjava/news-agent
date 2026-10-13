"""Settings + source registry loading."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .models import SourceConfig

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCES_FILE = ROOT / "sources.yaml"
DEFAULT_OUT_DIR = ROOT / "output"


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    load_dotenv(ROOT / ".env", override=False)


@dataclass
class Settings:
    """Runtime configuration for one agent run (kept JSON-serializable: it is
    stored in LangGraph state)."""

    hours: int = 48
    top: int = 40
    per_section: int = 10
    per_source: int = 25
    use_llm: bool = True
    provider: str = ""  # empty -> first configured of openrouter/deepseek/openai/custom
    model: str = ""
    timeout: float = 20.0
    sources_file: str = str(DEFAULT_SOURCES_FILE)
    out_dir: str = str(DEFAULT_OUT_DIR)
    only: list[str] = field(default_factory=list)      # source id filter
    categories: list[str] = field(default_factory=list)  # category filter

    def to_dict(self) -> dict:
        return {
            "hours": self.hours,
            "top": self.top,
            "per_section": self.per_section,
            "per_source": self.per_source,
            "use_llm": self.use_llm,
            "provider": self.provider,
            "model": self.model,
            "timeout": self.timeout,
            "sources_file": self.sources_file,
            "out_dir": self.out_dir,
            "only": list(self.only),
            "categories": list(self.categories),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Settings":
        return cls(**data)


def load_sources(path: str | Path | None = None) -> list[SourceConfig]:
    """Read and validate the source registry."""
    _load_dotenv()
    path = Path(path or os.getenv("NEWS_AGENT_SOURCES") or DEFAULT_SOURCES_FILE)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = raw.get("sources", raw if isinstance(raw, list) else [])
    return [SourceConfig.model_validate(e) for e in entries]


def default_settings(**overrides) -> Settings:
    _load_dotenv()
    settings = Settings(
        provider=os.getenv("NEWS_AGENT_PROVIDER", ""),
        model=os.getenv("NEWS_AGENT_MODEL", ""),
        out_dir=os.getenv("NEWS_AGENT_OUT", str(DEFAULT_OUT_DIR)),
        sources_file=os.getenv("NEWS_AGENT_SOURCES", str(DEFAULT_SOURCES_FILE)),
    )
    for key, value in overrides.items():
        if value is not None:
            setattr(settings, key, value)
    return settings