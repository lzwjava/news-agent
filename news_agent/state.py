"""LangGraph state schema.

Every channel is JSON-serializable so the state can be checkpointed.
List channels that receive writes from parallel branches use ``operator.add``
reducers, which is what lets the fan-out nodes merge safely.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict, total=False):
    # run configuration (Settings.to_dict())
    settings: dict

    # select_sources -> router (fan-out over these)
    sources: list[dict]

    # transient: the source assigned to a branch by Send()
    source: dict | None

    # written by the parallel "fetch" branches
    raw_entries: Annotated[list[dict], operator.add]
    fetch_log: Annotated[list[dict], operator.add]
    errors: Annotated[list[str], operator.add]

    # normalize / rank (single writer, replaced each time)
    articles: list[dict]

    # build_sections -> router (fan-out over these)
    sections: list[dict]

    # transient: the section assigned to a branch by Send()
    section: dict | None

    # written by the parallel "summarize_section" branches
    drafts: Annotated[list[dict], operator.add]

    # compile_digest -> render
    digest: dict | None

    # render -> END
    outputs: dict