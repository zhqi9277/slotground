from __future__ import annotations
from collections.abc import Sequence
from slotground.types import EvidenceItem, Slot, TaskInput

def rrf_fuse(rankings: Sequence[Sequence[str]], k: int=60) -> tuple[tuple[str, float], ...]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, eid in enumerate(ranking, start=1):
            scores[eid] = scores.get(eid, 0.0) + 1.0 / (k + rank)
    return tuple(sorted(scores.items(), key=lambda pair: (-pair[1], pair[0])))
CHARS_PER_TOKEN = 4

def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)

def truncate_to_budget(ranked: Sequence[tuple[EvidenceItem, float]], budget_tokens: int) -> tuple[EvidenceItem, ...]:
    kept: list[EvidenceItem] = []
    used = 0
    for item, _score in ranked:
        cost = estimate_tokens(item.text)
        if used + cost > budget_tokens:
            continue
        kept.append(item)
        used += cost
    return tuple(kept)

def serialize_query(task: TaskInput, slots: Sequence[Slot]) -> str:
    parts = [str(task.tool_schema.get('name', '')), str(task.tool_schema.get('description', ''))]
    parts.extend((part for slot in slots for part in (slot.name, slot.description)))
    parts.append(task.query)
    return ' '.join((part for part in parts if part))

def parameter_queries(task: TaskInput, slots: Sequence[Slot]) -> tuple[str, ...]:
    return tuple((serialize_query(task, (slot,)) for slot in slots)) if slots else (serialize_query(task, slots),)

def select_evidence(evidence: Sequence[EvidenceItem], lexical_rankings: Sequence[Sequence[str]], dense_rankings: Sequence[Sequence[str]], candidate_cap=10, budget_tokens=600, rrf_k=60) -> tuple[EvidenceItem, ...]:
    if candidate_cap < 1 or rrf_k < 1 or budget_tokens < 0:
        raise ValueError('Invalid retrieval capacity, fusion constant, or budget')
    if len(lexical_rankings) != len(dense_rankings):
        raise ValueError('Lexical and dense query counts differ')
    by_id = {item.evidence_id: item for item in evidence}
    if len(by_id) != len(evidence):
        raise ValueError('Duplicate evidence identifiers')
    rankings = [tuple((eid for eid, score in rrf_fuse((lexical, dense), k=rrf_k))) for lexical, dense in zip(lexical_rankings, dense_rankings, strict=True)]
    ranked = rrf_fuse(rankings, k=rrf_k)[:candidate_cap]
    if any((eid not in by_id for eid, score in ranked)):
        raise ValueError('Ranking contains an unavailable evidence identifier')
    return truncate_to_budget([(by_id[eid], score) for eid, score in ranked], budget_tokens)
