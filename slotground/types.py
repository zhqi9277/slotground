from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, field_validator

class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid', strict=False)
EvidenceSource = Literal['chunk', 'fact', 'query']

class Slot(Frozen):
    name: str
    type: str
    required: bool
    default: str | None = None
    description: str = ''
    enum: tuple[str, ...] = ()

class EvidenceItem(Frozen):
    evidence_id: str
    session_id: str
    turn: int
    text: str
    block_id: str
    source: EvidenceSource = 'chunk'

    @field_validator('evidence_id')
    @classmethod
    def _nonempty(cls, v: str) -> str:
        if not v:
            raise ValueError('evidence_id must not be empty')
        return v

class TaskInput(Frozen):
    task_id: str
    query: str
    tool_schema: dict[str, Any]
    candidate_tools: tuple[dict[str, Any], ...]
    history_id: str
    no_history: bool
