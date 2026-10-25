from __future__ import annotations
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field
from slotground.grounding.compile_state import validate_parameter_value
from slotground.grounding.field_candidates import build_field_candidates
from slotground.types import EvidenceItem, TaskInput
Operation = Literal['mention', 'set', 'remove', 'focus']
_UPDATE = re.compile('\\b(?:chang\\w*|updat\\w*|replac\\w*|switch\\w*|instead|now|set|rather)\\b|\u6539|\u6362|\u8c03\u6574', re.I)
_REMOVE = re.compile('\\b(?:remove|cancel|drop|without|no longer|not need)\\b|\u53d6\u6d88|\u4e0d\u8981', re.I)
_FOCUS = re.compile('\\b(?:then|after|next|follow|also|instead|switch)\\w*\\b|\u63a5\u7740|\u7136\u540e', re.I)
_RECENCY = re.compile('\\b(?:latest|current|most recent|updated)\\b|\u6700\u65b0|\u76ee\u524d', re.I)
_LANGUAGES = {'english': 'en', 'spanish': 'es', 'french': 'fr', 'german': 'de', 'chinese': 'zh', 'japanese': 'ja', 'korean': 'ko', 'italian': 'it', 'portuguese': 'pt', 'russian': 'ru', 'arabic': 'ar', 'hindi': 'hi'}

class ExtractedObservation(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    slot: str
    value_json: str
    raw_value: str
    evidence_id: str
    quote: str
    entity: str = ''
    entity_evidence_id: str = ''
    entity_quote: str = ''
    group: str = ''
    operation: Operation = 'mention'
    operation_quote: str = ''

class ObservationOutput(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    observations: tuple[ExtractedObservation, ...] = Field(default=(), max_length=48)
    references: tuple[QueryReference, ...] = ()

@dataclass(frozen=True)
class QueryReference:
    query_quote: str
    entity: str
    evidence_id: str
    evidence_quote: str
ObservationOutput.model_rebuild()

@dataclass(frozen=True)
class Observation:
    slot: str
    value: Any
    evidence_id: str
    quote: str
    span_start: int
    span_end: int
    entity: str
    group: str
    entity_evidence_id: str
    block_id: str
    session_id: str
    turn: int
    operation: Operation
    operation_quote: str
    source: str = 'semantic_extractor'

@dataclass(frozen=True)
class CandidatePool:
    observations: tuple[Observation, ...]
    rejected: tuple[str, ...]
    references: tuple[QueryReference, ...] = ()

@dataclass(frozen=True)
class StateOption:
    slot: str
    value: Any
    omit: bool
    entity: str
    group: str
    scope: str
    evidence_ids: tuple[str, ...]
    unary: float
    object_score: float
    update_score: float
    source: str

def value_key(value: Any) -> str:
    if isinstance(value, str):
        return ' '.join(value.casefold().split())
    return json.dumps(value, sort_keys=True, ensure_ascii=False)

def _contains(text: str, phrase: str) -> bool:
    return bool(phrase.strip()) and bool(re.search('(?<!\\w)' + re.escape(phrase.casefold().strip()) + '(?!\\w)', text.casefold()))

def _supported(value: Any, raw: str, spec: Mapping[str, Any]) -> bool:
    surface = raw.strip().strip('"\'')
    if value_key(value) == value_key(surface):
        return True
    if isinstance(value, bool):
        if surface.casefold() in ({'true', 'yes', 'enabled', 'on'} if value else {'false', 'no', 'disabled', 'off'}):
            return True
        description = str(spec.get('description', ''))
        if re.search('include|enable|allow|whether', description, re.I):
            return bool(re.search('\\b(?:include|including|with|enable)\\b' if value else '\\b(?:without|exclude|excluding|disable)\\b', surface, re.I))
        return False
    if isinstance(value, int | float):
        cleaned = re.sub('[$\u20ac\u00a3, ]', '', surface)
        try:
            return float(cleaned) == float(value)
        except ValueError:
            return False
    if isinstance(value, str):
        if _contains(surface, value):
            return True
        if _LANGUAGES.get(surface.casefold()) == value.casefold() and re.search('language|locale|iso.?639', str(spec.get('description', '')), re.I):
            return True
        for fmt in ('%I:%M %p', '%I %p', '%I:%M%p', '%I%p', '%H:%M'):
            try:
                if datetime.strptime(surface, fmt).replace(tzinfo=UTC).strftime('%H:%M') == value:
                    return True
            except ValueError:
                pass
        for fmt in ('%B %d, %Y', '%b %d, %Y', '%d %B %Y', '%m/%d/%Y', '%Y-%m-%d'):
            try:
                if datetime.strptime(surface, fmt).replace(tzinfo=UTC).strftime('%Y-%m-%d') == value:
                    return True
            except ValueError:
                pass
    try:
        return bool(json.loads(surface) == value)
    except ValueError:
        return False

def ground_observations(task: TaskInput, evidence: Sequence[EvidenceItem], extracted: ObservationOutput) -> CandidatePool:
    schema = task.tool_schema.get('parameters') or task.tool_schema.get('input_schema') or {}
    properties = schema.get('properties', {})
    sources = {item.evidence_id: item for item in evidence}
    text = {key: item.text for key, item in sources.items()}
    text['__current__'] = task.query
    accepted: list[Observation] = []
    rejected = []
    seen = set()
    for raw in extracted.observations:
        reason = ''
        if raw.slot not in properties or raw.evidence_id not in text:
            reason = 'unknown_field_or_source'
        elif not raw.quote or raw.quote not in text[raw.evidence_id]:
            reason = 'quote_not_in_source'
        if reason:
            rejected.append(reason)
            continue
        try:
            parsed = json.loads(raw.value_json)
            kind = properties[raw.slot].get('type')
            if isinstance(parsed, bool) and isinstance(kind, str) and (kind in {'str', 'string'}):
                parsed = json.dumps(parsed)
            value = validate_parameter_value(parsed, properties[raw.slot], raw.slot)
        except (ValueError, TypeError):
            rejected.append('invalid_schema_value')
            continue
        surface = raw.raw_value
        if not surface or surface not in raw.quote:
            literal = str(value) if not isinstance(value, bool) else json.dumps(value)
            if _contains(raw.quote, literal):
                surface = literal
            else:
                rejected.append('value_not_in_quote')
                continue
        if not _supported(value, surface, properties[raw.slot]):
            rejected.append('unsupported_normalization')
            continue
        item = sources.get(raw.evidence_id)
        entity_source = sources.get(raw.entity_evidence_id)
        same_scope = raw.entity_evidence_id == '__current__' or (item is not None and entity_source is not None and ((item.block_id, item.session_id) == (entity_source.block_id, entity_source.session_id)))
        entity = raw.entity if same_scope and raw.entity_quote and (raw.entity in raw.entity_quote) and (raw.entity_quote in text.get(raw.entity_evidence_id, '')) else ''
        operation = raw.operation
        cue = raw.operation_quote
        patterns = {'set': _UPDATE, 'remove': _REMOVE, 'focus': _FOCUS}
        if operation != 'mention' and (not cue or cue not in text[raw.evidence_id] or (not patterns[operation].search(cue))):
            operation, cue = ('mention', '')
        if raw.evidence_id == '__current__':
            block, session, turn = ('__current__', '__current__', 1000000)
        else:
            assert item is not None
            block, session, turn = (item.block_id, item.session_id, item.turn)
        start = text[raw.evidence_id].index(raw.quote)
        identity = (raw.slot, value_key(value), raw.evidence_id, start, entity, operation)
        if identity in seen:
            continue
        seen.add(identity)
        accepted.append(Observation(raw.slot, value, raw.evidence_id, raw.quote, start, start + len(raw.quote), entity, raw.group.casefold().strip() if entity else '', raw.entity_evidence_id, block, session, turn, operation, cue))
    references = tuple((ref for ref in extracted.references if ref.query_quote and ref.query_quote in task.query and ref.entity and (ref.entity in ref.evidence_quote) and (ref.evidence_quote in text.get(ref.evidence_id, ''))))
    return CandidatePool(tuple(accepted), tuple(rejected), references)

def augment_literal_fields(task: TaskInput, evidence: Sequence[EvidenceItem], pool: CandidatePool) -> CandidatePool:
    schema = task.tool_schema.get('parameters') or task.tool_schema.get('input_schema') or {}
    by_id = {item.evidence_id: item for item in evidence}
    observations = list(pool.observations)
    seen = {(o.slot, value_key(o.value), o.evidence_id) for o in observations}
    for slot, spec in schema.get('properties', {}).items():
        for candidate in build_field_candidates(task, slot, spec, evidence, None):
            if candidate.source not in {'json_field', 'key_value', 'natural_field'}:
                continue
            if not _supported(candidate.value, candidate.raw_span, spec):
                continue
            for evidence_id in candidate.evidence_ids:
                item = by_id.get(evidence_id)
                key = (slot, value_key(candidate.value), evidence_id)
                if item is None or key in seen:
                    continue
                if candidate.raw_span not in item.text:
                    continue
                start = item.text.index(candidate.raw_span)
                observations.append(Observation(slot, candidate.value, evidence_id, candidate.raw_span, start, start + len(candidate.raw_span), '', '', '', item.block_id, item.session_id, item.turn, 'mention', '', candidate.source))
                seen.add(key)
    return replace(pool, observations=tuple(observations))

def build_options(task: TaskInput, pool: CandidatePool, baseline: Mapping[str, Any]) -> dict[str, tuple[StateOption, ...]]:
    schema = task.tool_schema.get('parameters') or task.tool_schema.get('input_schema') or {}
    result = {}
    for slot in schema.get('properties', {}):
        observed = [o for o in pool.observations if o.slot == slot]
        reference_entities = {ref.entity.casefold() for ref in pool.references}
        anchors = {o.entity for o in observed if o.entity and (_contains(task.query, o.entity) or o.entity.casefold() in reference_entities)}
        current = [o for o in observed if o.evidence_id == '__current__']
        options = []
        for obs in observed:
            scope = obs.block_id + ':' + obs.session_id
            is_reader = slot in baseline and value_key(baseline[slot]) == value_key(obs.value)
            unary = 0.7 + 0.45 * is_reader
            if obs.evidence_id == '__current__':
                unary += 5.0
            object_score = (2.0 if obs.entity in anchors else -1.0) if anchors else 0.0
            update_score = 0.0
            related = [other for other in observed if other.entity == obs.entity and other.block_id == obs.block_id and (other.session_id == obs.session_id)]
            newer_updates = [other for other in related if other.turn > obs.turn and other.operation in {'set', 'remove'}]
            if obs.operation in {'set', 'remove'}:
                update_score += 1.5
            if newer_updates:
                update_score -= 2.0
            if _RECENCY.search(task.query) and related:
                update_score += 0.5 * (obs.turn == max((o.turn for o in related)))
            options.append(StateOption(slot, obs.value, obs.operation == 'remove', obs.entity, obs.group, scope, (obs.evidence_id,), unary, object_score, update_score, obs.source))
        if current:
            options = [o for o in options if o.evidence_ids == ('__current__',)]
        else:
            options.append(StateOption(slot, baseline.get(slot), slot not in baseline, '', '', '', (), 1.0 if slot in baseline else 0.8, 0.0, 0.0, 'reader_fallback'))
        result[slot] = tuple(options)
    return result
