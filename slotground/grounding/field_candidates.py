from __future__ import annotations
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any
from slotground.grounding.compile_state import validate_parameter_value
from slotground.types import EvidenceItem, TaskInput
_TOKEN_RE = re.compile("[A-Za-z0-9][A-Za-z0-9_./@:%+'-]*|[\\u4e00-\\u9fff]")
_NUMBER_RE = re.compile('(?<![A-Za-z0-9])[-+]?\\d+(?:\\.\\d+)?(?![A-Za-z0-9])')
_QUOTED_RE = re.compile('(?P<quote>[\'\\"])(?P<value>[^\'\\"\\n]{1,180})(?P=quote)')
_SPLIT_RE = re.compile('[.!?;\u3002\uff01\uff1f\uff1b\\n]+')
_KEY_VALUE_RE = re.compile('(?:[`*\\"\']?\\s*(?P<key>[A-Za-z][A-Za-z0-9_. -]{0,60})\\s*[`*\\"\']?\\s*)(?P<sep>[:=])\\s*(?P<value>\\"[^\\"\\n]{1,180}\\"|\'[^\'\\n]{1,180}\'|[^,;\\n}]{1,180})', re.IGNORECASE)
_CUE_RE = re.compile('(?:is|was|are|were|equals?|set\\s+to|called|named|known\\s+as|stands\\s+for|of)', re.IGNORECASE)
_VALUE_TOKEN_RE = re.compile("^[A-Za-z0-9][A-Za-z0-9_./@:%+'#-]*$")
_STOPWORDS = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'can', 'could', 'did', 'do', 'does', 'for', 'from', 'get', 'give', 'go', 'have', 'if', 'in', 'is', 'it', 'like', 'me', 'my', 'of', 'on', 'or', 'please', 'pull', 'show', 'some', 'that', 'the', 'their', 'this', 'to', 'up', 'use', 'want', 'we', 'what', 'which', 'with', 'you', 'your'}
_LANGUAGE_WORDS = {'arabic', 'chinese', 'dutch', 'english', 'french', 'german', 'hindi', 'italian', 'japanese', 'korean', 'portuguese', 'russian', 'spanish', 'turkish'}

@dataclass(frozen=True)
class FieldCandidate:
    slot: str
    value: Any
    source: str
    evidence_ids: tuple[str, ...] = ()
    block_ids: tuple[str, ...] = ()
    session_ids: tuple[str, ...] = ()
    turn: int = 0
    raw_span: str = ''
    field_span: str = ''
    context: str = ''
    explicitness: float = 0.0
    schema_match: float = 1.0
    query_overlap: float = 0.0
    temporal_score: float = 0.0
    conflict_penalty: float = 0.0
    score: float = 0.0
    head_scores: tuple[tuple[str, float], ...] = ()

    @property
    def scope_key(self) -> tuple[str, ...]:
        if self.block_ids:
            return tuple((f'block:{value}' for value in self.block_ids))
        if self.session_ids:
            return tuple((f'session:{value}' for value in self.session_ids))
        return (self.source,)

def _norm(value: Any) -> str:
    return ' '.join(str(value).casefold().split())

def _tokens(value: Any) -> tuple[str, ...]:
    return tuple((token.casefold() for token in _TOKEN_RE.findall(str(value))))

def _content_tokens(value: Any) -> set[str]:
    return {token for token in _tokens(value) if token not in _STOPWORDS and len(token) > 1}

def _schema_properties(tool_schema: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    schema = tool_schema.get('parameters') or tool_schema.get('input_schema') or {}
    properties = schema.get('properties') or {}
    return {str(name): spec for name, spec in properties.items() if isinstance(spec, dict)}

def _coerce(raw: Any, spec: Mapping[str, Any]) -> Any | None:
    try:
        return validate_parameter_value(raw, dict(spec), 'candidate', partial=True)
    except (TypeError, ValueError):
        return None

def _clean_raw(value: str) -> str:
    cleaned = value.strip().strip('`*').strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and (cleaned[0] in '\'"'):
        cleaned = cleaned[1:-1].strip()
    return cleaned.strip(' ,:;.!?)]}')

def _split_name(name: str) -> list[str]:
    value = re.sub('([a-z0-9])([A-Z])', '\\1 \\2', name)
    value = value.replace('_', ' ').replace('-', ' ').replace('.', ' ')
    return [part.casefold() for part in value.split() if part]

def _aliases(slot: str, spec: Mapping[str, Any]) -> tuple[str, ...]:
    parts = _split_name(slot)
    if not parts:
        return ()
    aliases = {' '.join(parts)}
    if len(parts) == 1 or parts[-1] not in {'id', 'code', 'url', 'uri', 'text', 'name', 'value', 'number'}:
        aliases.add(parts[-1])
    final = parts[-1]
    joined = ''.join(parts)
    aliases.add(joined)
    if final in {'id', 'code', 'url', 'uri', 'text', 'name', 'value', 'number'} and len(parts) > 1:
        base = ' '.join(parts[:-1])
        aliases.update({base, f'{base} {final}', f'{base} identifier' if final == 'id' else base})
    if final in {'id', 'code'} and len(parts) == 1:
        aliases.add('identifier')
    if slot.casefold() in {'q', 'query', 'search'}:
        aliases.update({'query', 'search query', 'search term', 'search terms', 'keywords'})
    if final in {'userid', 'user'} or 'user id' in aliases:
        aliases.update({'user', 'user id', 'account', 'account id'})
    if final in {'language', 'lang', 'locale', 'hl', 'gl'}:
        aliases.update({'language', 'language code', 'locale'})
    if final in {'pagesize', 'page_size', 'nfpr'} or 'page size' in aliases:
        aliases.update({'page size', 'results per page', 'number per page'})
    description = str(spec.get('description', ''))
    description_tokens = _tokens(description)
    semantic_tokens = {'amount', 'category', 'code', 'country', 'date', 'email', 'format', 'id', 'identifier', 'industry', 'language', 'name', 'number', 'page', 'price', 'query', 'region', 'symbol', 'text', 'time', 'topic', 'type', 'url', 'value'}
    base_tokens = {part for part in parts if part not in {'id', 'code', 'url', 'text'}}
    for index, token in enumerate(description_tokens):
        if token in parts or (token in semantic_tokens and token in base_tokens):
            for width in (1, 2, 3):
                start = max(0, index - width + 1)
                phrase = ' '.join(description_tokens[start:index + 1])
                if phrase and base_tokens.intersection(_content_tokens(phrase)) and (len(_content_tokens(phrase)) <= 4):
                    aliases.add(phrase)
    return tuple(sorted({' '.join(alias.split()) for alias in aliases if alias}, key=lambda x: (-len(x), x)))

def _alias_pattern(aliases: Sequence[str]) -> str:
    return '|'.join((re.escape(alias) for alias in sorted(aliases, key=len, reverse=True)))

def _hint_overlap(value: Any, task: TaskInput, slot: str, spec: Mapping[str, Any]) -> float:
    value_tokens = _content_tokens(value)
    hint_text = f"{task.query} {slot} {spec.get('description', '')} {task.tool_schema.get('description', '')}"
    hints = _content_tokens(hint_text)
    if not value_tokens or not hints:
        return 0.0
    return len(value_tokens & hints) / math.sqrt(len(value_tokens) * len(hints))

def _type_head(value: Any, spec: Mapping[str, Any], *, enum_hit: bool=False) -> float:
    score = 1.0
    if enum_hit:
        score += 1.5
    if isinstance(value, str):
        if len(_tokens(value)) > 12:
            score -= 0.8
        if not value.strip():
            score -= 1.0
    return score

def _conflict_penalty(context: str, raw_span: str) -> float:
    window = f'{context} {raw_span}'.casefold()
    cues = ('previous', 'old', 'former', 'another', 'different', 'cancelled', 'canceled', 'no longer')
    return min(1.5, 0.45 * sum((window.count(cue) for cue in cues)))

def _fuse_heads(heads: Mapping[str, float], weights: Mapping[str, float]) -> float:
    values = [float(value) for value in heads.values()]
    pivot = max(values, default=0.0)
    exponentials = [math.exp(value - pivot) for value in values]
    total = sum(exponentials) or 1.0
    normalized = {name: max(weights.get(name, 1.0), 0.0) * math.log(max(exp_value / total, 1e-12)) for (name, _), exp_value in zip(heads.items(), exponentials, strict=True)}
    return sum(normalized.values())

def _candidate(*, task: TaskInput, slot: str, spec: Mapping[str, Any], value: Any, source: str, raw_span: str, field_span: str, context: str, item: EvidenceItem | None, explicitness: float, enum_hit: bool=False, temporal_score: float=0.0) -> FieldCandidate | None:
    coerced = _coerce(value, spec)
    if coerced is None:
        return None
    if isinstance(coerced, str) and (not coerced.strip()):
        return None
    final_token = _split_name(slot)[-1] if _split_name(slot) else ''
    compact_slot = {'id', 'code', 'url', 'uri', 'email', 'number', 'page', 'index', 'count', 'limit'}
    if source == 'natural_field' and final_token in compact_slot and (len(_tokens(coerced)) > 1):
        return None
    overlap = _hint_overlap(coerced, task, slot, spec)
    conflict = _conflict_penalty(context, raw_span)
    heads = {'field': explicitness, 'schema': _type_head(coerced, spec, enum_hit=enum_hit), 'query': 1.5 * overlap, 'temporal': temporal_score, 'conflict': -conflict}
    weights = {'field': 1.8, 'schema': 1.0, 'query': 1.1, 'temporal': 0.35, 'conflict': 1.4}
    score = sum((weights[name] * value for name, value in heads.items()))
    if source == 'reader':
        score += 0.45
    if source == 'default':
        score += 0.2
    return FieldCandidate(slot=slot, value=coerced, source=source, evidence_ids=(item.evidence_id,) if item else (), block_ids=(item.block_id,) if item and item.block_id else (), session_ids=(item.session_id,) if item and item.session_id else (), turn=item.turn if item else 0, raw_span=raw_span, field_span=field_span, context=context, explicitness=explicitness, schema_match=_type_head(coerced, spec, enum_hit=enum_hit), query_overlap=overlap, temporal_score=temporal_score, conflict_penalty=conflict, score=score, head_scores=tuple(heads.items()))

def _json_values(text: str, aliases: set[str]) -> list[tuple[Any, str, str]]:
    values: list[tuple[Any, str, str]] = []
    decoder = json.JSONDecoder()

    def walk(node: Any, path: tuple[str, ...]) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                key_text = str(key)
                key_norm = _norm(key_text)
                next_path = (*path, key_norm)
                if (key_norm in aliases or key_norm.replace('_', ' ') in aliases) and isinstance(value, (str, int, float, bool, dict, list)):
                    values.append((value, key_text, '.'.join(next_path)))
                walk(value, next_path)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, (*path, str(index)))
    for index, char in enumerate(text):
        if char not in '[{':
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except (json.JSONDecodeError, TypeError):
            continue
        walk(parsed, ())
    return values

def _explicit_pairs(text: str, aliases: Sequence[str], slot: str='') -> list[tuple[str, str, str]]:
    alias_set = {_norm(alias) for alias in aliases}
    pairs: list[tuple[str, str, str]] = []
    for match in _KEY_VALUE_RE.finditer(text):
        key = _norm(match.group('key'))
        if key in alias_set:
            value = _clean_raw(match.group('value'))
            if value:
                pairs.append((value, match.group('key'), 'key_value'))
    alias_pattern = _alias_pattern(aliases)
    if not alias_pattern:
        return pairs
    cue_pattern = f"""(?<![A-Za-z0-9_])(?P<field>{alias_pattern})(?![A-Za-z0-9_])\\s*(?:(?:is|was|are|were)\\s+(?:called|named)|is|was|are|were|equals?|set\\s+to|called|named|known\\s+as|stands\\s+for|:)\\s*(?P<value>\\"[^\\"\\n]{{1,180}}\\"|'[^'\\n]{{1,180}}'|[^,;.!?\\n]{{1,180}})"""
    for match in re.finditer(cue_pattern, text, re.IGNORECASE):
        value = _clean_raw(match.group('value'))
        tokens = _tokens(value)
        first = tokens[0] if tokens else ''
        discourse = {'a', 'an', 'answer', 'can', 'could', 'for', 'here', 'already', 'created', 'getting', 'live', 'processed', 'successfully', 'i', 'it', 'please', 'support', 'the', 'that', 'this', 'to', 'we', 'will', 'would', 'you'}
        if value and tokens and (len(tokens) <= (12 if slot.casefold() in {'q', 'query', 'text', 'keyword'} else 8)) and (first not in discourse) and (not _content_tokens(value).issubset(_STOPWORDS)):
            pairs.append((value, match.group('field'), 'natural_field'))
    lower = text.casefold()
    for match in _QUOTED_RE.finditer(text):
        left = max(0, match.start() - 70)
        prefix = lower[left:match.start()]
        relation = re.search(f'(?<![a-z0-9_])(?:{alias_pattern})(?![a-z0-9_]).{{0,45}}(?:is|was|are|were|equals?|set\\s+to|called|named|known\\s+as|stands\\s+for)\\s*$', prefix, re.IGNORECASE)
        suffix = lower[match.end():min(len(text), match.end() + 240)]
        reverse_relation = re.search(f'(?:{alias_pattern})', suffix, re.IGNORECASE) if re.search('(?:called|named)\\s*$', prefix[-45:], re.IGNORECASE) else None
        if relation is None and reverse_relation is None:
            continue
        value = _clean_raw(match.group('value'))
        if value:
            pairs.append((value, 'quoted-field', 'natural_field'))
    return pairs

def _enum_mentions(text: str, spec: Mapping[str, Any]) -> list[str]:
    values = spec.get('enum') or spec.get('choices') or ()
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    lower = text.casefold()
    return [str(value) for value in values if str(value).casefold() in lower]

def _language_mentions(text: str, spec: Mapping[str, Any]) -> list[str]:
    description = str(spec.get('description', '')).casefold()
    if not any((marker in description for marker in ('language', 'locale', 'iso-639'))):
        return []
    lower = text.casefold()
    return [word.title() for word in sorted(_LANGUAGE_WORDS) if re.search(f'\\b{word}\\b', lower)]

def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, str) and isinstance(right, str):
        return _norm(left) == _norm(right)
    return bool(left == right)

def _merge_candidate(old: FieldCandidate, new: FieldCandidate) -> FieldCandidate:
    if new.score > old.score:
        best, other = (new, old)
    else:
        best, other = (old, new)
    return best.__class__(**{**best.__dict__, 'evidence_ids': tuple(dict.fromkeys((*old.evidence_ids, *new.evidence_ids))), 'block_ids': tuple(dict.fromkeys((*old.block_ids, *new.block_ids))), 'session_ids': tuple(dict.fromkeys((*old.session_ids, *new.session_ids))), 'turn': max(old.turn, new.turn), 'context': best.context or other.context})

def _fuse_candidate_set(candidates: Sequence[FieldCandidate]) -> tuple[FieldCandidate, ...]:
    if not candidates:
        return ()
    weights = {'field': 1.8, 'schema': 1.0, 'query': 1.1, 'temporal': 0.35, 'conflict': 1.4}
    names = tuple(weights)
    head_values = {name: [dict(candidate.head_scores).get(name, 0.0) for candidate in candidates] for name in names}
    probabilities: dict[str, tuple[float, ...]] = {}
    for name, values in head_values.items():
        pivot = max(values)
        exp_values = tuple((math.exp(value - pivot) for value in values))
        total = sum(exp_values) or 1.0
        probabilities[name] = tuple((value / total for value in exp_values))
    fused: list[FieldCandidate] = []
    for index, candidate in enumerate(candidates):
        score = sum((weights[name] * math.log(max(probabilities[name][index], 1e-12)) for name in names))
        score += 0.15 * candidate.score
        fused.append(replace(candidate, score=score))
    return tuple(fused)

def build_field_candidates(task: TaskInput, slot: str, spec: Mapping[str, Any], evidence: Sequence[EvidenceItem], reader_value: Any | None, *, max_candidates: int=24) -> tuple[FieldCandidate, ...]:
    if max_candidates < 1:
        raise ValueError('max_candidates must be positive')
    aliases = _aliases(slot, spec)
    alias_set = {_norm(alias) for alias in aliases}
    by_key: dict[tuple[str, tuple[str, ...]], FieldCandidate] = {}
    max_turn = max((item.turn for item in evidence), default=0)

    def add(candidate: FieldCandidate | None) -> None:
        if candidate is None:
            return
        key = (_norm(candidate.value), candidate.scope_key)
        old = by_key.get(key)
        by_key[key] = candidate if old is None else _merge_candidate(old, candidate)
    if reader_value is not None:
        matching = [item for item in evidence if _norm(reader_value) in _norm(item.text)]
        add(_candidate(task=task, slot=slot, spec=spec, value=reader_value, source='reader', raw_span=str(reader_value), field_span='reader output', context=' '.join((item.text for item in matching)), item=matching[-1] if matching else None, explicitness=0.45, temporal_score=matching[-1].turn / max_turn if matching and max_turn else 0.0))
    default = spec.get('default')
    if default is not None:
        add(_candidate(task=task, slot=slot, spec=spec, value=default, source='default', raw_span=str(default), field_span='schema default', context='', item=None, explicitness=0.35))
    for raw, field, _source in _explicit_pairs(task.query, aliases, slot):
        add(_candidate(task=task, slot=slot, spec=spec, value=raw, source='query_field', raw_span=raw, field_span=field, context=task.query, item=None, explicitness=1.35, temporal_score=1.0))
    for raw in _enum_mentions(task.query, spec):
        add(_candidate(task=task, slot=slot, spec=spec, value=raw, source='query_enum', raw_span=raw, field_span='schema enum', context=task.query, item=None, explicitness=1.0, enum_hit=True, temporal_score=1.0))
    for raw in _language_mentions(task.query, spec):
        add(_candidate(task=task, slot=slot, spec=spec, value=raw, source='query_language', raw_span=raw, field_span='language mention', context=task.query, item=None, explicitness=0.9, temporal_score=1.0))
    for item in evidence:
        context = item.text
        for raw_value, field, _path in _json_values(context, alias_set):
            add(_candidate(task=task, slot=slot, spec=spec, value=raw_value, source='json_field', raw_span=str(raw_value), field_span=field, context=context, item=item, explicitness=1.3, temporal_score=item.turn / max_turn if max_turn else 0.0))
        for raw, field, source in _explicit_pairs(context, aliases, slot):
            add(_candidate(task=task, slot=slot, spec=spec, value=raw, source=source, raw_span=raw, field_span=field, context=context, item=item, explicitness=1.05 if source == 'key_value' else 0.92, temporal_score=item.turn / max_turn if max_turn else 0.0))
        for raw in (*_enum_mentions(context, spec), *_language_mentions(context, spec)):
            add(_candidate(task=task, slot=slot, spec=spec, value=raw, source='enum_mention', raw_span=raw, field_span='schema enum/language', context=context, item=item, explicitness=0.7, enum_hit=True, temporal_score=item.turn / max_turn if max_turn else 0.0))
    fused = _fuse_candidate_set(tuple(by_key.values()))
    ranked = sorted(fused, key=lambda candidate: (-candidate.score, -candidate.explicitness, -candidate.temporal_score, _norm(candidate.value), candidate.scope_key))
    return tuple(ranked[:max_candidates])
__all__ = ['FieldCandidate', 'build_field_candidates']
