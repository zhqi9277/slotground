from __future__ import annotations
import re
from collections.abc import Mapping, Sequence
from typing import Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from slotground.grounding.compile_state import validate_parameter_value
from slotground.grounding.evidence_state import ObservationOutput, StateOption, augment_literal_fields, build_options, ground_observations, value_key
from slotground.types import EvidenceItem, TaskInput
FEATURES = ('reader_agreement', 'reader_original', 'default_added', 'omit', 'required', 'current', 'has_evidence', 'literal_source', 'entity', 'n_choices', 'object_anchor', 'update_relation', 'default_value', 'default_added_omit', 'reader_original_omit', 'required_omit', 'wildcard_default_omit', 'operational_default_omit', 'empty_default_omit', 'date_default_omit')

def option_features(task, option, choices, original, baseline):
    schema = task.tool_schema.get('parameters') or task.tool_schema.get('input_schema') or {}
    spec = schema.get('properties', {}).get(option.slot, {})
    agreement = option.omit and option.slot not in baseline or (not option.omit and option.slot in baseline and (value_key(option.value) == value_key(baseline[option.slot])))
    default = str(spec.get('default', '')).casefold()
    operational = bool(re.search('page|cursor|offset|pagination|locale|language|sort|order|format|storage|folder|number of (?:articles|results)|maximum number of (?:tweets|results)', f"{option.slot} {spec.get('description', '')}", re.I))
    return np.asarray([agreement, option.slot in original, option.slot not in original and option.slot in baseline, option.omit, option.slot in schema.get('required', []), '__current__' in option.evidence_ids, bool(option.evidence_ids), option.source in {'json_field', 'key_value', 'natural_field'}, bool(option.entity), np.log1p(len(choices)), option.object_score, option.update_score, 'default' in spec and value_key(option.value) == value_key(spec.get('default')), option.slot not in original and option.slot in baseline and option.omit, option.slot in original and option.omit, option.slot in schema.get('required', []) and option.omit, option.omit and default in {'dontcare', 'dont care', 'none', 'null', 'any'}, option.omit and 'default' in spec and operational, option.omit and spec.get('default') == '', option.omit and bool(re.fullmatch('\\d{4}-\\d{2}-\\d{2}', default))], dtype=float)

def add_omission_options(task, options, baseline):
    result = {}
    for slot, choices in options.items():
        choices = tuple(choices)
        if not any((o.omit for o in choices)):
            choices = (*choices, StateOption(slot, None, True, '', '', '', (), 0.8, 0, 0, 'unknown'))
        if slot in baseline and (not any((not o.omit and value_key(o.value) == value_key(baseline[slot]) for o in choices))):
            choices = (*choices, StateOption(slot, baseline[slot], False, '', '', '', (), 1.0, 0, 0, 'reader_fallback'))
        result[slot] = choices
    return result

def fit_model(xs, ys, weights, mask=(8, 10, 11)):
    x = np.asarray(xs).copy()
    x[:, mask] = 0
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000).fit(scaler.transform(x), ys, sample_weight=weights)
    return {'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(), 'coef': model.coef_[0].tolist(), 'intercept': float(model.intercept_[0]), 'mask': list(mask)}

def probabilities(model, x):
    x = np.asarray(x).copy()
    x[:, model['mask']] = 0
    logits = (x - model['mean']) / model['scale'] @ model['coef'] + model['intercept']
    return 1 / (1 + np.exp(-np.clip(logits, -40, 40)))

def fill_defaults(task: TaskInput, arguments: dict[str, Any]) -> dict[str, Any]:
    schema = task.tool_schema.get('parameters') or task.tool_schema.get('input_schema') or {}
    result = dict(arguments)
    for slot, spec in (schema.get('properties') or {}).items():
        if slot in result or spec.get('default') is None:
            continue
        try:
            result[slot] = validate_parameter_value(spec['default'], spec, slot)
        except ValueError:
            continue
    return result
DISABLED_FEATURES = (8, 10, 11)
ACTIVE_FEATURES = tuple((name for index, name in enumerate(FEATURES) if index not in DISABLED_FEATURES))

def select_state(options, scores, reference, margin=0.3):
    arguments = dict(reference)
    for slot, choices in options.items():
        values = tuple((float(p) for p in scores[slot]))
        if len(choices) != len(values):
            raise ValueError('Candidate and score counts differ')
        if not choices:
            continue
        chosen_index = max(range(len(choices)), key=lambda i: min(max(values[i], 1e-06), 1 - 1e-06))
        chosen = choices[chosen_index]
        new = max((p for option, p in zip(choices, values, strict=True) if option.omit == chosen.omit and value_key(option.value) == value_key(chosen.value)), default=0.0)
        old = max((p for option, p in zip(choices, values, strict=True) if option.omit and slot not in reference or (not option.omit and slot in reference and (value_key(option.value) == value_key(reference[slot])))), default=0.0)
        if new - old < margin:
            continue
        if chosen.omit:
            arguments.pop(slot, None)
        else:
            arguments[slot] = chosen.value
    return arguments

def construct_candidates(task: TaskInput, evidence: Sequence[EvidenceItem], reader: Mapping[str, Any], extracted: ObservationOutput):
    pool = augment_literal_fields(task, evidence, ground_observations(task, evidence, extracted))
    reference = fill_defaults(task, dict(reader))
    options = add_omission_options(task, build_options(task, pool, reference), reference)
    return (pool, options, reference)

def score_candidates(task, options, reader, reference, model):
    return {slot: probabilities(model, [option_features(task, option, choices, reader, reference) for option in choices]) for slot, choices in options.items()}

def infer_arguments(task: TaskInput, evidence: Sequence[EvidenceItem], reader: Mapping[str, Any], extracted: ObservationOutput, model, margin=0.3):
    pool, options, reference = construct_candidates(task, evidence, reader, extracted)
    scores = score_candidates(task, options, reader, reference, model)
    return select_state(options, scores, reference, margin)
