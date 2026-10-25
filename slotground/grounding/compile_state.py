from __future__ import annotations
import json
import math
from typing import Any

def validate_parameter_value(value: Any, spec: dict[str, Any], path: str, *, partial: bool=False) -> Any:
    raw_type = spec.get('type', '')
    kinds = raw_type if isinstance(raw_type, list) else [raw_type]
    if value is None:
        if 'null' in kinds:
            return None
        raise ValueError(f'{path}:NULL_NOT_ALLOWED')
    kind = next((k for k in kinds if k != 'null'), '')
    if 'properties' in spec:
        kind = 'object'
    if kind in {'dict', 'object', 'array', 'list'} and isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError as exc:
            raise ValueError(f'{path}:INVALID_JSON_CONTAINER') from exc
    if kind in {'dict', 'object'}:
        if not isinstance(value, dict):
            raise ValueError(f'{path}:EXPECTED_OBJECT')
        props = spec.get('properties', {})
        out = {}
        for name, item in value.items():
            if name in props:
                out[name] = validate_parameter_value(item, props[name], f'{path}.{name}', partial=partial)
            elif spec.get('additionalProperties') is False:
                raise ValueError(f'{path}.{name}:UNKNOWN_PROPERTY')
            else:
                out[name] = item
        if not partial and any((name not in out for name in spec.get('required', []) or [])):
            raise ValueError(f'{path}:NESTED_REQUIRED_MISSING')
        value = out
    elif kind in {'array', 'list'}:
        if not isinstance(value, list):
            raise ValueError(f'{path}:EXPECTED_ARRAY')
        value = [validate_parameter_value(item, spec.get('items', {}), f'{path}.{i}') for i, item in enumerate(value)]
    elif kind in {'int', 'integer', 'number', 'float'}:
        if isinstance(value, bool):
            raise ValueError(f'{path}:BOOLEAN_IS_NOT_NUMBER')
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{path}:EXPECTED_NUMBER') from exc
        if not math.isfinite(numeric) or (kind in {'int', 'integer'} and (not numeric.is_integer())):
            raise ValueError(f'{path}:INVALID_NUMBER')
        value = int(numeric) if kind in {'int', 'integer'} else numeric
        if 'minimum' in spec and value < spec['minimum']:
            raise ValueError(f'{path}:BELOW_MINIMUM')
        if 'maximum' in spec and value > spec['maximum']:
            raise ValueError(f'{path}:ABOVE_MAXIMUM')
    elif kind in {'bool', 'boolean'}:
        if isinstance(value, str) and value.strip().casefold() in {'true', 'false'}:
            value = value.strip().casefold() == 'true'
        if not isinstance(value, bool):
            raise ValueError(f'{path}:EXPECTED_BOOLEAN')
    elif kind in {'str', 'string'}:
        if isinstance(value, dict | list | bool) or value is None:
            raise ValueError(f'{path}:EXPECTED_STRING')
        value = str(value)
    if 'enum' in spec and value not in spec['enum']:
        raise ValueError(f'{path}:INVALID_ENUM')
    return value
