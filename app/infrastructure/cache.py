"""Small bounded in-process cache for deterministic expensive stage results."""

from collections import OrderedDict
from copy import deepcopy
from hashlib import sha256
import json
from threading import Lock
from typing import Any
from weakref import WeakKeyDictionary

_MAX_ENTRIES = 128
_cache: OrderedDict[str, Any] = OrderedDict()
_lock = Lock()
_object_ids: WeakKeyDictionary = WeakKeyDictionary()
_next_object_id = 0


def content_key(namespace: str, *values: Any) -> str:
    rendered = json.dumps(
        values,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )
    return f"{namespace}:{sha256(rendered.encode()).hexdigest()}"


def get_cached(key: str) -> Any | None:
    with _lock:
        if key not in _cache:
            return None
        value = _cache.pop(key)
        _cache[key] = value
        return deepcopy(value)


def set_cached(key: str, value: Any) -> None:
    with _lock:
        _cache.pop(key, None)
        _cache[key] = deepcopy(value)
        while len(_cache) > _MAX_ENTRIES:
            _cache.popitem(last=False)


def clear_stage_cache() -> None:
    with _lock:
        _cache.clear()


def object_identity(value: Any) -> int:
    """Return a non-recycled process identity for a live dependency object."""
    global _next_object_id
    with _lock:
        try:
            identity = _object_ids.get(value)
            if identity is None:
                _next_object_id += 1
                identity = _next_object_id
                _object_ids[value] = identity
            return identity
        except TypeError:
            return id(value)


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return str(value)
