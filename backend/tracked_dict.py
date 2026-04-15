"""
Dict and list wrappers that track key access for reporting unconsumed data.

Wrap a parsed JSON/YAML/TOML structure in TrackedDict, read what you need
through normal dict access, then call unaccessed() to get dotted paths for
every key that was never touched. This catches both parser omissions and
unknown user-supplied fields without maintaining a separate list of
expected keys.

Usage::

    import json
    from tracked_dict import TrackedDict

    with open("config.json") as f:
        data = TrackedDict(json.load(f))

    name = data["project"]["name"]
    version = data["project"]["version"]

    for path in data.unaccessed():
        print(f"  unhandled: {path}")

Author: Eremey Valetov
"""

from __future__ import annotations

__all__ = ["TrackedDict", "TrackedList"]
__version__ = "0.1.0"


class TrackedDict:
    """Dict wrapper that records which keys are accessed."""

    __slots__ = ("_data", "_path", "_accessed", "_children")

    def __init__(self, data: dict, _path: str = "") -> None:
        self._data = data
        self._path = _path
        self._accessed: set[str] = set()
        self._children: dict[str, TrackedDict | TrackedList] = {}

    # --- read access ---

    def __getitem__(self, key: str):
        self._accessed.add(key)
        return self._wrap(key, self._data[key])

    def get(self, key: str, default=None):
        self._accessed.add(key)
        if key in self._data:
            return self._wrap(key, self._data[key])
        return default

    # --- container protocol ---

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __bool__(self) -> bool:
        return bool(self._data)

    def __iter__(self):
        return iter(self._data)

    def keys(self):
        return self._data.keys()

    def values(self):
        self._accessed.update(self._data.keys())
        for key in self._data:
            yield self._wrap(key, self._data[key])

    def items(self):
        self._accessed.update(self._data.keys())
        for key in self._data:
            yield key, self._wrap(key, self._data[key])

    # --- tracking helpers ---

    @property
    def raw(self) -> dict:
        """Return the underlying plain dict."""
        return self._data

    def mark_accessed(self, *keys: str) -> None:
        """Explicitly mark keys as accessed (e.g. for bulk-forwarded sections)."""
        self._accessed.update(keys)

    def mark_all_accessed(self) -> None:
        """Mark every key in this dict (non-recursive) as accessed."""
        self._accessed.update(self._data.keys())

    def unaccessed(self) -> list[str]:
        """Return sorted list of dotted paths for all unaccessed keys, recursively."""
        result: list[str] = []
        for key in self._data:
            path = self._child_path(key)
            if key not in self._accessed:
                result.append(path)
            elif key in self._children:
                result.extend(self._children[key].unaccessed())
        return sorted(result)

    def accessed_keys(self) -> set[str]:
        """Return the set of keys that have been accessed at this level."""
        return set(self._accessed)

    # --- internals ---

    def _child_path(self, key: str) -> str:
        return f"{self._path}.{key}" if self._path else str(key)

    def _wrap(self, key: str, value):
        if isinstance(value, dict):
            if key not in self._children:
                self._children[key] = TrackedDict(value, self._child_path(key))
            return self._children[key]
        if isinstance(value, list):
            if key not in self._children:
                self._children[key] = TrackedList(value, self._child_path(key))
            return self._children[key]
        return value

    def __repr__(self) -> str:
        return (
            f"TrackedDict({len(self._accessed)}/{len(self._data)} keys accessed, "
            f"path={self._path!r})"
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TrackedDict):
            return self._data == other._data
        if isinstance(other, dict):
            return self._data == other
        return NotImplemented


class TrackedList:
    """List wrapper that wraps dict/list items for access tracking."""

    __slots__ = ("_data", "_path", "_children")

    def __init__(self, data: list, _path: str = "") -> None:
        self._data = data
        self._path = _path
        self._children: dict[int, TrackedDict | TrackedList] = {}

    def __getitem__(self, index: int):
        return self._wrap(index, self._data[index])

    def __len__(self) -> int:
        return len(self._data)

    def __bool__(self) -> bool:
        return bool(self._data)

    def __iter__(self):
        for i in range(len(self._data)):
            yield self._wrap(i, self._data[i])

    @property
    def raw(self) -> list:
        """Return the underlying plain list."""
        return self._data

    def unaccessed(self) -> list[str]:
        """Collect unaccessed paths from all child TrackedDicts/TrackedLists."""
        result: list[str] = []
        for child in self._children.values():
            result.extend(child.unaccessed())
        return sorted(result)

    def _wrap(self, index: int, value):
        if isinstance(value, dict):
            if index not in self._children:
                self._children[index] = TrackedDict(
                    value, f"{self._path}[{index}]"
                )
            return self._children[index]
        if isinstance(value, list):
            if index not in self._children:
                self._children[index] = TrackedList(
                    value, f"{self._path}[{index}]"
                )
            return self._children[index]
        return value

    def __repr__(self) -> str:
        return f"TrackedList({len(self._data)} items, path={self._path!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TrackedList):
            return self._data == other._data
        if isinstance(other, list):
            return self._data == other
        return NotImplemented
