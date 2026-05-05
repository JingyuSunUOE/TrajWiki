"""JSON helpers with consistent formatting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import orjson
except Exception:  # noqa: BLE001
    orjson = None
    import json


def dumps_json(data: Any, *, indent: bool = False) -> str:
    if orjson is not None:
        option = orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS if indent else 0
        return orjson.dumps(data, option=option).decode("utf-8")
    return json.dumps(
        data,
        indent=2 if indent else None,
        sort_keys=indent,
        ensure_ascii=False,
        default=str,
    )


def write_json(path: Path, data: Any, *, indent: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_json(data, indent=indent), encoding="utf-8")


def append_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(dumps_json(row, indent=False))
            handle.write("\n")
