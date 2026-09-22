"""TASK2 配置：`task2.yaml` 载入、默认值合并与校验。

配置只描述"这次跑什么"，不藏逻辑：所有默认值都能在 YAML 里看到并覆盖，
未知键直接报错（拼错的键静默忽略比报错危险得多）。
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

TASK_ROOT = Path(__file__).resolve().parents[1]

# task2.yaml 是唯一默认配置来源；不再维护一份 Python 字典副本。
DEFAULTS: dict[str, Any] = yaml.safe_load((TASK_ROOT / "task2.yaml").read_text(encoding="utf-8"))


def _deep_merge(base: dict, override: dict, path: str = "") -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key not in merged:
            raise KeyError(f"配置里有未知键：{path}{key}（可用键：{', '.join(sorted(merged))}）")
        if isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value, f"{path}{key}.")
        else:
            merged[key] = value
    return merged


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """读 YAML 并与默认值合并。`path=None` 时用任务目录下的 `task2.yaml`。"""
    if path is None or Path(path).resolve() == (TASK_ROOT / "task2.yaml").resolve():
        return copy.deepcopy(DEFAULTS)
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return _deep_merge(DEFAULTS, raw)


def parse_courses(value) -> tuple[int, ...]:
    if isinstance(value, str):
        parts = [part for part in value.replace(" ", "").split(",") if part]
        return tuple(int(part) for part in parts)
    return tuple(int(item) for item in value)


def default_report_path(arm: str, policy: str, root: Path | None = None) -> Path:
    base = TASK_ROOT if root is None else root
    return base / "reports" / f"task2_{arm}_{policy}.json"
