"""params/*.json から実行パラメータを組み立てる。

- "sim" / "fitness" / "ga": 各 dataclass (SimParams・FitnessParams・GAParams) のフィールド。
- "creature": triangulated_ladder のキーワード引数。
- 未知のキーは typo とみなしてエラーにする。
- 筋肉モデル (sim.muscle_model) は models.MODELS のキーで切り替える。
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from .creature import CreatureSpec, triangulated_ladder
from .fitness import FitnessParams
from .ga import GAParams
from .models import MODELS
from .physics import SimParams

DEFAULT_CONFIG = "params/default.json"

T = TypeVar("T")


@dataclass(frozen=True)
class Config:
    sim: SimParams
    fitness: FitnessParams
    ga: GAParams
    creature: dict[str, Any]

    def build_creature(self, **overrides: Any) -> CreatureSpec:
        return triangulated_ladder(**{**self.creature, **overrides})


def _build(cls: type[T], values: dict[str, Any]) -> T:
    known = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"{cls.__name__} に未知のキー: {sorted(unknown)}")
    return cls(**values)  # type: ignore[return-value]


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    sim = _build(SimParams, raw.get("sim", {}))
    if sim.muscle_model not in MODELS:
        raise ValueError(
            f"sim.muscle_model={sim.muscle_model!r} は未定義 (選択肢: {sorted(MODELS)})"
        )
    return Config(
        sim=sim,
        fitness=_build(FitnessParams, raw.get("fitness", {})),
        ga=_build(GAParams, raw.get("ga", {})),
        creature=dict(raw.get("creature", {})),
    )
