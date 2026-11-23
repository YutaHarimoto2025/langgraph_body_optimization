"""評価関数。first goal は重心の右向き変位のみ、無効候補は常に下位。

駆動仕事 (筋肉がバネに注入した正の仕事) は常に記録し、
energy_budget を設定した場合だけ無効判定に使う (既定は無制限)。
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .creature import CreatureSpec
from .physics import RolloutResult, SimParams, rollout

#: 無効候補のスコア。有効候補の変位 (±数 m オーダー) より常に下。
INVALID_SCORE = -1.0e6


@dataclass(frozen=True)
class FitnessParams:
    max_penetration: float = 0.10  # これを超えるめり込みは無効
    energy_budget: float | None = None  # 駆動仕事の上限 [J]。None なら無制限


@dataclass(frozen=True)
class EvalStats:
    """分析用の付随統計 (GA の選択には score しか使わない)。vmap 出力に使うため pytree。"""

    score: jax.Array  # スカラー (無効なら INVALID_SCORE)
    valid: jax.Array  # bool
    distance: jax.Array  # 有効判定前の重心変位 [m]
    drive_energy: jax.Array  # 駆動仕事 [J]
    max_penetration: jax.Array  # 最大めり込み [m]
    contact_rate: jax.Array  # 接触していたフレーム×質点の割合 [0,1]


jax.tree_util.register_dataclass(
    EvalStats,
    data_fields=["score", "valid", "distance", "drive_energy", "max_penetration", "contact_rate"],
    meta_fields=[],
)


def com_trajectory(pos: jax.Array, spec: CreatureSpec) -> jax.Array:
    """質量加重重心 [T+1, 2]。"""
    m = spec.masses
    return jnp.einsum("tni,n->ti", pos, m) / jnp.sum(m)


def _summary_from_rollout(
    r: RolloutResult, spec: CreatureSpec, fp: FitnessParams
) -> EvalStats:
    com = com_trajectory(r.pos, spec)
    distance = com[-1, 0] - com[0, 0]
    finite = jnp.all(jnp.isfinite(r.pos)) & jnp.all(jnp.isfinite(r.vel))
    max_pen = jnp.maximum(0.0, -jnp.min(r.pos[..., 1]))
    # 瞬間パワー [T+1, E] → 仕事: 最終フレームを除き dt で積算
    energy = jnp.sum(jnp.maximum(0.0, r.drive_power[:-1])) * r.times[1]
    contact = (r.contact_force > 0).mean()

    valid = finite & (max_pen < fp.max_penetration)
    if fp.energy_budget is not None:
        valid = valid & (energy <= fp.energy_budget)
    score = jnp.where(valid, distance, INVALID_SCORE)
    return EvalStats(
        score=score,
        valid=valid,
        distance=distance,
        drive_energy=energy,
        max_penetration=max_pen,
        contact_rate=contact,
    )


def evaluate_gene(
    gene: jax.Array,
    spec: CreatureSpec,
    sim_p: SimParams,
    fit_p: FitnessParams,
    integrator: str | None = None,
) -> EvalStats:
    """1 個体評価。vmap で集団評価に使う (jit 対応)。"""
    r = rollout(gene, spec, sim_p, integrator)  # type: ignore[arg-type]
    return _summary_from_rollout(r, spec, fit_p)


def rollout_full(
    gene: jax.Array,
    spec: CreatureSpec,
    sim_p: SimParams,
    integrator: str | None = None,
) -> RolloutResult:
    """記録・再生用の完全ロールアウト (評価と同じ物理コード)。"""
    return rollout(gene, spec, sim_p, integrator)  # type: ignore[arg-type]


def evaluate_population(
    genes: jax.Array,  # [P, 2*E_m]
    spec: CreatureSpec,
    sim_p: SimParams,
    fit_p: FitnessParams,
) -> EvalStats:
    """集団一括評価。GA から呼ぶ jit 対応の入口。"""
    return jax.vmap(lambda g: evaluate_gene(g, spec, sim_p, fit_p))(genes)
