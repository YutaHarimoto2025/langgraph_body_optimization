"""評価関数。first goal は重心の右向き変位のみ、無効候補は常に下位。

駆動仕事 (筋肉がバネに注入した正の仕事) は常に記録し、
energy_budget を設定した場合だけ無効判定に使う (既定は無制限)。
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .creature import CreatureSpec
from .physics import SimParams, rollout, rollout_summary

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


def evaluate_gene(
    gene: jax.Array,
    spec: CreatureSpec,
    sim_p: SimParams,
    fit_p: FitnessParams,
    integrator: str | None = None,
) -> EvalStats:
    """1 個体評価。集計は scan 内で行い、フレーム列を保持しない (vmap で集団評価)。"""
    s = rollout_summary(gene, spec, sim_p, integrator)  # type: ignore[arg-type]
    valid = s.finite & (s.max_penetration < fit_p.max_penetration)
    if fit_p.energy_budget is not None:
        valid = valid & (s.drive_energy <= fit_p.energy_budget)
    score = jnp.where(valid, s.distance, INVALID_SCORE)
    return EvalStats(
        score=score,
        valid=valid,
        distance=s.distance,
        drive_energy=s.drive_energy,
        max_penetration=s.max_penetration,
        contact_rate=s.contact_rate,
    )


def rollout_full(
    gene: jax.Array,
    spec: CreatureSpec,
    sim_p: SimParams,
    integrator: str | None = None,
):
    """記録・再生用の完全ロールアウト (評価と同じ物理コード)。"""
    return rollout(gene, spec, sim_p, integrator)  # type: ignore[arg-type]


#: 一度に vmap する個体数。これを超える集団は分割して評価する。
#: 大きな集団を 1 カーネルで回すと占有率が落ちて時間が超線形に伸びるため。
EVAL_CHUNK = 50_000


def evaluate_population(
    genes: jax.Array,  # [P, 3*E_m]
    spec: CreatureSpec,
    sim_p: SimParams,
    fit_p: FitnessParams,
) -> EvalStats:
    """集団一括評価。GA から呼ぶ jit 対応の入口。

    個体数が EVAL_CHUNK を超える場合は分割して順に評価する (結果は同じ、順序だけ違う)。
    """
    n = genes.shape[0]
    eval_chunk = lambda c: jax.vmap(lambda g: evaluate_gene(g, spec, sim_p, fit_p))(c)
    if n <= EVAL_CHUNK:
        return eval_chunk(genes)

    n_chunk = -(-n // EVAL_CHUNK)
    pad = n_chunk * EVAL_CHUNK - n
    if pad:
        genes = jnp.concatenate([genes, jnp.repeat(genes[-1:], pad, axis=0)], axis=0)
    chunks = genes.reshape(n_chunk, EVAL_CHUNK, genes.shape[1])
    out = jax.lax.map(eval_chunk, chunks)
    return jax.tree.map(lambda x: x.reshape(n_chunk * EVAL_CHUNK, *x.shape[2:])[:n], out)
