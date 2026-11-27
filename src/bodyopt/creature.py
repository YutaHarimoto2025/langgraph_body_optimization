"""身体定義。2D の質点集合と伸縮エッジ (受動バネ + 能動筋肉)。"""
from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

@dataclass(frozen=True)
class CreatureSpec:
    """1 個体分の身体。質点・エッジの接続は固定で、GA は制御遺伝子のみ探索する。"""

    base_pos: jax.Array  # [N, 2] 基準位置 (rest pose)
    masses: jax.Array  # [N] 質量 [kg]
    edges: jax.Array  # [E, 2] int 質点 index
    stiffness: jax.Array  # [E] バネ剛性 [N/m]
    rest_lengths: jax.Array  # [E] 基準自然長 (初期辺長に一致)
    muscle_mask: jax.Array  # [E] bool 能動筋肉かどうか
    muscle_idx: jax.Array  # [E_m] 能動辺の index (構築時に確定、JIT 内で使う)
    name: str = field(default="ladder")

    @property
    def n_nodes(self) -> int:
        return int(self.base_pos.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edges.shape[0])

    @property
    def n_muscles(self) -> int:
        return int(self.muscle_idx.shape[0])


def triangulated_ladder(
    n_cols: int = 3,
    col_spacing: float = 0.5,
    row_spacing: float = 0.4,
    bottom_clearance: float = 0.05,
    mass: float = 1.0,
    stiffness: float = 500.0,
    muscle_all: bool = True,
) -> CreatureSpec:
    """上下 2 列 × n_cols の三角補強ラダー。

    下列を床 (y=0) から bottom_clearance 上に置く。横辺・縦辺・対角辺
    (交互方向) を張り、初回は全辺を能動筋肉とする。
    """
    xs = jnp.arange(n_cols, dtype=jnp.float32) * col_spacing
    bottom = jnp.stack([xs, jnp.full(n_cols, bottom_clearance)], axis=1)
    top = jnp.stack([xs, jnp.full(n_cols, bottom_clearance + row_spacing)], axis=1)
    # 下列 0..n_cols-1、上列 n_cols..2n_cols-1
    base_pos = jnp.concatenate([bottom, top], axis=0)

    pairs: list[tuple[int, int]] = []
    for i in range(n_cols - 1):
        pairs.append((i, i + 1))  # 下列横
        pairs.append((n_cols + i, n_cols + i + 1))  # 上列横
        pairs.append((i, n_cols + i))  # 縦
        pairs.append((i, n_cols + i + 1))  # 対角 (昇り)
        pairs.append((i + 1, n_cols + i))  # 対角 (降り)
    pairs.append((n_cols - 1, n_cols + n_cols - 1))  # 最後の縦
    edges = jnp.array(pairs, dtype=jnp.int32)

    edge_len = jnp.linalg.norm(base_pos[edges[:, 0]] - base_pos[edges[:, 1]], axis=1)
    e = int(edges.shape[0])
    mask = jnp.ones(e, dtype=jnp.bool_) if muscle_all else jnp.zeros(e, dtype=jnp.bool_)
    return CreatureSpec(
        base_pos=base_pos,
        masses=jnp.full(base_pos.shape[0], mass, dtype=jnp.float32),
        edges=edges,
        stiffness=jnp.full(e, stiffness, dtype=jnp.float32),
        rest_lengths=edge_len.astype(jnp.float32),
        muscle_mask=mask,
        muscle_idx=jnp.where(mask)[0],  # JIT 内で nonzero しないよう構築時に確定
    )
