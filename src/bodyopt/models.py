"""筋肉モデル。params/default.json の sim.muscle_model で切り替える。

遺伝子レイアウト (全モデル共通): [振幅 (E_m)] + [周波数 (E_m)] + [位相 (E_m)]。
辺の張力は「受動バネ + 能動要素」で表し、能動要素の実装だけをここに閉じ込める。

- rest_length: 自然長を周期的に変える (初回実装)。振幅は自然長に対する比。
- active_force: 受動バネと並列に、0 以上の収縮力を周期的に加える。振幅は最大張力 [N]。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp

#: 筋肉の周波数の範囲 [Hz]。全モデル共通で、遺伝子はこの区間にクリップする。
FREQ_LO = 0.1
FREQ_HI = 1.0


class MuscleModel(Protocol):
    """辺ごとの能動要素。JIT 内で呼ぶので、値はすべて jax.Array で扱う。"""

    @property
    def name(self) -> str:
        """params の sim.muscle_model に書く識別子。"""
        ...

    @property
    def gene_hi(self) -> float:
        """振幅の上限 (単位はモデル依存)。"""
        ...

    @property
    def gene_unit(self) -> str:
        """振幅の単位 (記録・表示用)。"""
        ...

    def rest_lengths(
        self,
        rest: jax.Array,  # [E] 基準自然長
        amp: jax.Array,  # [E_m]
        freq: jax.Array,  # [E_m] Hz
        phase: jax.Array,  # [E_m] rad
        t: jax.Array,  # スカラー
        muscle_idx: jax.Array,  # [E_m]
    ) -> jax.Array:  # [E] 現在の自然長
        ...

    def active_tension(
        self,
        amp: jax.Array,
        freq: jax.Array,
        phase: jax.Array,
        t: jax.Array,
        muscle_idx: jax.Array,
        n_edges: int,
    ) -> jax.Array:  # [E] 能動張力 (引く向きが正)
        ...

    def drive_power(
        self,
        active: jax.Array,  # [E] 既に計算済みの能動張力
        tension: jax.Array,  # [E] 辺張力 (受動 + 能動)
        rest: jax.Array,  # [E] 基準自然長
        dLdt: jax.Array,  # [E] 辺長の時間変化率
        amp: jax.Array,
        freq: jax.Array,
        phase: jax.Array,
        t: jax.Array,
        muscle_idx: jax.Array,
    ) -> jax.Array:  # [E] 身体へ注入する仕事率 (正の分のみ)
        ...


@dataclass(frozen=True)
class RestLengthMuscle:
    """自然長を l_rest(t) = L (1 + a sin(2π f t + φ)) で動かすモデル。"""

    name: str = "rest_length"
    gene_hi: float = 0.20
    gene_unit: str = "ratio"

    def rest_lengths(self, rest, amp, freq, phase, t, muscle_idx):
        w = 2.0 * jnp.pi * freq
        return rest.at[muscle_idx].multiply(1.0 + amp * jnp.sin(w * t + phase))

    def active_tension(self, amp, freq, phase, t, muscle_idx, n_edges):
        return jnp.zeros((n_edges,), dtype=amp.dtype)

    def drive_power(self, active, tension, rest, dLdt, amp, freq, phase, t, muscle_idx):
        w = 2.0 * jnp.pi * freq
        rate = rest[muscle_idx] * amp * w * jnp.cos(w * t + phase)
        power = jnp.maximum(0.0, -tension[muscle_idx] * rate)
        return jnp.zeros_like(tension).at[muscle_idx].set(power)


@dataclass(frozen=True)
class ActiveForceMuscle:
    """受動バネと並列に F(t) = A (1 + sin(2π f t + φ)) / 2 を引くモデル。"""

    name: str = "active_force"
    gene_hi: float = 100.0  # 最大能動張力 [N]。体重 (1 ノード 9.81 N) の約 10 倍。
    gene_unit: str = "N"

    def rest_lengths(self, rest, amp, freq, phase, t, muscle_idx):
        return rest  # 自然長は固定

    def active_tension(self, amp, freq, phase, t, muscle_idx, n_edges):
        w = 2.0 * jnp.pi * freq
        act = amp * (1.0 + jnp.sin(w * t + phase)) * 0.5  # 常に 0 以上
        return jnp.zeros((n_edges,), dtype=act.dtype).at[muscle_idx].set(act)

    def drive_power(self, active, tension, rest, dLdt, amp, freq, phase, t, muscle_idx):
        # active は active_tension で計算済み。sin を再計算しない。
        power = jnp.maximum(0.0, -active[muscle_idx] * dLdt[muscle_idx])
        return jnp.zeros_like(tension).at[muscle_idx].set(power)


MODELS: dict[str, MuscleModel] = {
    RestLengthMuscle().name: RestLengthMuscle(),
    ActiveForceMuscle().name: ActiveForceMuscle(),
}


def get_model(name: str) -> MuscleModel:
    try:
        return MODELS[name]
    except KeyError:
        raise ValueError(f"未知の筋肉モデル: {name!r} (選択肢: {sorted(MODELS)})") from None
