"""JAX 物理シミュレータ。半陰的オイラー法 (第一候補) と RK4 (比較用)。

設計方針 (docs/agents/sim/physics.md):
- 1 個体分の物理を素直に書き、集団軸は vmap、時間軸は lax.scan で処理する。
- 接触は床までのめり込みに対するペナルティ反力、摩擦は tanh で滑らかにした
  クーロン摩擦 (静止摩擦は再現しない)。
- フレーム k は時刻 t = k*dt 開始時点の状態で一貫して記録する。
- 世代ループ内で CPU 待ちを作らない。同期は run の最後だけ。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp

from .creature import CreatureSpec

Integrator = Literal["semi_implicit_euler", "rk4"]

#: 長さ計算のゼロ除算防止。実際の無効判定は fitness 側で行う。
_LEN_EPS = 1e-9


@dataclass(frozen=True)
class SimParams:
    """ケース非依存の物理パラメータ (params/default.json と対応)。"""

    gravity: float = 9.81
    dt: float = 0.005
    episode_seconds: float = 5.0
    k_contact: float = 2000.0
    c_contact: float = 89.44  # 2√(k_c·m) 臨界減衰: バウンドの非物理的な継続を抑える
    mu_friction: float = 0.7
    friction_vel_eps: float = 0.05
    omega: float = 2.0 * jnp.pi  # 全筋肉共通の駆動角周波数 [rad/s] (1 Hz)
    integrator: Integrator = "semi_implicit_euler"

    @property
    def n_steps(self) -> int:
        return int(round(self.episode_seconds / self.dt))


@dataclass(frozen=True)
class RolloutResult:
    """1 個体分のロールアウト結果。フレーム k = 時刻 k*dt の状態。"""

    pos: jax.Array  # [T+1, N, 2]
    vel: jax.Array  # [T+1, N, 2]
    times: jax.Array  # [T+1]
    spring_force: jax.Array  # [T+1, E] 辺張力 (張りが正)
    contact_force: jax.Array  # [T+1, N] 法線反力
    drive_power: jax.Array  # [T+1, E] 筋肉の注入仕事率 (正の分のみ)
    rest_lengths: jax.Array  # [T+1, E] 時変自然長


def muscle_rest_lengths(
    base_rest: jax.Array,  # [E]
    amplitude: jax.Array,  # [E_m] 遺伝子
    phase: jax.Array,  # [E_m] 遺伝子
    omega: float,
    muscle_idx: jax.Array,  # [E_m] 能動辺の index
    t: jax.Array,  # スカラーまたは [T]
) -> jax.Array:
    """時変自然長 l_rest(t) = L_e (1 + a_e sin(ωt + φ_e))。非駆動辺は基準のまま。"""
    t_col = jnp.expand_dims(t, -1)  # t スカラー: (1,)、ベクトル: [T, 1]
    act = amplitude * jnp.sin(omega * t_col + phase)  # [E_m] or [T, E_m]
    if act.ndim == 1:
        return base_rest.at[muscle_idx].multiply(1.0 + act)
    base = jnp.broadcast_to(base_rest, (act.shape[0], base_rest.shape[0]))
    upd = base[:, muscle_idx] * (1.0 + act)
    return base.at[:, muscle_idx].set(upd)


def _spring_forces(
    pos: jax.Array,  # [N, 2]
    edges: jax.Array,  # [E, 2]
    stiffness: jax.Array,  # [E]
    rest_t: jax.Array,  # [E] 現在時刻の自然長
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """辺張力と各質点へのバネ力。戻り値: node_forces [N,2], tension [E], length [E]。"""
    d = pos[edges[:, 0]] - pos[edges[:, 1]]  # [E, 2]
    length = jnp.linalg.norm(d, axis=1)  # [E]
    u = d / jnp.maximum(length, _LEN_EPS)[:, None]
    tension = stiffness * (length - rest_t)  # 張りが正
    f_on_i = -tension[:, None] * u  # 辺の i 側質点にかかる力
    forces = jnp.zeros_like(pos)
    forces = forces.at[edges[:, 0]].add(f_on_i)
    forces = forces.at[edges[:, 1]].add(-f_on_i)
    return forces, tension, length


def _contact_forces(
    pos: jax.Array,
    vel: jax.Array,
    p: SimParams,
) -> tuple[jax.Array, jax.Array]:
    """床 (y=0) とのペナルティ反力と滑らかなクーロン摩擦。戻り値: total [N,2], normal [N]。"""
    penetration = jnp.maximum(0.0, -pos[:, 1])  # めり込み量 [N]
    normal = p.k_contact * penetration - p.c_contact * jnp.minimum(0.0, vel[:, 1])
    normal = jnp.maximum(0.0, normal)  # 引っ張らない
    friction = -p.mu_friction * normal * jnp.tanh(vel[:, 0] / p.friction_vel_eps)
    total = jnp.stack([friction, normal], axis=1)
    return total, normal


def _drive_power(
    tension: jax.Array,  # [E] 現在張力
    base_rest: jax.Array,  # [E]
    muscle_idx: jax.Array,  # [E_m]
    amplitude: jax.Array,  # [E_m]
    phase: jax.Array,  # [E_m]
    omega: float,
    t: jax.Array,
) -> jax.Array:
    """筋肉がバネへ注入する仕事率 P = -∂U/∂l_rest · dl_rest/dt (正の分のみ)。

    張った筋肉がさらに縮むときに正となり、弛んだ筋肉の自然長変化は
    記録上ゼロとする (回生を数えない)。
    """
    dl = omega * amplitude * jnp.cos(omega * t + phase)  # [E_m] d(act)/dt
    f = tension[muscle_idx]
    power = jnp.maximum(0.0, -(f * base_rest[muscle_idx]) * dl)  # [E_m]
    out = jnp.zeros(base_rest.shape[0], dtype=power.dtype)
    return out.at[muscle_idx].set(power)


def rollout(
    gene: jax.Array,  # [2*E_m] amplitude, phase の連結
    spec: CreatureSpec,
    p: SimParams,
    integrator: Integrator | None = None,
) -> RolloutResult:
    """1 個体分のエピソード。jit 対応 (dataclass は trace 時に静的に扱われる)。"""
    integ = p.integrator if integrator is None else integrator
    muscle_idx = spec.muscle_idx
    n_m = int(muscle_idx.shape[0])
    amp, phase = gene[:n_m], gene[n_m:]
    edges, stiffness, base_rest = spec.edges, spec.stiffness, spec.rest_lengths
    masses_inv = 1.0 / spec.masses
    n_steps = p.n_steps
    dt = p.dt

    def observe(pos: jax.Array, vel: jax.Array, t: jax.Array):
        rest_t = muscle_rest_lengths(base_rest, amp, phase, p.omega, muscle_idx, t)
        spring_f, tension, _ = _spring_forces(pos, edges, stiffness, rest_t)
        contact_f, normal = _contact_forces(pos, vel, p)
        drive = _drive_power(tension, base_rest, muscle_idx, amp, phase, p.omega, t)
        acc = (spring_f + contact_f) * masses_inv[:, None]
        obs = dict(
            pos=pos,
            vel=vel,
            spring_force=tension,
            contact_force=normal,
            drive_power=drive,
            rest_lengths=rest_t,
        )
        return obs, acc

    def rk4_update(pos: jax.Array, vel: jax.Array, t: jax.Array) -> tuple[jax.Array, jax.Array]:
        def f(tt: jax.Array, pv: jax.Array, vv: jax.Array):
            a, _, _ = _forces_only(tt, pv, vv)
            return vv, a

        k1p, k1v = f(t, pos, vel)
        k2p, k2v = f(t + dt / 2, pos + dt / 2 * k1p, vel + dt / 2 * k1v)
        k3p, k3v = f(t + dt / 2, pos + dt / 2 * k2p, vel + dt / 2 * k2v)
        k4p, k4v = f(t + dt, pos + dt * k3p, vel + dt * k3v)
        return (
            pos + dt / 6.0 * (k1p + 2 * k2p + 2 * k3p + k4p),
            vel + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v),
        )

    def _forces_only(t: jax.Array, pos: jax.Array, vel: jax.Array):
        rest_t = muscle_rest_lengths(base_rest, amp, phase, p.omega, muscle_idx, t)
        spring_f, tension, _ = _spring_forces(pos, edges, stiffness, rest_t)
        contact_f, _ = _contact_forces(pos, vel, p)
        return (spring_f + contact_f) * masses_inv[:, None], tension, rest_t

    def scan_fn(carry, _):
        pos, vel, t = carry
        obs, acc = observe(pos, vel, t)
        if integ == "rk4":
            pos_n, vel_n = rk4_update(pos, vel, t)
        else:
            vel_n = vel + dt * acc  # 半陰的オイラー
            pos_n = pos + dt * vel_n
        return (pos_n, vel_n, t + dt), obs

    t0 = jnp.float32(0.0)
    init = (spec.base_pos, jnp.zeros((spec.n_nodes, 2), dtype=jnp.float32), t0)
    (pos_T, vel_T, t_T), outs = jax.lax.scan(scan_fn, init, None, length=n_steps)

    final_obs, _ = observe(pos_T, vel_T, t_T)
    outs = jax.tree.map(lambda s, f: jnp.concatenate([s, f[None]], axis=0), outs, final_obs)
    times = jnp.arange(n_steps + 1, dtype=jnp.float32) * dt
    return RolloutResult(
        pos=outs["pos"],
        vel=outs["vel"],
        times=times,
        spring_force=outs["spring_force"],
        contact_force=outs["contact_force"],
        drive_power=outs["drive_power"],
        rest_lengths=outs["rest_lengths"],
    )
