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

from jax.typing import DTypeLike

from .creature import CreatureSpec
from .models import get_model

Integrator = Literal["semi_implicit_euler", "rk4"]

#: 長さ計算のゼロ除算防止。実際の無効判定は fitness 側で行う。
_LEN_EPS = 1e-9


@dataclass(frozen=True)
class SimParams:
    """ケース非依存の物理パラメータ (params/default.json と対応)。"""

    gravity: float = 9.81
    dt: float = 0.005
    episode_seconds: float = 5.0
    k_contact: float = 20000.0
    c_contact: float = 282.84  # 2√(k_c·m) 臨界減衰: バウンドの非物理的な継続を抑える
    mu_friction: float = 0.7
    friction_vel_eps: float = 0.05
    muscle_model: str = "active_force"  # models.MODELS のキー
    edge_damping_ratio: float = 0.0  # 辺の軸方向粘性減衰の減衰比 ζ (0 で減衰なし)
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
    rest_lengths: jax.Array  # [T+1, E] 基準自然長 (固定)


def _edge_damping_coeff(stiffness: jax.Array, masses: jax.Array, edges: jax.Array, zeta: float) -> jax.Array:
    """辺ごとの粘性減衰係数 c_e = 2 ζ √(k_e m_eff)、m_eff = m_i m_j / (m_i + m_j)。"""
    m_i = masses[edges[:, 0]]
    m_j = masses[edges[:, 1]]
    m_eff = m_i * m_j / (m_i + m_j)
    return 2.0 * zeta * jnp.sqrt(stiffness * m_eff)


def _incidence(edges: jax.Array, n_nodes: int, dtype: DTypeLike) -> jax.Array:
    """[N, E] の入射行列。node が辺の終点なら +1、始点なら -1。構築時に 1 回だけ作る。"""
    ids = jnp.arange(n_nodes)[:, None]
    return (ids == edges[None, :, 1]).astype(dtype) - (ids == edges[None, :, 0]).astype(dtype)


def _spring_forces(
    pos: jax.Array,  # [N, 2]
    vel: jax.Array,  # [N, 2]
    edges: jax.Array,  # [E, 2]
    stiffness: jax.Array,  # [E]
    rest: jax.Array,  # [E] 基準自然長
    active: jax.Array,  # [E] 能動収縮力 (引く向きが正)
    damping: jax.Array,  # [E] 軸方向粘性減衰の係数 c_e
    incidence: jax.Array,  # [N, E] 固定順の和に使う
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """受動バネ + 能動収縮力 + 軸方向粘性減衰。

    dL/dt は両端の相対速度を辺軸へ射影して求める。減衰は辺の伸縮だけを減らし、
    身体全体の並進・剛体回転には働かない。質点ごとの和は固定順の行列和で取る
    (GPU の atomic scatter より速く、加算順も決定的)。戻り値: node_forces, tension, length, u。
    """
    d = pos[edges[:, 0]] - pos[edges[:, 1]]  # [E, 2]
    length = jnp.linalg.norm(d, axis=1)  # [E]
    u = d / jnp.maximum(length, _LEN_EPS)[:, None]
    dLdt = jnp.sum(u * (vel[edges[:, 0]] - vel[edges[:, 1]]), axis=1)  # [E]
    tension = stiffness * (length - rest) + active + damping * dLdt  # 張りが正
    contrib = tension[:, None] * u  # [E, 2]
    forces = jnp.stack(
        [jnp.sum(incidence * contrib[None, :, k], axis=1) for k in range(2)], axis=1
    )  # [N, 2]
    return forces, tension, length, u


def _contact_forces(
    pos: jax.Array,
    vel: jax.Array,
    p: SimParams,
) -> tuple[jax.Array, jax.Array]:
    """床 (y=0) とのペナルティ反力と滑らかなクーロン摩擦。戻り値: total [N,2], normal [N]。"""
    penetration = jnp.maximum(0.0, -pos[:, 1])  # めり込み量 [N]
    normal = p.k_contact * penetration - p.c_contact * jnp.minimum(0.0, vel[:, 1])
    # 減衰項は接触中だけ効かせる (空中で落下速度に比例した抗力が出るのを防ぐ)
    normal = jnp.maximum(0.0, normal) * (penetration > 0.0)  # 引っ張らない
    friction = -p.mu_friction * normal * jnp.tanh(vel[:, 0] / p.friction_vel_eps)
    total = jnp.stack([friction, normal], axis=1)
    return total, normal


def _make_step(
    gene: jax.Array,
    spec: CreatureSpec,
    p: SimParams,
    integ: Integrator,
):
    """1 ステップの物理を組み立てる。rollout (全フレーム記録) と summary (集計のみ) で共有。"""
    muscle_idx = spec.muscle_idx
    n_m = int(muscle_idx.shape[0])
    amp, freq, phase = gene[:n_m], gene[n_m : 2 * n_m], gene[2 * n_m :]
    edges, stiffness, base_rest = spec.edges, spec.stiffness, spec.rest_lengths
    masses_inv = 1.0 / spec.masses
    dt = p.dt
    g_vec = jnp.array([0.0, -p.gravity], dtype=spec.base_pos.dtype)
    model = get_model(p.muscle_model)  # static (trace 時に確定)
    damping = _edge_damping_coeff(stiffness, spec.masses, edges, p.edge_damping_ratio)
    incidence = _incidence(edges, spec.n_nodes, spec.base_pos.dtype)

    def _edges(t: jax.Array, pos: jax.Array, vel: jax.Array):
        rest_t = model.rest_lengths(base_rest, amp, freq, phase, t, muscle_idx)
        active = model.active_tension(amp, freq, phase, t, muscle_idx, spec.n_edges)
        spring_f, tension, length, u = _spring_forces(
            pos, vel, edges, stiffness, rest_t, active, damping, incidence
        )
        return rest_t, spring_f, tension, length, u, active

    def observe(pos: jax.Array, vel: jax.Array, t: jax.Array):
        rest_t, spring_f, tension, _, u, active = _edges(t, pos, vel)
        contact_f, normal = _contact_forces(pos, vel, p)
        dLdt = jnp.sum(u * (vel[edges[:, 0]] - vel[edges[:, 1]]), axis=1)
        drive = model.drive_power(active, tension, base_rest, dLdt, amp, freq, phase, t, muscle_idx)
        acc = (spring_f + contact_f) * masses_inv[:, None] + g_vec
        obs = dict(
            pos=pos,
            vel=vel,
            spring_force=tension,
            contact_force=normal,
            drive_power=drive,
            rest_lengths=rest_t,
        )
        return obs, acc

    def _accel(t: jax.Array, pos: jax.Array, vel: jax.Array) -> jax.Array:
        _, spring_f, _, _, _, _ = _edges(t, pos, vel)
        contact_f, _ = _contact_forces(pos, vel, p)
        return (spring_f + contact_f) * masses_inv[:, None] + g_vec

    def rk4_update(pos: jax.Array, vel: jax.Array, t: jax.Array) -> tuple[jax.Array, jax.Array]:
        def f(tt: jax.Array, pv: jax.Array, vv: jax.Array):
            return vv, _accel(tt, pv, vv)

        k1p, k1v = f(t, pos, vel)
        k2p, k2v = f(t + dt / 2, pos + dt / 2 * k1p, vel + dt / 2 * k1v)
        k3p, k3v = f(t + dt / 2, pos + dt / 2 * k2p, vel + dt / 2 * k2v)
        k4p, k4v = f(t + dt, pos + dt * k3p, vel + dt * k3v)
        return (
            pos + dt / 6.0 * (k1p + 2 * k2p + 2 * k3p + k4p),
            vel + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v),
        )

    def step(pos: jax.Array, vel: jax.Array, t: jax.Array):
        obs, acc = observe(pos, vel, t)
        if integ == "rk4":
            pos_n, vel_n = rk4_update(pos, vel, t)
        else:
            vel_n = vel + dt * acc  # 半陰的オイラー
            pos_n = pos + dt * vel_n
        return pos_n, vel_n, obs

    return observe, step


def rollout(
    gene: jax.Array,
    spec: CreatureSpec,
    p: SimParams,
    integrator: Integrator | None = None,
) -> RolloutResult:
    """全フレームを記録するロールアウト (replay・記録用)。"""
    integ = p.integrator if integrator is None else integrator
    observe, step = _make_step(gene, spec, p, integ)
    dt = p.dt

    def scan_fn(carry, _):
        pos, vel, t = carry
        pos_n, vel_n, obs = step(pos, vel, t)
        return (pos_n, vel_n, t + dt), obs

    t0 = jnp.float32(0.0)
    init = (spec.base_pos, jnp.zeros((spec.n_nodes, 2), dtype=jnp.float32), t0)
    (pos_T, vel_T, t_T), outs = jax.lax.scan(scan_fn, init, None, length=p.n_steps)

    final_obs, _ = observe(pos_T, vel_T, t_T)
    outs = jax.tree.map(lambda s, f: jnp.concatenate([s, f[None]], axis=0), outs, final_obs)
    times = jnp.arange(p.n_steps + 1, dtype=jnp.float32) * dt
    return RolloutResult(
        pos=outs["pos"],
        vel=outs["vel"],
        times=times,
        spring_force=outs["spring_force"],
        contact_force=outs["contact_force"],
        drive_power=outs["drive_power"],
        rest_lengths=outs["rest_lengths"],
    )


@dataclass(frozen=True)
class RolloutSummary:
    """評価に必要な集計値だけを持つ。フレーム列を保持しないのでメモリはステップ数に依存しない。"""

    distance: jax.Array
    max_penetration: jax.Array
    drive_energy: jax.Array
    contact_rate: jax.Array
    finite: jax.Array


def rollout_summary(
    gene: jax.Array,
    spec: CreatureSpec,
    p: SimParams,
    integrator: Integrator | None = None,
) -> RolloutSummary:
    """集計値を scan 内で累積する評価用ロールアウト。中間フレームをメモリに残さない。"""
    integ = p.integrator if integrator is None else integrator
    observe, step = _make_step(gene, spec, p, integ)
    dt = p.dt
    masses = spec.masses
    m_total = jnp.sum(masses)

    def com_x(pos: jax.Array) -> jax.Array:
        return jnp.sum(pos[:, 0] * masses) / m_total

    def penetration(pos: jax.Array) -> jax.Array:
        return jnp.maximum(0.0, -jnp.min(pos[:, 1]))

    x0 = com_x(spec.base_pos)
    z = jnp.float32(0.0)
    init = (
        spec.base_pos,
        jnp.zeros((spec.n_nodes, 2), dtype=jnp.float32),
        jnp.float32(0.0),
        z,  # drive energy (和)
        z,  # max penetration
        z,  # contact count
        jnp.array(True),  # finite
    )

    def scan_fn(carry, _):
        pos, vel, t, energy, pen, contact, finite = carry
        pos_n, vel_n, obs = step(pos, vel, t)
        energy = energy + jnp.sum(jnp.maximum(0.0, obs["drive_power"]))
        pen = jnp.maximum(pen, penetration(pos))
        contact = contact + jnp.sum(obs["contact_force"] > 0)
        finite = finite & jnp.all(jnp.isfinite(pos)) & jnp.all(jnp.isfinite(vel))
        return (pos_n, vel_n, t + dt, energy, pen, contact, finite), None

    (pos_T, vel_T, t_T, energy, pen, contact, finite), _ = jax.lax.scan(
        scan_fn, init, None, length=p.n_steps
    )
    obs_T, _ = observe(pos_T, vel_T, t_T)
    pen = jnp.maximum(pen, penetration(pos_T))
    contact = contact + jnp.sum(obs_T["contact_force"] > 0)
    finite = finite & jnp.all(jnp.isfinite(pos_T)) & jnp.all(jnp.isfinite(vel_T))
    n_frames = p.n_steps + 1
    return RolloutSummary(
        distance=com_x(pos_T) - x0,
        max_penetration=pen,
        drive_energy=energy * dt,
        contact_rate=contact / (n_frames * spec.n_nodes),
        finite=finite,
    )
