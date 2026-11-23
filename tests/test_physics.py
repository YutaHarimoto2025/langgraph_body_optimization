"""物理シミュレータの静的テスト。GPU/CPU どちらでも数秒で終わる規模。"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from bodyopt.creature import triangulated_ladder
from bodyopt.physics import SimParams, muscle_rest_lengths, rollout


@pytest.fixture(scope="module")
def spec():
    return triangulated_ladder()


@pytest.fixture(scope="module")
def short_params():
    return SimParams(episode_seconds=0.5)  # 100 ステップで高速


def _flat_gene(spec, amp=0.0):
    n_m = int(spec.muscle_idx.shape[0])
    return jnp.concatenate([jnp.full(n_m, amp), jnp.zeros(n_m)])


class Test静止:
    def test_無駆動では身体が動かない(self, spec, short_params):
        """振幅 0 なら初期形状が釣り合い (自然長=初期辺長) で、重力+反力のみ。"""
        r = rollout(_flat_gene(spec), spec, short_params)
        pos = np.asarray(r.pos)
        drift = np.abs(pos[-1] - pos[0]).max()
        assert drift < 5e-3, f"無駆動で {drift:.4f} m 変位した"

    def test_無駆動で水平方向に流れない(self, spec, short_params):
        r = rollout(_flat_gene(spec), spec, short_params)
        dx = np.asarray(r.pos[-1, :, 0] - r.pos[0, :, 0])
        assert np.abs(dx).max() < 1e-4, "静止個体が横に流れた"


class Test力の整合:
    def test_バネ力の作用反作用(self, spec):
        from bodyopt.physics import _spring_forces

        pos = np.asarray(spec.base_pos) + 0.05  # 適当に変形
        forces, _, _ = _spring_forces(
            jnp.asarray(pos, dtype=jnp.float32),
            spec.edges,
            spec.stiffness,
            spec.rest_lengths,
        )
        total = np.asarray(forces).sum(axis=0)
        assert np.abs(total).max() < 1e-4, f"力の総和がゼロでない: {total}"

    def test_空中では接触力がゼロ(self, spec, short_params):
        from bodyopt.physics import _contact_forces

        pos = spec.base_pos + jnp.array([[0.0, 5.0]])  # 高く浮かす
        vel = jnp.zeros_like(pos)
        _, normal = _contact_forces(pos, vel, short_params)
        assert float(jnp.max(normal)) == 0.0

    def test_めり込むと上向き反力(self, spec, short_params):
        from bodyopt.physics import _contact_forces

        pos = spec.base_pos.at[:, 1].set(-0.01)  # 全質点を 1cm めり込ませる
        vel = jnp.zeros_like(pos)
        _, normal = _contact_forces(pos, vel, short_params)
        assert float(jnp.min(normal)) > 0.0


class Test自然長:
    def test_振幅ゼロで基準自然長(self, spec):
        n_m = int(spec.muscle_idx.shape[0])
        rest = muscle_rest_lengths(
            spec.rest_lengths,
            jnp.zeros(n_m),
            jnp.linspace(0, jnp.pi, n_m),
            2.0 * jnp.pi,
            spec.muscle_idx,
            jnp.float32(1.23),
        )
        np.testing.assert_allclose(np.asarray(rest), np.asarray(spec.rest_lengths), rtol=1e-6)

    def test_周期指令の上下限(self, spec):
        """自然長は [L(1-a), L(1+a)] に収まる。"""
        n_m = int(spec.muscle_idx.shape[0])
        amp = jnp.full(n_m, 0.2)
        phase = jnp.zeros(n_m)
        ts = jnp.linspace(0, 2.0, 200)
        rest = muscle_rest_lengths(
            spec.rest_lengths, amp, phase, 2.0 * jnp.pi, spec.muscle_idx, ts
        )
        base = np.asarray(spec.rest_lengths)[np.asarray(spec.muscle_idx)]
        ratio = np.asarray(rest) / base
        assert ratio.min() >= 0.8 - 1e-5
        assert ratio.max() <= 1.2 + 1e-5


class Test記録形状:
    def test_ロールアウトのフレーム数(self, spec, short_params):
        r = rollout(_flat_gene(spec, amp=0.1), spec, short_params)
        n_steps = short_params.n_steps
        assert r.pos.shape == (n_steps + 1, spec.n_nodes, 2)
        assert r.spring_force.shape == (n_steps + 1, spec.n_edges)
        assert float(r.times[0]) == 0.0
        np.testing.assert_allclose(float(r.times[-1]), short_params.dt * n_steps, rtol=1e-5)
