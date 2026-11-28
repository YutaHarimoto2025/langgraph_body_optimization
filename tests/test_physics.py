"""物理シミュレータの静的テスト。GPU/CPU どちらでも数秒で終わる規模。"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bodyopt.creature import triangulated_ladder
from bodyopt.fitness import com_trajectory
from bodyopt.models import get_model
from bodyopt.physics import SimParams, rollout


@pytest.fixture(scope="module")
def spec():
    return triangulated_ladder()


@pytest.fixture(scope="module")
def short_params():
    return SimParams(episode_seconds=0.5)  # 100 ステップで高速


def _flat_gene(spec, amp=0.0):
    n_m = int(spec.muscle_idx.shape[0])
    return jnp.concatenate([jnp.full(n_m, amp), jnp.full(n_m, 1.0), jnp.zeros(n_m)])


class Test重力:
    def test_自由落下は重力加速度に一致(self, spec):
        """非接地・無駆動なら重心は半陰的オイラーの放物運動をする。"""
        p = SimParams(episode_seconds=0.08, integrator="semi_implicit_euler")  # 16 ステップ
        r = rollout(_flat_gene(spec), spec, p)
        com_y = np.asarray(com_trajectory(r.pos, spec))[:, 1]
        n = np.arange(p.n_steps + 1, dtype=np.float32)
        expected = com_y[0] - p.gravity * p.dt**2 * n * (n + 1) / 2
        np.testing.assert_allclose(com_y, expected, atol=1e-6)


class Test接触と重力の釣り合い:
    def test_無駆動で床に沈まない(self, spec):
        """振幅 0 の身体は 5 cm 落ちて着地する。床反力が重力を受け止め、底列は床面付近に留まる。

        上段は重力で潰れて倒れる (固定形状は受動では立っていられない) ため、
        ここでは底列の沈み込みだけを見る。
        """
        p = SimParams(episode_seconds=1.0)
        pos = np.asarray(rollout(_flat_gene(spec), spec, p).pos)
        assert pos[..., 1].min() > -0.02, f"床にめり込みすぎた: {pos[..., 1].min():.4f} m"
        assert np.abs(pos[-100:, :, 1]).min() < 0.02, "床に接している質点がない (浮いた)"


class Test力の整合:
    def test_バネ力の作用反作用(self, spec):
        from bodyopt.physics import _incidence, _spring_forces

        pos = np.asarray(spec.base_pos) + 0.05  # 適当に変形
        vel = np.zeros_like(pos)
        forces, _, _, _ = _spring_forces(
            jnp.asarray(pos, dtype=jnp.float32),
            jnp.asarray(vel, dtype=jnp.float32),
            spec.edges,
            spec.stiffness,
            spec.rest_lengths,
            jnp.zeros(spec.n_edges),  # 能動張力なし
            jnp.zeros(spec.n_edges),  # 減衰なし
            _incidence(spec.edges, spec.n_nodes, jnp.float32),
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


class Test粘性減衰:
    @staticmethod
    def _damping(spec, zeta=0.05):
        from bodyopt.physics import _edge_damping_coeff

        return _edge_damping_coeff(spec.stiffness, spec.masses, spec.edges, zeta)

    @staticmethod
    def _forces(spec, pos, vel):
        from bodyopt.physics import _incidence, _spring_forces

        e = spec.n_edges
        return _spring_forces(
            jnp.asarray(pos, dtype=jnp.float32),
            jnp.asarray(vel, dtype=jnp.float32),
            spec.edges,
            jnp.zeros(e),  # バネなし: 減衰だけを見る
            spec.rest_lengths,
            jnp.zeros(e),  # 能動張力なし
            Test粘性減衰._damping(spec),
            _incidence(spec.edges, spec.n_nodes, jnp.float32),
        )[0]

    def test_作用反作用(self, spec):
        pos = np.asarray(spec.base_pos) + 0.05
        vel = np.random.default_rng(0).normal(size=pos.shape) * 0.3
        total = np.asarray(self._forces(spec, pos, vel)).sum(axis=0)
        assert np.abs(total).max() < 1e-4, f"力の総和がゼロでない: {total}"

    def test_剛体並進では力が出ない(self, spec):
        pos = np.asarray(spec.base_pos)
        vel = np.tile(np.array([0.8, -0.3]), (len(pos), 1))  # 全ノード同速度
        forces = np.asarray(self._forces(spec, pos, vel))
        assert np.abs(forces).max() < 1e-4, f"並進で減衰力が出た: {np.abs(forces).max():.2e}"

    def test_剛体回転では力が出ない(self, spec):
        pos = np.asarray(spec.base_pos)
        r = pos - pos.mean(axis=0)
        omega = 1.3
        vel = omega * np.stack([-r[:, 1], r[:, 0]], axis=1)  # 辺長が変わらない回転
        forces = np.asarray(self._forces(spec, pos, vel))
        assert np.abs(forces).max() < 1e-4, f"回転で減衰力が出た: {np.abs(forces).max():.2e}"

    def test_減衰力の仕事率は非正(self, spec):
        pos = np.asarray(spec.base_pos) + 0.05
        vel = np.random.default_rng(1).normal(size=pos.shape) * 0.5
        forces = np.asarray(self._forces(spec, pos, vel))
        power = float(np.sum(forces * vel))
        assert power <= 1e-8, f"減衰がエネルギーを注入した: {power:.3e}"


class Test筋肉モデル:
    def test_能動張力は0以上かつ上限以下(self, spec):
        model = get_model("active_force")
        n_m = int(spec.muscle_idx.shape[0])
        amp = jnp.full((n_m,), model.gene_hi)
        freq = jnp.full((n_m,), 1.0)
        phase = jnp.linspace(0.0, 2.0 * jnp.pi, n_m)
        ts = jnp.linspace(0.0, 2.0, 200)
        act = jax.vmap(
            lambda tt: model.active_tension(
                amp, freq, phase, tt, spec.muscle_idx, spec.n_edges
            )
        )(ts)
        assert float(act.min()) >= 0.0
        assert float(act.max()) <= model.gene_hi + 1e-5
        # 周期平均は A/2
        np.testing.assert_allclose(
            float(jnp.mean(act[:, spec.muscle_idx])), model.gene_hi / 2.0, rtol=0.05
        )

    def test_振幅ゼロなら受動バネのみ(self, spec):
        model = get_model("active_force")
        n_m = int(spec.muscle_idx.shape[0])
        zeros = jnp.zeros(n_m)
        freq = jnp.full(n_m, 1.0)
        t = jnp.float32(1.0)
        act = model.active_tension(zeros, freq, zeros, t, spec.muscle_idx, spec.n_edges)
        assert float(jnp.abs(act).max()) == 0.0
        rest = model.rest_lengths(
            spec.rest_lengths, zeros, freq, zeros, t, spec.muscle_idx
        )
        np.testing.assert_allclose(np.asarray(rest), np.asarray(spec.rest_lengths))

    def test_自然長モデルは自然長が変わる(self, spec):
        model = get_model("rest_length")
        n_m = int(spec.muscle_idx.shape[0])
        amp = jnp.full((n_m,), model.gene_hi)
        rest = model.rest_lengths(
            spec.rest_lengths,
            amp,
            jnp.ones(n_m),  # 1 Hz
            jnp.zeros(n_m),
            jnp.float32(0.25),  # sin(2π·1·0.25) = 1
            spec.muscle_idx,
        )
        idx = np.asarray(spec.muscle_idx)
        ratio = np.asarray(rest)[idx] / np.asarray(spec.rest_lengths)[idx]
        np.testing.assert_allclose(ratio, 1.0 + model.gene_hi, rtol=1e-5)


class Test集計ロールアウト:
    def test_全フレーム版と集計値が一致(self, spec, short_params):
        from bodyopt.physics import rollout, rollout_summary

        n_m = int(spec.muscle_idx.shape[0])
        gene = jnp.concatenate(
            [jnp.full(n_m, 30.0), jnp.full(n_m, 2.0), jnp.linspace(0.0, 3.0, n_m)]
        )
        r = rollout(gene, spec, short_params)
        s = rollout_summary(gene, spec, short_params)
        com = com_trajectory(r.pos, spec)
        np.testing.assert_allclose(float(s.distance), float(com[-1, 0] - com[0, 0]), rtol=1e-5)
        np.testing.assert_allclose(
            float(s.max_penetration), float(np.maximum(0.0, -np.asarray(r.pos)[..., 1].min())),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            float(s.drive_energy),
            float(np.sum(np.maximum(0.0, np.asarray(r.drive_power)[:-1])) * short_params.dt),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            float(s.contact_rate), float((np.asarray(r.contact_force) > 0).mean()), rtol=1e-5
        )


class Test記録形状:
    def test_ロールアウトのフレーム数(self, spec, short_params):
        r = rollout(_flat_gene(spec, amp=0.1), spec, short_params)
        n_steps = short_params.n_steps
        assert r.pos.shape == (n_steps + 1, spec.n_nodes, 2)
        assert r.spring_force.shape == (n_steps + 1, spec.n_edges)
        assert float(r.times[0]) == 0.0
        np.testing.assert_allclose(float(r.times[-1]), short_params.dt * n_steps, rtol=1e-5)
