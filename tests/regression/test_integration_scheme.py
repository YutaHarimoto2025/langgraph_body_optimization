"""積分法の回帰比較 (tests/regression/)。

半陰的オイラー法 (第一候補) の刻み依存性と、RK4 との差を同一遺伝子で確認する。

実測 (2026-11-27、active_force モデル・辺剛性 2000 N/m・重力あり):
- 無駆動は着地後ほぼ静止し、5 ms / 2.5 ms / RK4 の距離差は 1 cm 未満。
- ランダム位相の駆動個体 (20/30/40 N) は 0.01〜0.06 m で、差は 5 cm 未満。
- 位相が揃った個体 (GA が選ぶ解) は踏切の位相が刻みに敏感になり得る。
- したがって「距離の一致」は無駆動で 2 cm 以内に限り保証し、
  駆動個体は有効判定の一致と数値破綻 (非有限・発散) の防止のみを保証する。
  GA が選ぶ最良個体の刻み頑健性は tests/regression/test_best_individual_scheme.py
  (slow) で検証する。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bodyopt.creature import triangulated_ladder
from bodyopt.fitness import FitnessParams, evaluate_gene
from bodyopt.physics import SimParams

pytestmark = pytest.mark.regression

DIST_TOL_PASSIVE = 0.02  # 受動着地の許容差 [m]
DIST_MAX_DRIVE = 10.0  # 駆動個体の破綻判定上限 [m] (5 秒で 2 m/s 相当)


@pytest.fixture(scope="module")
def spec():
    return triangulated_ladder()


def _gene(spec, amp: float, phase_seed: int, freq: float = 1.0) -> jax.Array:
    n_m = spec.n_muscles
    rng = np.random.default_rng(phase_seed)
    return jnp.asarray(
        np.concatenate(
            [np.full(n_m, amp), np.full(n_m, freq), rng.uniform(0, 2 * np.pi, n_m)]
        )
    ).astype(jnp.float32)


@pytest.fixture(scope="module")
def walk_genes(spec):
    """無駆動 + 中振幅 2 個体 (歩行域)。scheme 間一致を保証する対象。"""
    return [
        _gene(spec, 0.0, 0),  # 無駆動
        _gene(spec, 20.0, 1),  # 中振幅 [N]
        _gene(spec, 30.0, 2),
    ]


@pytest.fixture(scope="module")
def jump_gene(spec):
    """全辺最大振幅 (跳躍域)。刻み感度の記録用。"""
    return _gene(spec, 40.0, 3)


def _eval(gene, spec, dt, integrator):
    p = SimParams(dt=dt, integrator=integrator)
    return evaluate_gene(gene, spec, p, FitnessParams())


SCHEMES = [
    ("semi_implicit_euler", 0.005),
    ("semi_implicit_euler", 0.0025),
    ("rk4", 0.005),
]


class Test無駆動の一致:
    def test_刻みと手法間で一致(self, spec, walk_genes):
        dists = [float(_eval(walk_genes[0], spec, dt, integ).distance) for integ, dt in SCHEMES]
        assert max(dists) - min(dists) < DIST_TOL_PASSIVE, f"無駆動が scheme 間で変わった: {dists}"

    def test_無駆動個体は破綻しない(self, spec, walk_genes):
        for integ, dt in SCHEMES:
            s = _eval(walk_genes[0], spec, dt, integ)
            assert bool(s.valid), f"無駆動が無効判定: {integ} {dt}"
            assert abs(float(s.distance)) < DIST_MAX_DRIVE


class Test駆動個体の破綻防止:
    """駆動個体は位相次第で跳躍モードになり刻みに敏感。破綻の防止のみ保証する。"""

    @pytest.mark.parametrize("i", [1, 2])
    def test_距離が発散しない(self, spec, walk_genes, i):
        dists = [float(_eval(walk_genes[i], spec, dt, integ).distance) for integ, dt in SCHEMES]
        assert all(np.isfinite(dists)), f"距離が非有限: {dists}"
        assert max(abs(d) for d in dists) < DIST_MAX_DRIVE, f"scheme 間で距離 {dists}"

    @pytest.mark.parametrize("i", [1, 2])
    def test_有効判定が一致(self, spec, walk_genes, i):
        flags = [bool(_eval(walk_genes[i], spec, dt, integ).valid) for integ, dt in SCHEMES]
        assert len(set(flags)) == 1, f"有効判定が scheme 間で変わった: {flags}"
