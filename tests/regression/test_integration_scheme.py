"""積分法の回帰比較 (tests/regression/)。

半陰的オイラー法 (第一候補) の刻み依存性と、RK4 との差を同一遺伝子で確認する。

first try の実測 (2026-11-23、c_contact=89.44):
- 無駆動は 5 ms / 2.5 ms / RK4 間で完全に一致する。
- 中振幅でも位相によっては跳躍モード (負方向へ移動) になり、
  その場合の着地位相は刻みに敏感で距離差 0.3 m 程度出る
  (amp=0.10, seed=1 の右歩行個体は 0.02 m 以内で一致)。
- したがって「距離の厳密な一致」は無駆動にのみ保証し、
  駆動個体は有効判定の一致と数値破綻の防止 (0.5 m 以内) を保証する。
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

DIST_TOL_STRICT = 1e-3  # 無駆動の許容差 [m]
DIST_TOL_DRIVE = 0.5  # 駆動個体 (モード依存) の破綻防止上限 [m]


@pytest.fixture(scope="module")
def spec():
    return triangulated_ladder()


def _gene(spec, amp: float, phase_seed: int) -> jax.Array:
    n_m = spec.n_muscles
    rng = np.random.default_rng(phase_seed)
    return jnp.asarray(
        np.concatenate([np.full(n_m, amp), rng.uniform(0, 2 * np.pi, n_m)])
    ).astype(jnp.float32)


@pytest.fixture(scope="module")
def walk_genes(spec):
    """無駆動 + 中振幅 2 個体 (歩行域)。scheme 間一致を保証する対象。"""
    return [
        _gene(spec, 0.0, 0),  # 無駆動
        _gene(spec, 0.10, 1),
        _gene(spec, 0.15, 2),
    ]


@pytest.fixture(scope="module")
def jump_gene(spec):
    """全辺最大振幅 (跳躍域)。刻み感度の記録用。"""
    return _gene(spec, 0.20, 3)


def _eval(gene, spec, dt, integrator):
    p = SimParams(dt=dt, integrator=integrator)
    return evaluate_gene(gene, spec, p, FitnessParams())


SCHEMES = [
    ("semi_implicit_euler", 0.005),
    ("semi_implicit_euler", 0.0025),
    ("rk4", 0.005),
]


class Test無駆動の一致:
    def test_刻みと手法間で完全に一致(self, spec, walk_genes):
        dists = [float(_eval(walk_genes[0], spec, dt, integ).distance) for integ, dt in SCHEMES]
        assert max(dists) - min(dists) < DIST_TOL_STRICT, f"無駆動が scheme 間で変わった: {dists}"

    def test_無駆動個体は動かない(self, spec, walk_genes):
        for integ, dt in SCHEMES:
            s = _eval(walk_genes[0], spec, dt, integ)
            assert abs(float(s.distance)) < DIST_TOL_STRICT


class Test駆動個体の破綻防止:
    """駆動個体は位相次第で跳躍モードになり刻みに敏感。破綻の防止のみ保証する。"""

    @pytest.mark.parametrize("i", [1, 2])
    def test_距離が破綻しない(self, spec, walk_genes, i):
        dists = [float(_eval(walk_genes[i], spec, dt, integ).distance) for integ, dt in SCHEMES]
        assert max(dists) - min(dists) < DIST_TOL_DRIVE, f"scheme 間で距離 {dists}"

    @pytest.mark.parametrize("i", [1, 2])
    def test_有効判定が一致(self, spec, walk_genes, i):
        flags = [bool(_eval(walk_genes[i], spec, dt, integ).valid) for integ, dt in SCHEMES]
        assert len(set(flags)) == 1, f"有効判定が scheme 間で変わった: {flags}"
