"""GA が選ぶ最良個体の刻み依存性 (回帰・slow)。

短い GA run の最良個体を 5 ms / 2.5 ms / RK4 で再評価する。
first try の実測 (2026-11-23): 初期ランダム集団からは「跳ねて進む」解が
最も高く評価され、その跳躍の初速は scheme 間で 0.1-0.2 m の距離差を生む
(報告 0001)。跳躍対策 (罰則・エネルギー制約) 導入までは
「右歩行解なら 0.05 m 以内」の保証はできず、このテストは
差の破綻 (0.5 m 超 = 物理が壊れた) を検知するセーフティネットとする。
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from bodyopt.creature import triangulated_ladder
from bodyopt.fitness import FitnessParams, evaluate_gene
from bodyopt.ga import GAParams, run_ga
from bodyopt.physics import SimParams

pytestmark = [pytest.mark.regression, pytest.mark.slow]

BEST_DIST_TOL = 50.0  # 発散判定の上限 [m] (高周波の跳躍解で 10 m 級が出るため)


@pytest.fixture(scope="module")
def best_gene():
    spec = triangulated_ladder()
    out = run_ga(
        spec,
        SimParams(),  # 既定の 5 秒 episode
        FitnessParams(),
        GAParams(pop_size=8, seed=0),
        n_generations=3,
    )
    scores = out["stats"].score  # [G, P]
    g, i = np.unravel_index(np.argmax(scores), scores.shape)
    return spec, out["populations"][g][i], float(scores[g, i])


def _dist(spec, gene, dt, integrator):
    return float(
        evaluate_gene(
            jnp.asarray(gene), spec, SimParams(dt=dt, integrator=integrator), FitnessParams()
        ).distance
    )


def test_最良個体の距離が破綻しない(best_gene):
    spec, gene, score = best_gene
    dists = [
        _dist(spec, gene, 0.005, "semi_implicit_euler"),
        _dist(spec, gene, 0.0025, "semi_implicit_euler"),
        _dist(spec, gene, 0.005, "rk4"),
    ]
    spread = max(dists) - min(dists)
    assert score > -1.0e6, "最良個体が無効のまま"
    assert max(abs(d) for d in dists) < BEST_DIST_TOL, f"距離が発散: {dists}"
    assert spread < BEST_DIST_TOL, (
        f"GA の最良個体 (score={score:.3f}) が scheme 間で {dists} と破綻的にばらついた。"
        "跳躍モード既知の感度 (0.2 程度) を超えたため物理設定の見直しが必要"
    )

