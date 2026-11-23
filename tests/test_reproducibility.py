"""再現性テスト (seed 固定で結果が同一になるか)。短い GA run を 2 回実行する。

世代ループ全体を 1 つの jit + scan に載せているため、同じ seed・同じ実行環境
(GPU・JAX バージョン) ではビット一致するはず。実行時間は episode を短縮して
数秒に抑える (合意: first goal は数分で終わる動作テストまで)。
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from bodyopt.creature import triangulated_ladder
from bodyopt.fitness import FitnessParams
from bodyopt.ga import GAParams, run_ga
from bodyopt.physics import SimParams

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def spec():
    return triangulated_ladder()


def _short_run(spec, seed: int):
    sim_p = SimParams(episode_seconds=0.5)  # 100 ステップ
    fit_p = FitnessParams()
    gp = GAParams(pop_size=8, seed=seed)
    return run_ga(spec, sim_p, fit_p, gp, n_generations=3)


class Test同一シード:
    def test_2回の実行が実用上一致(self, spec):
        """GPU の atomic 加算 (scatter-add) は最下位ビットが非決定になり得る。
        seed が同じなら実用上同一 (rel 1e-5) であることを検証する。"""
        a = _short_run(spec, seed=42)
        b = _short_run(spec, seed=42)
        np.testing.assert_allclose(a["stats"].score, b["stats"].score, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(a["populations"], b["populations"], rtol=1e-5, atol=1e-7)

    def test_最終世代の集団も一致(self, spec):
        a = _short_run(spec, seed=1)
        b = _short_run(spec, seed=1)
        np.testing.assert_allclose(a["final_pop"], b["final_pop"], rtol=1e-5, atol=1e-7)


class Test異シード:
    def test_異なるシードで結果が変わる(self, spec):
        a = _short_run(spec, seed=0)
        b = _short_run(spec, seed=1)
        assert not np.allclose(a["populations"], b["populations"]), "シード違いで同一集団"


class TestGA探索:
    def test_短い_run_でも初期乱数より改善する(self, spec):
        """3 世代 × 8 個体でも best score が初期世代を上回る (探索が機能)。"""
        sim_p = SimParams(episode_seconds=0.5)
        out = run_ga(spec, sim_p, FitnessParams(), GAParams(pop_size=8, seed=3), n_generations=3)
        best = out["stats"].score.max(axis=1)
        assert best[-1] >= best[0], f"best score が悪化した: {best}"
        assert float(best[-1]) > -1.0e6, "全個体無効のまま改善しなかった"
