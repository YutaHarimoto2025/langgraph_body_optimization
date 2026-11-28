"""GA 演算子の静的テスト。"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bodyopt.creature import triangulated_ladder
from bodyopt.ga import GAParams, _clip_gene, evolve_population, gene_bounds, inject_worst, random_population
from bodyopt.models import get_model

MODEL = "active_force"
GENE_HI = get_model(MODEL).gene_hi


@pytest.fixture(scope="module")
def spec():
    return triangulated_ladder()


class Test初期集団:
    def test_遺伝子の範囲(self, spec):
        n_m = int(spec.muscle_idx.shape[0])
        key = jax.random.PRNGKey(0)
        pop = random_population(key, 32, n_m, GENE_HI)
        arr = np.asarray(pop)
        assert arr.shape == (32, 4 * n_m)
        assert (arr[:, :n_m] >= 0.0).all()
        assert (arr[:, :n_m] <= GENE_HI).all()
        assert (arr[:, n_m : 2 * n_m] >= 0.1 - 1e-6).all()
        assert (arr[:, n_m : 2 * n_m] <= 1.0 + 1e-6).all()
        assert (arr[:, 2 * n_m :] < 2 * np.pi).all()


class Testクリップ:
    def test_位相の周期折返し(self):
        n = 4
        genes = jnp.full((3, 3 * n), 7.0)  # 位相が 2π より大きい
        lo, hi = gene_bounds(n, MODEL)
        out = np.asarray(_clip_gene(genes, lo, hi))
        assert (out[:, 2 * n :] < 2 * np.pi).all()

    def test_振幅の上限クリップ(self):
        n = 4
        genes = jnp.full((2, 3 * n), 99.0)
        lo, hi = gene_bounds(n, MODEL)
        out = np.asarray(_clip_gene(genes, lo, hi))
        assert (out[:, :n] <= GENE_HI + 1e-6).all()
        assert (out[:, n : 2 * n] <= 1.0 + 1e-6).all()


class Test移民:
    def test_移民を入れても集団サイズと範囲が保たれる(self, spec):
        n_m = int(spec.muscle_idx.shape[0])
        key = jax.random.PRNGKey(5)
        pop = random_population(key, 20, n_m, GENE_HI)
        new_pop = inject_worst(pop, jnp.arange(20.0), jax.random.PRNGKey(6), 5, GENE_HI)
        arr = np.asarray(new_pop)
        base = np.asarray(pop)
        assert arr.shape == (20, 4 * n_m)
        # 下位 5 個体だけが置換され、残りは変わらない
        assert not np.allclose(arr[:5], base[:5])
        np.testing.assert_allclose(arr[5:], base[5:])
        assert (arr[:, n_m : 2 * n_m] >= 0.1 - 1e-5).all()
        assert (arr[:, n_m : 2 * n_m] <= 1.0 + 1e-5).all()


class Test集団更新:
    def test_エリートが次世代に残る(self, spec):
        n_m = int(spec.muscle_idx.shape[0])
        lo, hi = gene_bounds(n_m, MODEL)
        key = jax.random.PRNGKey(1)
        pop = random_population(key, 12, n_m, GENE_HI)
        fit = jnp.linspace(0.0, 1.0, 12)  # 末尾が最良
        gp = GAParams(pop_size=12, n_elite=2)
        new_pop, _ = evolve_population(pop, fit, key, gp, lo, hi)
        # fit 降順の先頭 2 個体 (index 11, 10) が new_pop[:2] にいる
        np.testing.assert_allclose(np.asarray(new_pop[0]), np.asarray(pop[11]), rtol=1e-6)
        np.testing.assert_allclose(np.asarray(new_pop[1]), np.asarray(pop[10]), rtol=1e-6)

    def test_集団サイズが保たれる(self, spec):
        n_m = int(spec.muscle_idx.shape[0])
        lo, hi = gene_bounds(n_m, MODEL)
        key = jax.random.PRNGKey(2)
        pop = random_population(key, 20, n_m, GENE_HI)
        fit = jnp.zeros(20)
        new_pop, _ = evolve_population(pop, fit, key, GAParams(), lo, hi)
        assert new_pop.shape == pop.shape

    def test_乱数を進めると結果が変わる(self, spec):
        n_m = int(spec.muscle_idx.shape[0])
        lo, hi = gene_bounds(n_m, MODEL)
        pop = random_population(jax.random.PRNGKey(3), 10, n_m, GENE_HI)
        fit = jnp.zeros(10)
        gp = GAParams(pop_size=10)
        _, k1 = evolve_population(pop, fit, jax.random.PRNGKey(10), gp, lo, hi)
        _, k2 = evolve_population(pop, fit, jax.random.PRNGKey(20), gp, lo, hi)
        assert not np.allclose(np.asarray(k1), np.asarray(k2))
