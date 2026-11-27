"""純粋 GA (固定戦略)。世代ループ全体を jit + lax.scan で回し、CPU 同期は最後だけ。

構成は trajectory-solver-generic の BRKGA (src/optim/brkga.py) を参考に、
「集団評価は注入された関数、集団更新は JAX 演算」の分離を踏襨する。
ただしエリート偏重交叉ではなく、合意どおりトーナメント選択・一様交叉・
ガウス突然変異・エリート継承とする (first goal は固定 GA)。
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .creature import CreatureSpec
from .fitness import EvalStats, FitnessParams, evaluate_population
from .models import FREQ_HI, FREQ_LO, get_model
from .physics import SimParams

#: 遺伝子の意味づけ: [0, E_m) 振幅 (モデル依存)、[E_m, 2E_m) 周波数 [Hz]、[2E_m, 3E_m) 位相 [rad]。
PHASE_HI = 2.0 * jnp.pi


@dataclass(frozen=True)
class GAParams:
    pop_size: int = 20
    n_elite: int = 2
    tournament_k: int = 3
    crossover_rate: float = 0.8
    mutation_rate: float = 0.10  # 遺伝子ごとの突然変異確率
    mutation_sigma_scale: float = 0.05  # 範囲に対する sigma の割合
    immigration_rate: float = 0.20  # 次世代のうちランダム個体で置き換える割合
    seed: int = 0


def gene_bounds(n_muscle: int, muscle_model: str) -> tuple[jax.Array, jax.Array]:
    """遺伝子の下限・上限 [3*E_m]。振幅の上限はモデル (gene_hi) に従う。"""
    lo = jnp.concatenate(
        [
            jnp.zeros(n_muscle),
            jnp.full(n_muscle, FREQ_LO),
            jnp.zeros(n_muscle),
        ]
    ).astype(jnp.float32)
    hi = jnp.concatenate(
        [
            jnp.full(n_muscle, get_model(muscle_model).gene_hi),
            jnp.full(n_muscle, FREQ_HI),
            jnp.full(n_muscle, PHASE_HI),
        ]
    ).astype(jnp.float32)
    return lo, hi


def random_population(key: jax.Array, n: int, n_muscle: int, amp_hi: float) -> jax.Array:
    """一様サンプリングで初期集団 [n, 3*E_m]。振幅 [0, amp_hi]、周波数 [FREQ_LO, FREQ_HI]、位相 [0, 2π)。"""
    k1, k2, k3 = jax.random.split(key, 3)
    amps = jax.random.uniform(k1, (n, n_muscle), minval=0.0, maxval=amp_hi)
    freqs = jax.random.uniform(k2, (n, n_muscle), minval=FREQ_LO, maxval=FREQ_HI)
    phases = jax.random.uniform(k3, (n, n_muscle), minval=0.0, maxval=PHASE_HI)
    return jnp.concatenate([amps, freqs, phases], axis=1)


def _tournament_parent(
    key: jax.Array, fit: jax.Array, pop: jax.Array, k: int
) -> jax.Array:
    """適応度最大の個体を 1 つ返すトーナメント選択。"""
    n = fit.shape[0]
    idx = jax.random.randint(key, (k,), 0, n)
    best = idx[jnp.argmax(fit[idx])]
    return pop[best]


def _clip_gene(genes: jax.Array, lo: jax.Array, hi: jax.Array) -> jax.Array:
    """振幅・周波数は範囲内にクリップし、位相は周期折返しする。"""
    n_m = lo.shape[0] // 3
    amp = jnp.clip(genes[..., :n_m], lo[:n_m], hi[:n_m])
    freq = jnp.clip(genes[..., n_m : 2 * n_m], lo[n_m : 2 * n_m], hi[n_m : 2 * n_m])
    phase = jnp.mod(genes[..., 2 * n_m :], PHASE_HI)
    return jnp.concatenate([amp, freq, phase], axis=-1)


def inject_worst(
    pop: jax.Array,  # [P, L]
    fit: jax.Array,  # [P] 現世代の適応度
    key: jax.Array,
    n_imm: int,
    amp_hi: float,
) -> jax.Array:
    """適応度の下位 n_imm 個体をランダム個体で置き換える (移民)。

    置換後の個体は次世代で評価される。この世代の選択には使わない前提で、
    適応度は置換前の値 (下位) のまま残す。
    """
    n_pop, L = pop.shape
    n_imm = min(n_imm, n_pop)
    if n_imm <= 0:
        return pop
    idx = jnp.argsort(fit)[:n_imm]
    imm = random_population(key, n_imm, L // 3, amp_hi)
    return pop.at[idx].set(imm)


def evolve_population(
    pop: jax.Array,  # [P, L]
    fit: jax.Array,  # [P]
    key: jax.Array,
    gp: GAParams,
    lo: jax.Array,
    hi: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """1 世代の集団更新。エリート継承 + トーナメント + 一様交叉 + ガウス変異。"""
    n_pop, L = pop.shape
    sigma = (hi - lo) * gp.mutation_sigma_scale

    order = jnp.argsort(fit)[::-1]  # 最大化: 降順
    elites = pop[order[: gp.n_elite]]

    n_off = n_pop - gp.n_elite
    # 0: 次世代用, 1..2n_off: 親選択, 2n_off+1: 交叉, 2n_off+2: 変異, 2n_off+3: ノイズ
    keys = jax.random.split(key, 2 * n_off + 4)
    k_par = keys[1 : 1 + 2 * n_off]  # (2*n_off, 2)

    pick = lambda kk: _tournament_parent(kk, fit, pop, gp.tournament_k)
    p1 = jax.vmap(pick)(k_par[:n_off])
    p2 = jax.vmap(pick)(k_par[n_off:])
    # 一様交叉: 遺伝子単位のスイッチ確率
    cross_mask = jax.random.bernoulli(
        keys[1 + 2 * n_off], p=gp.crossover_rate, shape=(n_off, L)
    )
    child = jnp.where(cross_mask, p1, p2)

    # ガウス突然変異 (遺伝子ごと)
    mut_mask = jax.random.bernoulli(
        keys[2 + 2 * n_off], p=gp.mutation_rate, shape=(n_off, L)
    )
    noise = jax.random.normal(keys[3 + 2 * n_off], (n_off, L)) * sigma[None, :]
    child = _clip_gene(child + mut_mask * noise, lo, hi)

    new_pop = jnp.concatenate([elites, child], axis=0)
    return new_pop, keys[0]


def run_ga(
    spec: CreatureSpec,
    sim_p: SimParams,
    fit_p: FitnessParams,
    gp: GAParams,
    n_generations: int,
) -> dict:
    """固定 GA の本体。世代ループ全体を 1 つの jit に載せ、最後にだけ同期する。

    戻り値: stats (numpy), final_pop, history (每世代の集団・適応度)。
    """
    n_m = int(spec.muscle_idx.shape[0])
    model = get_model(sim_p.muscle_model)
    lo, hi = gene_bounds(n_m, sim_p.muscle_model)

    key = jax.random.PRNGKey(gp.seed)
    key, k_init = jax.random.split(key)
    pop0 = random_population(k_init, gp.pop_size, n_m, model.gene_hi)

    n_imm = int(round(gp.pop_size * gp.immigration_rate))

    def gen_loop(carry, _):
        pop, key = carry
        stats: EvalStats = evaluate_population(pop, spec, sim_p, fit_p)
        key, k_imm = jax.random.split(key)
        pop_inj = inject_worst(pop, stats.score, k_imm, n_imm, model.gene_hi)
        pop_next, key_next = evolve_population(pop_inj, stats.score, key, gp, lo, hi)
        return (pop_next, key_next), (stats, pop)

    (final_pop, _), (stats_all, pops_all) = jax.lax.scan(
        gen_loop, (pop0, key), None, length=n_generations
    )
    # ここで初めて CPU へ転送 (同期はこの1回)
    out = jax.tree.map(np.asarray, stats_all)
    return dict(stats=out, final_pop=np.asarray(final_pop), populations=np.asarray(pops_all))
