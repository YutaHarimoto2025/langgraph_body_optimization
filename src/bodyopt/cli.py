"""CLI。evolve (描画なし・同期は最後だけ) と replay (保存個体の再ロールアウト)。

使い方:
  uv run python -m bodyopt evolve --generations 10 --seed 0 \
      --record population --out resources/runs/first_try
  uv run python -m bodyopt replay --run resources/runs/first_try \
      --generation 9 --rank 0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

from .creature import CreatureSpec, triangulated_ladder
from .fitness import FitnessParams, rollout_full
from .ga import GAParams, run_ga
from .io import (
    RECORD_LEVELS,
    creature_from_summary,
    load_population,
    save_run,
    save_trajectory,
)
from .physics import SimParams


def _creature(args: argparse.Namespace) -> CreatureSpec:
    return triangulated_ladder(n_cols=args.n_cols)


def cmd_evolve(args: argparse.Namespace) -> None:
    spec = _creature(args)
    sim_p = SimParams(
        dt=args.dt,
        episode_seconds=args.episode,
        integrator=args.integrator,
    )
    fit_p = FitnessParams(
        max_penetration=args.max_penetration,
        energy_budget=args.energy_budget,
    )
    gp = GAParams(pop_size=args.pop_size, seed=args.seed)

    t0 = time.perf_counter()
    result = run_ga(spec, sim_p, fit_p, gp, args.generations)
    elapsed = time.perf_counter() - t0  # run_ga 内の tree.map で同期済み

    out_dir = Path(args.out)
    cmd = "evolve " + " ".join(f"{k}={v}" for k, v in sorted(vars(args).items()) if k not in ("func", "out"))
    save_run(out_dir, args.record, spec, sim_p, fit_p, gp, args.generations, result, elapsed, cmd)

    if args.record == "full":
        # 最良個体 (累積スコア最大) の完全ロールアウトを生データとして保存
        scores = result["stats"].score  # [G, P]
        g_best, p_best = np.unravel_index(np.argmax(scores), scores.shape)
        best_gene = result["populations"][g_best][p_best]
        traj = rollout_full(jax.numpy.asarray(best_gene), spec, sim_p)
        save_trajectory(out_dir, traj, best_gene, f"gen{g_best}_rank0")

    stats = result["stats"]
    print(
        f"[evolve] generations={args.generations} "
        f"best_score={float(stats.score.max()):.4f} "
        f"valid_rate_last={float(stats.valid[-1].mean()):.2f} "
        f"elapsed={elapsed:.2f}s -> {out_dir}"
    )


def cmd_replay(args: argparse.Namespace) -> None:
    run_dir = Path(args.run)
    with open(run_dir / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    # 身体は保存された定義から復元する (実行時の --n-cols に依存しない)
    spec = creature_from_summary(meta["creature"])
    sim_p = SimParams(**{k: v for k, v in meta["sim_params"].items() if k in SimParams.__dataclass_fields__})

    pop = load_population(run_dir)
    genes = pop["genes"]
    scores = pop["scores"]
    g = args.generation
    order = np.argsort(scores[g])[::-1]
    idx = order[args.rank]
    gene = genes[g][idx]

    traj = rollout_full(jax.numpy.asarray(gene), spec, sim_p)
    label = f"gen{g}_rank{args.rank}"
    out = save_trajectory(run_dir, traj, np.asarray(gene), label)
    print(
        f"[replay] {label} score={float(scores[g][idx]):.4f} "
        f"distance={float(pop['distance'][g][idx]):.4f} -> {out}"
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="bodyopt", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--n-cols", type=int, default=3, help="ラダーの列数 (既定 3 = 6質点)")
    common.add_argument("--dt", type=float, default=SimParams.dt)
    common.add_argument("--episode", type=float, default=SimParams.episode_seconds)
    common.add_argument("--integrator", default="semi_implicit_euler",
                        choices=["semi_implicit_euler", "rk4"])

    ev = sub.add_parser("evolve", parents=[common], help="GA を回して resources/runs に保存")
    ev.add_argument("--generations", type=int, default=10)
    ev.add_argument("--pop-size", type=int, default=20)
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--record", default="population", choices=RECORD_LEVELS)
    ev.add_argument("--max-penetration", type=float, default=FitnessParams.max_penetration)
    ev.add_argument("--energy-budget", type=float, default=None)
    ev.add_argument("--out", default="resources/runs/run")
    ev.set_defaults(func=cmd_evolve)

    rp = sub.add_parser("replay", parents=[common], help="保存個体を再ロールアウトして軌跡 JSON を出力")
    rp.add_argument("--run", required=True, help="resources/runs/<name> のパス")
    rp.add_argument("--generation", type=int, required=True)
    rp.add_argument("--rank", type=int, default=0, help="世代内の順位 (0 = 最良)")
    rp.set_defaults(func=cmd_replay)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
