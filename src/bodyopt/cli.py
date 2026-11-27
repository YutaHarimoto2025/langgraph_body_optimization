"""CLI。evolve (描画なし・同期は最後だけ) と replay (保存個体の再ロールアウト)。

使い方:
  uv run python -m bodyopt evolve --generations 10 --seed 0 \
      --record summary --out resources/runs/first_try
  uv run python -m bodyopt replay --run resources/runs/first_try \
      --generation 9 --rank 0
  uv run python -m bodyopt viewer --port 8765
  uv run python -m bodyopt stats-html --run resources/runs/first_try
  uv run python -m bodyopt chaos --run resources/runs/first_try --repeats 20
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any, TypeVar

import jax
import numpy as np

from .config import DEFAULT_CONFIG, load_config
from .fitness import com_trajectory, rollout_full
from .ga import run_ga
from .io import (
    RECORD_LEVELS,
    creature_from_summary,
    load_population,
    save_run,
    save_trajectory,
)
from .physics import RolloutResult, SimParams
from .stats_html import write_report


T = TypeVar("T")


def _override(obj: T, **values: object) -> T:
    """None の引数は設定ファイルの値のままにする。"""
    return dataclasses.replace(obj, **{k: v for k, v in values.items() if v is not None})  # pyright: ignore[reportArgumentType]


def _git_dirty() -> bool:
    """作業ツリーに未 commit の変更があるか。run の再現性チェック用。"""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10
        )
        return bool(out.stdout.strip())
    except Exception:
        return False


def _confirm_full(args: argparse.Namespace, spec, gp) -> None:
    """full 記録の推定ディスク量が 1 GB を超える場合は対話で確認する。"""
    if args.record != "full":
        return
    dim = 3 * spec.n_muscles + 6  # 遺伝子 + 付随統計 (scores/valid/distance/...)
    est_gb = args.generations * gp.pop_size * dim * 4 / 1e9
    if est_gb <= 1.0 or args.yes:
        return
    msg = f"[evolve] full 記録の推定ディスク使用量は約 {est_gb:.1f} GB です。続行しますか? [y/N] "
    if not sys.stdin.isatty():
        raise SystemExit(msg + "(非対話環境では --yes を付けてください)")
    if input(msg).strip().lower() not in ("y", "yes"):
        raise SystemExit("中断しました")


def cmd_evolve(args: argparse.Namespace) -> None:
    args.record = "full" if args.record == "population" else args.record
    if not args.allow_dirty and _git_dirty():
        raise SystemExit(
            "[evolve] 作業ツリーが dirty です。run 前に commit するか --allow-dirty を付けてください。"
        )
    cfg = load_config(args.config)
    spec = cfg.build_creature(**({"n_cols": args.n_cols} if args.n_cols is not None else {}))
    sim_p = _override(
        cfg.sim, dt=args.dt, episode_seconds=args.episode, integrator=args.integrator
    )
    fit_p = _override(
        cfg.fitness,
        max_penetration=args.max_penetration,
        energy_budget=args.energy_budget,
    )
    gp = _override(cfg.ga, pop_size=args.pop_size, seed=args.seed)
    _confirm_full(args, spec, gp)

    t0 = time.perf_counter()
    result = run_ga(spec, sim_p, fit_p, gp, args.generations)
    elapsed = time.perf_counter() - t0  # run_ga 内の tree.map で同期済み

    out_dir = Path(args.out)
    cmd = "evolve " + " ".join(
        f"{k}={v}"
        for k, v in sorted(vars(args).items())
        if v is not None and k not in ("func", "record")
    )
    save_run(out_dir, args.record, spec, sim_p, fit_p, gp, args.generations, result, elapsed, cmd)

    stats = result["stats"]
    print(
        f"[evolve] generations={args.generations} "
        f"best_score={float(stats.score.max()):.4f} "
        f"valid_rate_last={float(stats.valid[-1].mean()):.2f} "
        f"elapsed={elapsed:.2f}s -> {out_dir}"
    )


def _check_range(name: str, value: int, size: int) -> None:
    if not 0 <= value < size:
        raise ValueError(f"{name} {value} は範囲外 (0..{size - 1})")


@dataclasses.dataclass(frozen=True)
class ReplayOutcome:
    """replay_batch の 1 個体分の結果。error が None なら path に書き出し済み。"""

    generation: int
    rank: int
    path: Path | None = None
    score: float = 0.0
    distance: float = 0.0
    error: str | None = None


def _replay_source(run_dir: Path) -> tuple[np.ndarray, np.ndarray, bool]:
    """(genes [G, P, L], scores [G, P], summary 記録か) を返す。"""
    if (run_dir / "population.npz").exists():
        pop = load_population(run_dir)
        return pop["genes"], pop["scores"], False
    # summary 記録は各世代の最良個体しか残さないので P=1 として扱う
    per = json.loads((run_dir / "stats.json").read_text(encoding="utf-8"))["per_generation"]
    genes = np.asarray(per["best_gene"], dtype=np.float32)[:, None, :]
    scores = np.asarray(per["best_score"], dtype=np.float32)[:, None]
    return genes, scores, True


# run ごとの (spec, sim_p, jit 済みバッチ rollout)。JAX のコンパイル結果を使い回す。
# ponytail: run ディレクトリ単位のキャッシュ。同じ run を上書きする運用なら再起動が要る。
_REPLAY_CONTEXT: dict[Path, tuple[Any, Any, Any]] = {}


def _replay_context(run_dir: Path) -> tuple[Any, Any, Any]:
    """meta.json から身体と物理を復元し、バッチ rollout (jit + vmap) を返す。"""
    key = run_dir.resolve()
    cached = _REPLAY_CONTEXT.get(key)
    if cached is not None:
        return cached
    with open(run_dir / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    # 身体は保存された定義から復元する (実行時の --n-cols に依存しない)
    spec = creature_from_summary(meta["creature"])
    sim_p = SimParams(**{k: v for k, v in meta["sim_params"].items() if k in SimParams.__dataclass_fields__})

    def one(gene: jax.Array) -> tuple[Any, ...]:
        r = rollout_full(gene, spec, sim_p)
        # RolloutResult は pytree 登録していないので tuple に開いて vmap に渡す
        return (r.pos, r.vel, r.times, r.spring_force, r.contact_force, r.drive_power, r.rest_lengths)

    context = (spec, sim_p, jax.jit(jax.vmap(one)))
    _REPLAY_CONTEXT[key] = context
    return context


def replay_batch(
    run_dir: Path,
    targets: list[tuple[int, int]],
    out_dir: Path | None = None,
    labels: list[str] | None = None,
) -> list[ReplayOutcome]:
    """複数個体を 1 回の vmap rollout でまとめて書き出す。

    targets の rank は 0 始まり。範囲外の個体は error を入れて他は続行する。
    labels を省くと trajectory_gen<g>_rank<r>.json になる。
    """
    genes, scores, summary_only = _replay_source(run_dir)
    spec, _sim_p, batch_rollout = _replay_context(run_dir)
    n_gens, n_ranks = scores.shape
    # 旧形式 (2 成分/筋肉) の遺伝子は現在の物理モデルで解釈できない。JAX の前に弾く。
    gene_error: str | None = None
    if genes.shape[2] != 3 * spec.n_muscles:
        gene_error = (
            f"遺伝子長 {genes.shape[2]} が現在の物理 (3 × 筋肉数 = {3 * spec.n_muscles}) と"
            "一致しません。旧形式の run は当時のコードで replay してください。"
        )
    outcomes: list[ReplayOutcome | None] = [None] * len(targets)
    pending: list[int] = []
    picked_genes: list[np.ndarray] = []
    picked_scores: list[float] = []
    for i, (generation, rank) in enumerate(targets):
        if gene_error is not None:
            outcomes[i] = ReplayOutcome(generation, rank, error=gene_error)
            continue
        try:
            _check_range("gen", generation, n_gens)
            if summary_only and rank != 0:
                raise ValueError(
                    f"{run_dir.name} は summary 記録 (population.npz なし) なので "
                    "最良個体 (rank 1) のみ再生できます"
                )
            if not 0 <= rank < n_ranks:
                raise ValueError(f"rank {rank + 1} は範囲外 (1..{n_ranks})")
        except ValueError as e:
            outcomes[i] = ReplayOutcome(generation, rank, error=str(e))
            continue
        order = np.argsort(scores[generation])[::-1]
        picked_genes.append(genes[generation][order[rank]])
        picked_scores.append(float(scores[generation][order[rank]]))
        pending.append(i)

    if pending:
        rolled = batch_rollout(jax.numpy.asarray(np.stack(picked_genes)))
        dest = run_dir if out_dir is None else out_dir
        dest.mkdir(parents=True, exist_ok=True)
        for k, i in enumerate(pending):
            generation, rank = targets[i]
            traj = RolloutResult(*jax.tree.map(lambda x: x[k], rolled))
            com = com_trajectory(traj.pos, spec)
            label = labels[i] if labels else f"gen{generation}_rank{rank}"
            out = save_trajectory(dest, traj, np.asarray(picked_genes[k]), label)
            outcomes[i] = ReplayOutcome(
                generation, rank, path=out, score=picked_scores[k],
                distance=float(com[-1, 0] - com[0, 0]),
            )
    return [o for o in outcomes if o is not None]


def replay_individual(
    run_dir: Path,
    generation: int,
    rank: int,
    out_dir: Path | None = None,
    label: str | None = None,
) -> tuple[Path, float, float]:
    """1 個体を再ロールアウトして軌跡 JSON を書き出す (CLI 用)。

    rank は 0 始まり (0 = 最良)。戻り値は (出力パス, 記録スコア, 重心の移動距離)。
    """
    outcome = replay_batch(run_dir, [(generation, rank)], out_dir, [label] if label else None)[0]
    if outcome.path is None:
        raise ValueError(outcome.error)
    return outcome.path, outcome.score, outcome.distance


def cmd_replay(args: argparse.Namespace) -> None:
    try:
        out, score, distance = replay_individual(
            Path(args.run), args.generation, args.rank,
            Path(args.out) if args.out else None,
        )
    except ValueError as e:
        raise SystemExit(f"[replay] {e}") from e
    print(f"[replay] {out.stem} score={score:.4f} distance={distance:.4f} -> {out}")


@dataclasses.dataclass(frozen=True)
class ChaosOutcome:
    """最良個体を同じ条件で複数回シミュレーションした結果のばらつき。"""

    generation: int
    rank: int
    repeats: int
    mean_distance: float
    std_distance: float
    spread_distance: float
    relative_spread: float
    recorded_score: float
    tolerance: float
    passed: bool


def chaos_check(
    run_dir: Path,
    generation: int | None = None,
    rank: int = 0,
    repeats: int = 20,
) -> ChaosOutcome:
    """保存個体を同じ条件で repeats 回まとめて走らせ、距離のばらつきを測る。

    カオスが強い設定では、同一遺伝子でも接触の原子加算順の差が増幅されて距離が
    ばらつく。ばらつきが大きい run は、設定間の比較を平均・分布で行う必要がある。
    """
    if repeats < 2:
        raise ValueError("--repeats は 2 以上にしてください")
    genes, scores, summary_only = _replay_source(run_dir)
    n_gens, n_ranks = scores.shape
    gen = n_gens - 1 if generation is None else generation
    _check_range("gen", gen, n_gens)
    if summary_only and rank != 0:
        raise ValueError("summary 記録は最良個体 (rank 0) のみ測れます")
    if not 0 <= rank < n_ranks:
        raise ValueError(f"rank {rank + 1} は範囲外 (1..{n_ranks})")
    spec, _sim_p, batch_rollout = _replay_context(run_dir)
    if genes.shape[2] != 3 * spec.n_muscles:
        raise ValueError(
            f"遺伝子長 {genes.shape[2]} が現在の物理 (3 × 筋肉数 = {3 * spec.n_muscles}) と"
            "一致しません。旧形式の run は当時のコードで測ってください。"
        )
    order = np.argsort(scores[gen])[::-1]
    gene = genes[gen][order[rank]]
    batch = np.tile(gene, (repeats, 1)).astype(np.float32)
    rolled = batch_rollout(jax.numpy.asarray(batch))
    pos = np.asarray(rolled[0])  # [repeats, T, N, 2]
    masses = np.asarray(spec.masses, dtype=np.float64)
    com_x = pos[..., 0] @ masses / masses.sum()
    dist = com_x[:, -1] - com_x[:, 0]
    mean = float(dist.mean())
    spread = float(dist.max() - dist.min())
    tolerance = max(0.005, 0.01 * abs(mean))
    return ChaosOutcome(
        generation=gen,
        rank=rank,
        repeats=repeats,
        mean_distance=mean,
        std_distance=float(dist.std()),
        spread_distance=spread,
        relative_spread=spread / max(abs(mean), 1e-9),
        recorded_score=float(scores[gen][order[rank]]),
        tolerance=tolerance,
        passed=spread <= tolerance,
    )


def cmd_chaos(args: argparse.Namespace) -> None:
    try:
        outcome = chaos_check(Path(args.run), args.generation, args.rank, args.repeats)
    except ValueError as e:
        raise SystemExit(f"[chaos] {e}") from e
    if args.json:
        print(json.dumps(dataclasses.asdict(outcome), ensure_ascii=False, indent=2))
        return
    verdict = "OK" if outcome.passed else "注意: ばらつき大"
    print(f"[chaos] gen={outcome.generation} rank={outcome.rank + 1} repeats={outcome.repeats}")
    print(
        f"  距離 平均={outcome.mean_distance:.4f} m 標準偏差={outcome.std_distance:.4f} "
        f"幅={outcome.spread_distance:.4f} m ({outcome.relative_spread * 100:.2f}%)"
    )
    print(
        f"  記録スコア={outcome.recorded_score:.4f} "
        f"再計算との差={outcome.mean_distance - outcome.recorded_score:+.4f} m"
    )
    print(
        f"  判定={verdict} (目安: 幅 <= max(5 mm, 平均の 1%) = {outcome.tolerance:.4f} m)"
    )


def cmd_stats_html(args: argparse.Namespace) -> None:
    try:
        out = write_report(args.run, args.out)
    except FileNotFoundError as e:
        raise SystemExit(f"[stats-html] {e}") from e
    print(f"[stats-html] -> {out}")


def cmd_viewer(args: argparse.Namespace) -> None:
    from .viewer_server import serve

    serve(host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="bodyopt", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    # 既定は params/*.json から読む。CLI で指定した値だけ上書きする。
    common.add_argument("--n-cols", type=int, default=None, help="ラダーの列数 (設定ファイルを上書き)")
    common.add_argument("--dt", type=float, default=None, help="刻み [s] (設定ファイルを上書き)")
    common.add_argument("--episode", type=float, default=None, help="試行時間 [s] (設定ファイルを上書き)")
    common.add_argument("--integrator", default=None,
                        choices=["semi_implicit_euler", "rk4"])

    ev = sub.add_parser("evolve", parents=[common], help="GA を回して resources/runs に保存")
    ev.add_argument("--config", default=DEFAULT_CONFIG, help="パラメータファイル (params/*.json)")
    ev.add_argument("--generations", type=int, default=10)
    ev.add_argument("--pop-size", type=int, default=None, help="設定ファイルを上書き")
    ev.add_argument("--seed", type=int, default=None, help="設定ファイルを上書き")
    ev.add_argument(
        "--record", default="summary", choices=[*RECORD_LEVELS, "population"],
        help="summary (既定) は集団を保存しない。full は全個体を保存",
    )
    ev.add_argument("--yes", action="store_true", help="full 記録の容量確認を省略")
    ev.add_argument("--allow-dirty", action="store_true", help="dirty でも実行する (非推奨)")
    ev.add_argument("--max-penetration", type=float, default=None)
    ev.add_argument("--energy-budget", type=float, default=None)
    ev.add_argument("--out", default="resources/runs/run")
    ev.set_defaults(func=cmd_evolve)

    rp = sub.add_parser("replay", parents=[common], help="保存個体を再ロールアウトして軌跡 JSON を出力")
    rp.add_argument("--run", required=True, help="resources/runs/<name> のパス")
    rp.add_argument("--generation", type=int, required=True)
    rp.add_argument("--rank", type=int, default=0, help="世代内の順位 (0 = 最良)")
    rp.add_argument("--out", default=None, help="軌跡 JSON の出力先 (既定は --run のディレクトリ)")
    rp.set_defaults(func=cmd_replay)

    ch = sub.add_parser("chaos", help="最良個体を複数回シミュレーションして距離のばらつきを測る")
    ch.add_argument("--run", required=True, help="resources/runs/<name> のパス")
    ch.add_argument("--generation", type=int, default=None, help="世代 (既定は最終世代)")
    ch.add_argument("--rank", type=int, default=0, help="世代内の順位 (0 = 最良)")
    ch.add_argument("--repeats", type=int, default=20, help="同じ条件で走らせる回数")
    ch.add_argument("--json", action="store_true", help="結果を JSON で出す")
    ch.set_defaults(func=cmd_chaos)

    sp = sub.add_parser("stats-html", help="stats.json から世代統計の HTML を生成する")
    sp.add_argument("--run", required=True, help="run ディレクトリまたは stats.json")
    sp.add_argument("--out", default=None, help="出力先 (既定は stats.json と同じディレクトリの stats.html)")
    sp.set_defaults(func=cmd_stats_html)

    vp = sub.add_parser("viewer", help="軌跡ビューアと replay API をローカル配信する")
    vp.add_argument("--host", default="127.0.0.1")
    vp.add_argument("--port", type=int, default=8765)
    vp.set_defaults(func=cmd_viewer)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
