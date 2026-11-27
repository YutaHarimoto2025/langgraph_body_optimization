"""実行の記録と再現メタデータ。resources/runs/<name>/ に実行単位で保存する。

記録レベル (合意):
- summary:  世代統計 + 各世代の最良遺伝子 (既定)
- full:     summary + 各世代の集団 (遺伝子・スコア・付随統計)。
              軌跡は replay で必要なときに再生成する。
"""
from __future__ import annotations

import json
import resource
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .creature import CreatureSpec
from .fitness import FitnessParams
from .ga import GAParams
from .physics import SimParams

RECORD_LEVELS = ("summary", "full")


def _git_state() -> dict[str, Any]:
    try:
        h = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.strip()
        )
        return {"commit": h, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def _device_state() -> list[dict[str, Any]]:
    devs = jax.devices()
    out = []
    for d in devs:
        st = getattr(d, "memory_stats", lambda: {})() or {}
        out.append(
            {
                "device_kind": str(d.device_kind),
                "platform": str(d.platform),
                "peak_bytes_in_use": st.get("peak_bytes_in_use"),
            }
        )
    return out


@dataclass
class RunMeta:
    """1 実行の再現に必要な情報。resources/runs/<name>/meta.json へ。"""

    command: str
    record_level: str
    git: dict[str, Any] = field(default_factory=_git_state)
    jax_version: str = ""
    devices: list[dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float | None = None
    peak_rss_mb: float | None = None
    sim_params: dict[str, Any] = field(default_factory=dict)
    fitness_params: dict[str, Any] = field(default_factory=dict)
    ga_params: dict[str, Any] = field(default_factory=dict)
    creature: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def build(
        command: str,
        record_level: str,
        sim_p: SimParams,
        fit_p: FitnessParams,
        gp: GAParams,
        creature_summary: dict[str, Any],
        elapsed: float,
    ) -> "RunMeta":
        fp = asdict(fit_p)
        sp = asdict(sim_p)
        gp_d = asdict(gp)
        return RunMeta(
            command=command,
            record_level=record_level,
            jax_version=jax.__version__,
            devices=_device_state(),
            elapsed_seconds=round(elapsed, 3),
            peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            sim_params=sp,
            fitness_params=fp,
            ga_params=gp_d,
            creature=creature_summary,
        )


def creature_from_summary(c: dict[str, Any]) -> CreatureSpec:
    """meta.json の creature 概要から CreatureSpec を復元 (replay 用)。

    旧形式 (masses/stiffness/muscle_mask を含まない) の meta.json にも対応する。
    """
    n_edges = int(c["n_edges"])
    mask_list = c.get("muscle_mask") or [True] * n_edges
    masses = c.get("masses") or [1.0] * int(c["n_nodes"])
    stiffness = c.get("stiffness") or [100.0] * n_edges
    mask = jnp.asarray(mask_list, dtype=jnp.bool_)
    return CreatureSpec(
        base_pos=jnp.asarray(c["base_pos"], dtype=jnp.float32),
        masses=jnp.asarray(masses, dtype=jnp.float32),
        edges=jnp.asarray(c["edges"], dtype=jnp.int32),
        stiffness=jnp.asarray(stiffness, dtype=jnp.float32),
        rest_lengths=jnp.asarray(c["rest_lengths"], dtype=jnp.float32),
        muscle_mask=mask,
        muscle_idx=jnp.where(mask)[0],
        name=str(c.get("name", "restored")),
    )


#: 遺伝子ブロック (振幅・周波数・位相) と、σ を出す選択群 (上位 1%・10%・全個体)。
_GENE_TYPES = (("amplitude", 0), ("frequency", 1), ("phase", 2))
_GENE_GROUPS = (("top1", 0.01), ("top10", 0.10), ("all", 1.0))


def _row_distribution(
    values: np.ndarray, mask: np.ndarray, quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
) -> dict[str, list[float | None]]:
    """世代ごとに、mask された値の分位点と標準偏差を出す。空の世代は None。"""
    keys = [f"p{int(q * 100)}" for q in quantiles] + ["std"]
    out: dict[str, list[float | None]] = {k: [] for k in keys}
    for gi in range(values.shape[0]):
        v = values[gi][mask[gi]]
        v = v[np.isfinite(v)]
        if v.size == 0:
            for k in keys:
                out[k].append(None)
            continue
        for q, k in zip(quantiles, keys):
            out[k].append(round(float(np.quantile(v, q)), 6))
        out["std"].append(round(float(v.std()), 6))
    return out


def _gene_sigma(pop: np.ndarray, scores: np.ndarray) -> dict[str, dict[str, list[float]]]:
    """遺伝子の種類ごとに、選択群×全エッジをまとめた標準偏差を世代ごとに出す。

    スコア降順で上位 1%・10%・全個体を選び、その群の振幅 (または周波数・位相) の
    全筋肉ぶんの値をまとめて np.std する。上位 1% が 1 個体のときは、その個体の
    エッジ間のばらつきになる。無効個体も順位どおり含める。
    """
    n_gen, n_pop, length = pop.shape
    n_m = length // 3
    out = {name: {group: [] for group, _ in _GENE_GROUPS} for name, _ in _GENE_TYPES}
    for gi in range(n_gen):
        order = np.argsort(scores[gi])[::-1]
        for group, frac in _GENE_GROUPS:
            k = max(1, int(np.ceil(frac * n_pop)))
            idx = order[:k]
            for name, block in _GENE_TYPES:
                vals = pop[gi][idx, block * n_m : (block + 1) * n_m]
                out[name][group].append(round(float(vals.std()), 5))
    return out


def creature_summary(spec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "n_nodes": int(spec.n_nodes),
        "n_edges": int(spec.n_edges),
        "n_muscles": int(np.asarray(spec.muscle_mask).sum()),
        "base_pos": np.asarray(spec.base_pos).round(6).tolist(),
        "edges": np.asarray(spec.edges).tolist(),
        "rest_lengths": np.asarray(spec.rest_lengths).round(6).tolist(),
        "masses": np.asarray(spec.masses).round(6).tolist(),
        "stiffness": np.asarray(spec.stiffness).round(6).tolist(),
        "muscle_mask": np.asarray(spec.muscle_mask).tolist(),
    }


def save_run(
    out_dir: Path,
    record_level: str,
    spec,
    sim_p: SimParams,
    fit_p: FitnessParams,
    gp: GAParams,
    n_generations: int,
    result: dict,
    elapsed: float,
    command: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = result["stats"]

    meta = RunMeta.build(command, record_level, sim_p, fit_p, gp, creature_summary(spec), elapsed)
    (out_dir / "meta.json").write_text(
        json.dumps(asdict(meta), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 世代統計 + 各世代の最良遺伝子 (summary 相当)
    best_idx = np.argmax(stats.score, axis=1)
    gen_arange = np.arange(n_generations)
    # 遺伝子の種類ごとの σ (選択群×全エッジ)。summary 記録でも保存時には集団がある。
    gene_sigma = _gene_sigma(
        np.asarray(result["populations"], dtype=np.float64),
        np.asarray(stats.score, dtype=np.float64),
    )
    # 集団を保存しない summary でも HTML を完全に描けるよう、分布統計も保存時に作る。
    valid = np.asarray(stats.valid, dtype=bool)
    score_dist = _row_distribution(np.asarray(stats.score, dtype=np.float64), valid)
    distance_dist = _row_distribution(np.asarray(stats.distance, dtype=np.float64), valid)
    energy_dist = _row_distribution(
        np.asarray(stats.drive_energy, dtype=np.float64), np.ones_like(valid), (0.5, 0.9)
    )
    gen_summary = {
        "n_generations": n_generations,
        "per_generation": {
            "best_score": stats.score.max(axis=1).round(6).tolist(),
            "mean_score": stats.score.mean(axis=1).round(6).tolist(),
            "valid_rate": stats.valid.mean(axis=1).round(4).tolist(),
            "best_distance": stats.distance.max(axis=1).round(6).tolist(),
            "mean_distance": stats.distance.mean(axis=1).round(6).tolist(),
            "n_valid": valid.sum(axis=1).astype(int).tolist(),
            "score_dist": score_dist,
            "distance_dist": distance_dist,
            "energy_dist": energy_dist,
            "gene_sigma": gene_sigma,
            "best_energy": stats.drive_energy[gen_arange, best_idx].round(4).tolist(),
            "mean_energy": stats.drive_energy.mean(axis=1).round(4).tolist(),
            "best_gene": [
                result["populations"][g][best_idx[g]].round(6).tolist()
                for g in range(n_generations)
            ],
        },
    }
    (out_dir / "stats.json").write_text(
        json.dumps(gen_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if record_level == "full":
        np.savez_compressed(
            out_dir / "population.npz",
            genes=result["populations"].astype(np.float32),
            scores=stats.score.astype(np.float32),
            valid=stats.valid,
            distance=stats.distance.astype(np.float32),
            drive_energy=stats.drive_energy.astype(np.float32),
            max_penetration=stats.max_penetration.astype(np.float32),
            contact_rate=stats.contact_rate.astype(np.float32),
        )
    return out_dir


def save_trajectory(
    out_dir: Path, traj, gene: np.ndarray, label: str
) -> Path:
    """replay 用の生ロールアウト。数値は 4 桁に丸めて軽量化。"""
    path = out_dir / f"trajectory_{label}.json"
    payload = {
        "label": label,
        "gene": gene.round(6).tolist(),
        "times": np.asarray(traj.times).round(4).tolist(),
        "pos": np.asarray(traj.pos).round(4).tolist(),
        "vel": np.asarray(traj.vel).round(4).tolist(),
        "spring_force": np.asarray(traj.spring_force).round(3).tolist(),
        "contact_force": np.asarray(traj.contact_force).round(3).tolist(),
        "rest_lengths": np.asarray(traj.rest_lengths).round(4).tolist(),
        "drive_power": np.asarray(traj.drive_power).round(3).tolist(),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def load_population(run_dir: Path) -> dict:
    with np.load(run_dir / "population.npz") as z:
        return {k: z[k] for k in z.files}
