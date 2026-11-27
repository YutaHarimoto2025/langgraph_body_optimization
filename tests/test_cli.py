"""CLI の evolve → replay 往復。summary 記録は最良個体 (rank 0) のみ再生できる。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bodyopt.cli import build_parser


def _run(argv: list[str]) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


def test_summary_replays_only_rank0(tmp_path: Path) -> None:
    out = tmp_path / "run"
    _run(["evolve", "--generations", "2", "--pop-size", "4", "--seed", "0",
          "--episode", "0.5", "--record", "summary", "--out", str(out), "--allow-dirty"])
    assert not (out / "population.npz").exists()
    stats = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    sigma = stats["per_generation"]["gene_sigma"]
    assert len(sigma["amplitude"]["all"]) == 2
    assert set(sigma["frequency"]) == {"top1", "top10", "all"}
    assert len(stats["per_generation"]["score_dist"]["p90"]) == 2

    _run(["replay", "--run", str(out), "--generation", "1"])
    assert (out / "trajectory_gen1_rank0.json").exists()

    with pytest.raises(SystemExit):
        _run(["replay", "--run", str(out), "--generation", "1", "--rank", "1"])

    _run(["chaos", "--run", str(out), "--repeats", "2"])
    _run(["stats-html", "--run", str(out)])
    html = (out / "stats.html").read_text(encoding="utf-8")
    assert "遺伝子σが stats.json に無い" not in html
    for label in ("振幅のσ", "周波数のσ", "位相のσ"):
        assert label in html
