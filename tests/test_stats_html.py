"""stats.html の生成。stats.json だけで描画し、判断ログを重ねられることを確認する。"""
from __future__ import annotations

import json
from pathlib import Path

from bodyopt.stats_html import write_report


def _write(tmp_path: Path, per: dict, decisions: list[dict] | None = None) -> Path:
    stats = {"n_generations": len(per["best_score"]), "per_generation": per}
    (tmp_path / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
    if decisions is not None:
        (tmp_path / "decisions.json").write_text(json.dumps(decisions), encoding="utf-8")
    return tmp_path


def _sigma(points: int) -> dict:
    return {
        name: {group: [1.0 * (i + 1) for i in range(points)] for group in ("top1", "top10", "all")}
        for name in ("amplitude", "frequency", "phase")
    }


def test_遺伝子種類ごとのσと判断ログを描く(tmp_path: Path) -> None:
    per = {
        "best_score": [0.1, 0.2, 0.3],
        "mean_score": [0.0, 0.1, 0.2],
        "valid_rate": [1.0, 1.0, 0.9],
        "gene_sigma": _sigma(3),
        "score_dist": {"p10": [0.0, 0.1, 0.2], "p90": [0.2, 0.3, 0.4]},
        "best_energy": [10.0, 20.0, 30.0],
        "mean_energy": [9.0, 18.0, 27.0],
    }
    decisions = [
        {"generation": 1, "label": "突然変異率を上げる", "detail": "停滞"},
        {"generation": 0, "end_generation": 2, "label": "局面 A"},
    ]
    html = write_report(_write(tmp_path, per, decisions)).read_text(encoding="utf-8")
    assert html.count('stroke-dasharray="5 4"') == 2
    assert "突然変異率を上げる" in html
    for label in ("振幅のσ", "周波数のσ", "位相のσ"):
        assert label in html
    assert "上位1%" in html and "上位10%" in html and "全個体" in html
    assert "p10–p90" in html


def test_旧_stats_でも描ける(tmp_path: Path) -> None:
    per = {
        "best_score": [0.1, 0.2],
        "mean_score": [0.0, 0.1],
        "valid_rate": [1.0, 1.0],
    }
    html = write_report(_write(tmp_path, per)).read_text(encoding="utf-8")
    assert "<svg" in html
    assert "振幅のσ" not in html
    assert "遺伝子σが stats.json に無い" in html
