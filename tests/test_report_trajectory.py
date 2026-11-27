"""報告に埋め込んだ軌跡が、保存済みの実験データから再生成できることを確認する。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_report_matches_recorded_trajectory(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    report = root / "docs/report/2026-11-23_first_ga_run.html"
    copy = tmp_path / report.name
    original = report.read_bytes()
    copy.write_bytes(original)
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/embed_report_trajectory.py"),
            str(root / "resources/runs/first_try/trajectory_gen9_rank0.json"),
            str(copy),
        ],
        check=True,
    )
    assert copy.read_bytes() == original
