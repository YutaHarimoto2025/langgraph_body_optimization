"""静的テスト: pyright の警告・エラーがないことを pytest で担保する。

エディタで出る未使用 import・未定義変数などの波線は、ここで検出して
毎回の静的テストで落とす。pyright の実行に数十秒かかるが、
first goal の規模で src + tests 全体を対象にする。
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYRIGHT_TIMEOUT_SECONDS = 300


def _pyright_command() -> str:
    exe = shutil.which("pyright")
    if exe:
        return exe
    # venv 内の同じディレクトリにある pyright (uv run 経由)
    candidate = Path(sys.executable).parent / "pyright"
    if candidate.exists():
        return str(candidate)
    return "pyright"


def test_pyright_zero_errors_and_warnings():
    result = subprocess.run(
        [_pyright_command(), "--outputjson"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=PYRIGHT_TIMEOUT_SECONDS,
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise AssertionError(
            "pyright の出力を解析できませんでした。\nstdout: "
            + result.stdout[-2000:]
            + "\nstderr: "
            + result.stderr[-2000:]
        )
    summary = payload["summary"]
    diagnostics = [
        d for d in payload.get("generalDiagnostics", []) if d["severity"] in ("error", "warning")
    ]
    lines = [
        f"{d['file']}:{d['range']['start']['line'] + 1} [{d['severity']}] "
        f"{d.get('rule', '')} {d['message']}"
        for d in diagnostics
    ]
    assert not lines, "pyright の警告があります:\n" + "\n".join(lines[:20])
    assert summary["errorCount"] == 0, f"pyright errors: {summary}"
