"""報告 HTML に、軌跡の再生に必要なデータだけを埋め込む。"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


_DATA = re.compile(r'(<script type="application/json" data-trajectory-frames>).*?(</script>)', re.S)


def _embed(trajectory: Path, report: Path, stride: int) -> None:
    frames = json.loads(trajectory.read_text(encoding="utf-8"))
    meta = json.loads(trajectory.with_name("meta.json").read_text(encoding="utf-8"))
    times = frames["times"]
    last = len(times) - 1
    indices = [*range(0, last, stride), last]
    payload = {
        "times": [round(times[i] - times[0], 6) for i in indices],
        "pos": [[[round(x, 4), round(y, 4)] for x, y in frames["pos"][i]] for i in indices],
        "spring": [[round(force, 3) for force in frames["spring_force"][i]] for i in indices],
        "contact": [[int(force > 0) for force in frames["contact_force"][i]] for i in indices],
        "edges": meta["creature"]["edges"],
    }
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    updated, count = _DATA.subn(lambda match: match[1] + encoded + match[2], report.read_text(encoding="utf-8"))
    if count != 1:
        raise ValueError("報告 HTML に軌跡データの埋め込み先が 1 つ必要です。")
    report.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--stride", type=int, default=6)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride には 1 以上を指定してください。")
    _embed(args.trajectory, args.report, args.stride)
