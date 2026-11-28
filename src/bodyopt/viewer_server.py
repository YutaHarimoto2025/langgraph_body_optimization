"""軌跡ビューア用のローカルサーバー (標準ライブラリのみ)。

viewer/ 以下の静的ファイルを配信しつつ、指定したケース・世代・順位の個体を
replay して /tmp/bodyopt-viewer に軌跡 JSON を書き出す API を提供する。

  uv run python -m bodyopt viewer --port 8765
"""
from __future__ import annotations

import json
import sys
import tempfile
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "resources" / "runs"
TEMP_DIR = Path(tempfile.gettempdir()) / "bodyopt-viewer"


def parse_step_spec(text: str) -> list[int]:
    """'1,100,10' を [1, 11, ..., 91] に展開する。数値 1 つなら単一値。

    '1,3' のように 2 つなら 1..3 (間隔 1) とみなす。開始 > 終了なら降順に展開し、
    読み込み順（レーンの手前からの順）もそのまま逆になる。
    """
    nums = [int(s) for s in text.replace(" ", "").split(",") if s]
    if not nums:
        raise ValueError("数値の指定が空です")
    if len(nums) == 1:
        return nums
    start, end = nums[0], nums[1]
    direction = 1 if end >= start else -1
    step = abs(nums[2]) if len(nums) > 2 else 1
    if step == 0:
        raise ValueError("間隔は 0 以外にしてください")
    return list(range(start, end + direction, direction * step))


def _resolve_run(case: str) -> Path:
    p = Path(case)
    p = (p if p.is_absolute() else RUNS / p).resolve()
    if not (p / "meta.json").is_file():
        raise ValueError(f"run が見つかりません: {p}")
    return p


class _Handler(SimpleHTTPRequestHandler):
    """静的配信 + /_api/cases・/_api/replay・/_api/tmp/<file>。"""

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/_api/cases":
            cases = sorted(p.name for p in RUNS.iterdir() if (p / "meta.json").is_file())
            self._json({"cases": cases, "root": str(ROOT)})
            return
        if path.startswith("/_api/tmp/"):
            self._serve_temp(Path(unquote(path[len("/_api/tmp/"):])).name)
            return
        super().do_GET()

    def do_OPTIONS(self) -> None:
        if not self.path.startswith("/_api/"):
            self.send_error(404)
            return
        # VS Code プレビュー (vscode-webview://) からの cross-origin POST を受ける
        self.send_response(204)
        self._cors()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/_api/replay":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            case = str(body.get("case", "")).strip()
            gens = parse_step_spec(str(body.get("gens", "0")))
            ranks = parse_step_spec(str(body.get("ranks", "0")))
            raw_seconds = body.get("seconds")
            seconds: float | None = None
            if raw_seconds not in (None, ""):
                try:
                    seconds = float(raw_seconds)
                except (TypeError, ValueError):
                    raise ValueError("秒数は数値で指定してください") from None
                if seconds <= 0:
                    raise ValueError("秒数は正の値にしてください")
            if not case:
                raise ValueError("ケース名が空です")
            if not gens or not ranks:
                raise ValueError("gen/rank の範囲が空です (開始 <= 終了)")
            if min(gens) < 0:
                raise ValueError("gen は 0 以上にしてください")
            if min(ranks) < 0:
                raise ValueError("rank は 0 始まりです (0 = 最良)")
            run_dir = _resolve_run(case)
        except (ValueError, AttributeError) as e:
            self._json({"items": [], "errors": [], "error": str(e)}, status=400)
            return

        import subprocess

        # 子プロセスで replay する。常に現在のソースコードが使われ、
        # 編集のたびにサーバーを再起動する必要がない。
        pairs = [(g, r) for g in gens for r in ranks]
        labels = [f"{run_dir.name}_gen{g}_rank{r}" for g, r in pairs]
        print(f"[viewer] replay {run_dir.name} gens={gens} ranks={ranks} ({len(pairs)} 個体)")
        cmd = [
            sys.executable, "-m", "bodyopt", "_replay-worker",
            "--run", str(run_dir),
            "--targets", json.dumps(pairs),
            "--labels", json.dumps(labels),
            "--out", str(TEMP_DIR),
        ]
        if seconds is not None:
            cmd += ["--seconds", str(seconds)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
            self._json(
                {"items": [], "errors": [], "error": "replay に失敗: " + " / ".join(detail)},
                status=500,
            )
            return
        try:
            results = json.loads(proc.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as e:
            self._json({"items": [], "errors": [], "error": f"replay 出力の解析に失敗: {e}"}, status=500)
            return

        items: list[dict[str, Any]] = []
        errors: list[str] = []
        for res in results:
            if not res.get("path"):
                errors.append(
                    f"{run_dir.name} gen{res['generation']} rank{res['rank']}: {res.get('error')}"
                )
                continue
            items.append({
                "case": run_dir.name,
                "gen": res["generation"],
                "rank": res["rank"],
                "score": round(res["score"], 6),
                "distance": round(res["distance"], 6),
                "url": f"/_api/tmp/{Path(res['path']).name}",
            })
        self._json({"items": items, "errors": errors})

    def _serve_temp(self, name: str) -> None:
        path = TEMP_DIR / name
        if not name or not path.is_file():
            self.send_error(404, "not found")
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def _json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def _cors(self) -> None:
        # ローカル専用 API。認証情報は使わないので全オリジンを許可する。
        self.send_header("Access-Control-Allow-Origin", "*")

    def log_message(self, format: str, *args: Any) -> None:
        # 静的ファイルの取得は黙らせ、API とエラーだけ残す
        if self.path.startswith("/_api/"):
            print(f"[viewer] {self.address_string()} {format % args}")


def source_fingerprint() -> float:
    """サーバー側モジュールの最終更新時刻。変更検知に使う。"""
    root = Path(__file__).resolve().parent
    return max((f.stat().st_mtime for f in root.glob("*.py")), default=0.0)


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    handler = partial(_Handler, directory=str(ROOT))
    with HTTPServer((host, port), handler) as httpd:
        print(f"[viewer] http://{host}:{port}/viewer/  (Ctrl+C で終了)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[viewer] 停止しました")
