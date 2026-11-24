"""報告 HTML へ埋め込むエリート個体の動画 (mp4) を生成する。

使い方:
  uv run python scripts/make_report_video.py \
      resources/runs/first_try/trajectory_gen9_rank0.json \
      docs/report/assets/anim_gen9.mp4 \
      --stride 2 --fps 30 --label "世代9 最良個体"

軌跡 JSON (位置・張力・接触) を Pillow でフレーム描画し、ffmpeg で H.264 の
mp4 に変換する。物理は 2D のまま、(x, y) を 2D 画像に描く
(視点移動は viewer/ が担当)。報告側は <video controls loop muted> で埋め込む。
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

WIDTH, HEIGHT = 960, 420
BG = (38, 40, 44)
GRID = (52, 56, 61)
GROUND = (138, 144, 153)
NODE = (232, 236, 239)
CONTACT = (224, 182, 133)
TRAIL = (158, 206, 255)


def tension_color(s: float) -> tuple[int, int, int]:
    """張力 -1..1 を青→緑→赤に写す (canvas 版と同じ HSL 近似)。"""
    s = max(-1.0, min(1.0, s))
    if s >= 0:  # 緑→赤
        r = int(90 + 165 * s)
        g = int(214 - 90 * s)
        b = int(191 - 120 * s)
    else:  # 弛み (青系)
        r = int(80 + 10 * (1 + s))
        g = int(140 + 40 * (1 + s))
        b = int(200 + 20 * (1 + s))
    return r, g, b


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trajectory", type=Path, help="trajectory_*.json")
    ap.add_argument("out", type=Path, help="出力 mp4")
    ap.add_argument("--stride", type=int, default=2, help="間引き間隔 (dt=5ms なら 2 で 100fps 素材)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--label", default="", help="動画に重ねるラベル")
    ap.add_argument("--y-max", type=float, default=None, help="縦軸の上限 (既定はデータから自動)")
    ap.add_argument(
        "--html",
        type=Path,
        default=None,
        help="この HTML 内の src=\"<出力ファイルの相対パス>\" を data URI (base64) に置換する。"
        "python -m http.server は Range 非対応で <video> が再生できないため、埋め込みを既定とする",
    )
    args = ap.parse_args()

    src = json.loads(args.trajectory.read_text())
    pos, tension, contact, times = src["pos"], src["spring_force"], src["contact_force"], src["times"]

    idx = list(range(0, len(pos), args.stride))
    y_max = args.y_max or max(max(q[1] for q in pos[k]) for k in idx) * 1.15
    win_h = max(y_max, 0.5)
    win_w = win_h * WIDTH / HEIGHT
    com0 = [sum(q[0] for q in pos[0]) / len(pos[0])]

    def world(x: float, y: float, cam_x: float) -> tuple[int, int]:
        sx = (x - cam_x) / win_w * WIDTH
        sy = HEIGHT - (y / win_h) * HEIGHT
        return int(round(sx)), int(round(sy))

    with tempfile.TemporaryDirectory(dir="/tmp/opencode") as tmp:
        tmp_dir = Path(tmp)
        for n, k in enumerate(idx):
            f = pos[k]
            com_x = sum(q[0] for q in f) / len(f)
            cam_x = max(0.0, com_x - win_w * 0.30)
            img = Image.new("RGB", (WIDTH, HEIGHT), BG)
            d = ImageDraw.Draw(img)
            # グリッド (1 m)
            gx = int(cam_x)
            while gx <= cam_x + win_w:
                x0, _ = world(gx, 0, cam_x)
                d.line([(x0, 0), (x0, HEIGHT)], fill=GRID, width=1)
                gx += 1
            gy = 0.0
            while gy <= win_h:
                _, y0 = world(0, gy, cam_x)
                d.line([(0, y0), (WIDTH, y0)], fill=GRID, width=1)
                gy += 1.0
            # 地面
            _, gy0 = world(0, 0, cam_x)
            d.line([(0, gy0), (WIDTH, gy0)], fill=GROUND, width=3)
            # 重心軌跡 (直近 60 サンプル)
            trail = []
            for j in range(max(0, n - 60), n + 1):
                fj = pos[idx[j]]
                trail.append(world(sum(q[0] for q in fj) / len(fj), sum(q[1] for q in fj) / len(fj), cam_x))
            if len(trail) > 1:
                d.line(trail, fill=TRAIL, width=2)
            # 辺 (張力色・太さ)
            max_t = max(abs(v) for v in tension[k]) or 1.0
            for e, (a, b) in enumerate(_edges_for(len(f))):
                p0 = world(f[a][0], f[a][1], cam_x)
                p1 = world(f[b][0], f[b][1], cam_x)
                s = max(-1.0, min(1.0, tension[k][e] / max_t))
                d.line([p0, p1], fill=tension_color(s), width=4 + int(2 * abs(s)))
            # 接触点と質点
            for i, q in enumerate(f):
                cx, cy = world(q[0], q[1], cam_x)
                if contact[k][i] > 0:
                    d.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], fill=CONTACT)
                d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=NODE)
            # テキスト
            d.text((10, 8), f"t = {times[k]:.2f} s   COM dx = {com_x - com0[0]:.3f} m", fill=(201, 205, 210))
            d.text((10, HEIGHT - 20), f"{args.label}  ({n + 1}/{len(idx)})", fill=(160, 170, 178))
            img.save(tmp_dir / f"f{n:04d}.png")

        args.out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg", "-y", "-framerate", str(args.fps),
                "-i", str(tmp_dir / "f%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                "-movflags", "+faststart", str(args.out),
            ],
            check=True,
            capture_output=True,
        )
    size_kb = args.out.stat().st_size // 1024
    print(f"[video] {args.out} frames={len(idx)} fps={args.fps} {size_kb} KB")

    if args.html is not None:
        embedded = embed_into_html(args.out, args.html)
        print(f"[video] {args.html} に data URI として埋め込み ({embedded} 箇所)")


def embed_into_html(video_path: Path, html_path: Path) -> int:
    """HTML の video src (video_path と同名の相対パス) を base64 の data URI に置換する。

    何度実行しても同じ結果になる (data URI は置換対象に含まれない)。
    """
    html = html_path.read_text(encoding="utf-8")
    b64 = base64.b64encode(video_path.read_bytes()).decode("ascii")
    data_uri = f"data:video/mp4;base64,{b64}"
    # ファイル名で終わる src のうち data URI でないものだけを置換する
    pattern = re.compile(r'src="(?!data:)[^"]*' + re.escape(video_path.name) + r'"')
    html, count = pattern.subn(f'src="{data_uri}"', html)
    html_path.write_text(html, encoding="utf-8")
    return count


def _edges_for(n_nodes: int) -> list[tuple[int, int]]:
    """creature.triangulated_ladder と同じ辺順序 (2 列 x n_cols ラダー)。"""
    n_cols = n_nodes // 2
    pairs: list[tuple[int, int]] = []
    for i in range(n_cols - 1):
        pairs += [(i, i + 1), (n_cols + i, n_cols + i + 1), (i, n_cols + i), (i, n_cols + i + 1)]
    pairs.append((n_cols - 1, 2 * n_cols - 1))
    return pairs


if __name__ == "__main__":
    main()
