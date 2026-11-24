"""報告 HTML へ埋め込むエリート個体の動画 (WebM / VP9) を生成する。

使い方:
  uv run python scripts/make_report_video.py \
      resources/runs/first_try/trajectory_gen9_rank0.json \
      docs/report/assets/anim_gen9.webm \
      --stride 6 --fps 30 --label "世代9 最良個体" \
      --html docs/report/2026-11-23_first_ga_run.html

軌跡 JSON (位置・張力・接触) を Pillow でフレーム描画し、ffmpeg で VP9 の
WebM に変換する。VP9 は Chromium 組み込みのコーデックで、VS Code の preview
(webview) でも <video controls> の再生・停止・シークが使える。
物理は 2D のまま、(x, y) を 2D 画像に描く。カメラは重心に追従する。
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 960, 420
BG = (38, 40, 44)
GRID = (52, 56, 61)
GROUND = (138, 144, 153)
NODE = (232, 236, 239)
CONTACT = (224, 182, 133)
TRAIL = (158, 206, 255)
VIEW_H = 4.2  # 表示窓の高さ [m] (重心に追従)
JP_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
)


def load_font(size: int = 17) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for path in JP_FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def tension_color(s: float) -> tuple[int, int, int]:
    """張力 -1..1 を青→緑→赤に写す。"""
    s = max(-1.0, min(1.0, s))
    if s >= 0:
        r = int(90 + 165 * s)
        g = int(214 - 90 * s)
        b = int(191 - 120 * s)
    else:
        r = int(80 + 10 * (1 + s))
        g = int(140 + 40 * (1 + s))
        b = int(200 + 20 * (1 + s))
    return r, g, b


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trajectory", type=Path, help="trajectory_*.json")
    ap.add_argument("out", type=Path, help="出力 WebM")
    ap.add_argument("--stride", type=int, default=3, help="間引き間隔 (dt=5ms なら 3 で約 67fps 素材)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--label", default="", help="動画に重ねるラベル")
    ap.add_argument("--view-h", type=float, default=VIEW_H, help="表示窓の高さ [m]")
    ap.add_argument(
        "--html",
        type=Path,
        default=None,
        help="この HTML 内の src を data URI (base64) に置換する (preview 内で再生するため)",
    )
    args = ap.parse_args()

    src = json.loads(args.trajectory.read_text())
    pos, tension, contact, times = src["pos"], src["spring_force"], src["contact_force"], src["times"]

    idx = list(range(0, len(pos), args.stride))
    win_h = args.view_h
    win_w = win_h * WIDTH / HEIGHT
    font = load_font(17)
    com0 = sum(q[0] for q in pos[0]) / len(pos[0])

    def world(x: float, y: float, cam_x: float, cam_y: float) -> tuple[int, int]:
        sx = (x - cam_x) / win_w * WIDTH
        sy = HEIGHT - (y - cam_y) / win_h * HEIGHT
        return int(round(sx)), int(round(sy))

    with tempfile.TemporaryDirectory(dir="/tmp/opencode") as tmp:
        tmp_dir = Path(tmp)
        for n, k in enumerate(idx):
            f = pos[k]
            com_x = sum(q[0] for q in f) / len(f)
            com_y = sum(q[1] for q in f) / len(f)
            cam_x = max(0.0, com_x - win_w * 0.30)
            cam_y = max(0.0, com_y - win_h * 0.55)  # 身体が上寄りに来るよう追従
            img = Image.new("RGB", (WIDTH, HEIGHT), BG)
            d = ImageDraw.Draw(img)
            # グリッド (1 m)
            gx = int(cam_x)
            while gx <= cam_x + win_w:
                x0, _ = world(gx, 0, cam_x, cam_y)
                d.line([(x0, 0), (x0, HEIGHT)], fill=GRID, width=1)
                gx += 1
            gy = int(cam_y)
            while gy <= cam_y + win_h + 1:
                _, y0 = world(0, gy, cam_x, cam_y)
                d.line([(0, y0), (WIDTH, y0)], fill=GRID, width=1)
                gy += 1
            # 地面 (窓内にあるときのみ)
            _, gy0 = world(0, 0, cam_x, cam_y)
            if 0 <= gy0 <= HEIGHT:
                d.line([(0, gy0), (WIDTH, gy0)], fill=GROUND, width=3)
            # 重心軌跡 (直近 45 サンプル)
            trail = []
            for j in range(max(0, n - 45), n + 1):
                fj = pos[idx[j]]
                trail.append(
                    world(sum(q[0] for q in fj) / len(fj), sum(q[1] for q in fj) / len(fj), cam_x, cam_y)
                )
            if len(trail) > 1:
                d.line(trail, fill=TRAIL, width=2)
            # 辺 (張力色・太さ)
            max_t = max(abs(v) for v in tension[k]) or 1.0
            for e, (a, b) in enumerate(_edges_for(len(f))):
                p0 = world(f[a][0], f[a][1], cam_x, cam_y)
                p1 = world(f[b][0], f[b][1], cam_x, cam_y)
                s = max(-1.0, min(1.0, tension[k][e] / max_t))
                d.line([p0, p1], fill=tension_color(s), width=5 + int(2 * abs(s)))
            # 接触点と質点
            for i, q in enumerate(f):
                cx, cy = world(q[0], q[1], cam_x, cam_y)
                if contact[k][i] > 0:
                    d.ellipse([cx - 10, cy - 10, cx + 10, cy + 10], fill=CONTACT)
                d.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=NODE)
            # テキスト
            d.text(
                (14, 10),
                f"t = {times[k]:.2f} s   COM dx = {com_x - com0:.3f} m",
                fill=(209, 213, 218),
                font=font,
            )
            d.text((14, HEIGHT - 28), f"{args.label}  ({n + 1}/{len(idx)})", fill=(160, 170, 178), font=font)
            img.save(tmp_dir / f"f{n:04d}.png")

        args.out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg", "-y", "-framerate", str(args.fps),
                "-i", str(tmp_dir / "f%04d.png"),
                "-c:v", "libvpx-vp9", "-crf", "30", "-b:v", "0",
                "-pix_fmt", "yuv420p", "-row-mt", "1", str(args.out),
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
    """HTML の video src を base64 の data URI に置換する (何度実行しても同じ結果)。"""
    html = html_path.read_text(encoding="utf-8")
    mime = "video/webm" if video_path.suffix == ".webm" else "video/mp4"
    b64 = base64.b64encode(video_path.read_bytes()).decode("ascii")
    data_uri = f"data:{mime};base64,{b64}"
    pattern = re.compile(
        r'src="(?!data:)[^"]*' + re.escape(video_path.name) + r'"'
        + r"|" + r'src="data:' + re.escape(mime) + r';base64,[^"]*"'
        + r"|" + r'src="data:video/(?:mp4|webm);base64,[^"]*"'
    )
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
