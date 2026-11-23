"""報告 HTML へ埋め込むエリート個体アニメーション (canvas) を生成する。

使い方:
  uv run python scripts/make_report_anim.py \
      resources/runs/first_try/trajectory_gen9_rank0.json \
      docs/report/assets/anim_gen9.js \
      --stride 6 --label "世代9 最良個体"

出力は <script src> で読み込む自己完結 JS。canvas[data-anim="<key>"] を探し、
再生・一時停止ボタンを figure に追加してループ再生する。
物理は 2D のまま、(x, y) をそのまま 2D canvas に描く (視点移動は viewer/ が担当)。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TEMPLATE = r"""
const W = D.world;
function setup(cv) {
  const ctx = cv.getContext("2d");
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cssW = cv.clientWidth || cv.width, cssH = cv.height;
  cv.width = Math.round(cssW * dpr);
  cv.height = Math.round(cssH * dpr);
  ctx.scale(dpr, dpr);
  const baseH = cssH, baseW = cssW;

  let playing = true, fi = 0, last = null, acc = 0;
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduce) playing = false;

  const fig = cv.closest("figure");
  const btn = document.createElement("button");
  btn.type = "button";
  btn.textContent = playing ? "⏸ 一時停止" : "▶ 再生";
  btn.style.cssText = "margin:.4rem .5rem .2rem 0; padding:.3rem .75rem; font:inherit; color:#ededed; background:#303030; border:1px solid #888; border-radius:.3rem; cursor:pointer";
  btn.addEventListener("click", () => { playing = !playing; last = null; btn.textContent = playing ? "⏸ 一時停止" : "▶ 再生"; });
  if (fig) fig.insertBefore(btn, cv);

  const world = (x, y) => {
    const winW = (W.yMax - W.yMin) * (baseW / baseH);
    const camX = Math.max(W.xMin, com[fi].x - winW * 0.30);
    const sx = (x - camX) / winW * baseW;
    const sy = baseH - (y - W.yMin) / (W.yMax - W.yMin) * baseH;
    return [sx, sy];
  };

  function draw() {
    const f = D.frames[fi];
    ctx.clearRect(0, 0, baseW, baseH);
    ctx.fillStyle = "#26282c"; ctx.fillRect(0, 0, baseW, baseH);
    // 1 m グリッド
    const winW = (W.yMax - W.yMin) * (baseW / baseH);
    const camX = Math.max(W.xMin, com[fi].x - winW * 0.30);
    ctx.strokeStyle = "#33373c"; ctx.lineWidth = 1;
    for (let g = Math.ceil(camX); g < camX + winW; g++) { const [sx] = world(g, 0); ctx.beginPath(); ctx.moveTo(sx, 0); ctx.lineTo(sx, baseH); ctx.stroke(); }
    for (let g = 0; g <= W.yMax; g++) { const [, sy] = world(0, g); ctx.beginPath(); ctx.moveTo(0, sy); ctx.lineTo(baseW, sy); ctx.stroke(); }
    // 地面
    const [, gy] = world(0, 0);
    ctx.strokeStyle = "#8a9099"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(baseW, gy); ctx.stroke();
    // 重心軌跡 (直近 60 フレーム)
    ctx.strokeStyle = "rgba(158,206,255,.55)"; ctx.lineWidth = 1.5; ctx.beginPath();
    for (let k = Math.max(0, fi - 60); k <= fi; k++) { const [sx, sy] = world(com[k].x, com[k].y); k === Math.max(0, fi - 60) ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy); }
    ctx.stroke();
    // 辺 (張力で色分け)
    const maxT = D.maxAbsTension || 1;
    for (let e = 0; e < D.edges.length; e++) {
      const [a, b] = D.edges[e];
      const pa = world(f.pos[a][0], f.pos[a][1]), pb = world(f.pos[b][0], f.pos[b][1]);
      const s = Math.max(-1, Math.min(1, f.tension[e] / maxT));
      ctx.strokeStyle = s >= 0 ? `hsl(${160 - 160 * s}, 75%, 55%)` : "hsl(210, 60%, 60%)";
      ctx.lineWidth = 3.5 + 2.5 * Math.abs(s);
      ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke();
    }
    // 質点と接触点
    for (let i = 0; i < f.pos.length; i++) {
      const [sx, sy] = world(f.pos[i][0], f.pos[i][1]);
      if (f.contact[i]) { ctx.fillStyle = "#e0b685"; ctx.beginPath(); ctx.arc(sx, sy, 9, 0, 2 * Math.PI); ctx.fill(); }
      ctx.fillStyle = "#e8ecef"; ctx.beginPath(); ctx.arc(sx, sy, 6, 0, 2 * Math.PI); ctx.fill();
    }
    // テキスト
    ctx.fillStyle = "#c9cdd2"; ctx.font = "13px ui-monospace, monospace";
    ctx.fillText(`t = ${f.t.toFixed(2)} s   COM Δx = ${(com[fi].x - com[0].x).toFixed(3)} m   (${fi + 1}/${D.frames.length})`, 10, 20);
  }

  function tick(now) {
    requestAnimationFrame(tick);
    if (playing) {
      acc += (now - (last ?? now)) / 1000 * 30;
      last = now;
      while (acc >= 1) { fi = (fi + 1) % D.frames.length; acc -= 1; }
      draw();
    } else {
      last = null;
    }
  }
  draw();
  requestAnimationFrame(tick);
}

const com = D.frames.map(f => f.com); // {x, y} の配列
document.querySelectorAll('canvas[data-anim]').forEach(cv => {
  if (cv.dataset.anim === D.key) setup(cv);
});
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trajectory", type=Path, help="trajectory_*.json")
    ap.add_argument("out", type=Path, help="出力 JS")
    ap.add_argument("--stride", type=int, default=6, help="間引き間隔 (dt=5ms なら 6 で約 30fps)")
    ap.add_argument("--label", default="", help="図の説明用ラベル")
    ap.add_argument("--y-max", type=float, default=None, help="縦軸の上限 (既定はデータから自動)")
    args = ap.parse_args()

    src = json.loads(args.trajectory.read_text())
    pos = src["pos"]
    tension = src["spring_force"]
    contact = src["contact_force"]
    times = src["times"]

    frames = []
    for k in range(0, len(pos), args.stride):
        p = pos[k]
        n = len(p)
        frames.append(
            {
                "t": round(times[k], 3),
                "pos": [[round(x, 2), round(y, 2)] for x, y in p],
                "tension": [round(v, 1) for v in tension[k]],
                "contact": [c > 0 for c in contact[k]],
                "com": {
                    "x": round(sum(q[0] for q in p) / n, 3),
                    "y": round(sum(q[1] for q in p) / n, 3),
                },
            }
        )
    y_max = args.y_max or max(f["com"]["y"] for f in frames) * 1.15
    max_tension = max(abs(v) for fr in frames for v in fr["tension"])

    payload = {
        "key": args.out.stem,
        "label": args.label,
        "world": {"xMin": 0.0, "yMin": 0.0, "yMax": round(y_max, 2)},
        "edges": _edges_for(len(frames[0]["pos"])),
        "maxAbsTension": round(max_tension, 3),
        "frames": frames,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "/* generated by scripts/make_report_anim.py -- do not edit */\n"
        "(() => {\nconst D = " + json.dumps(payload, separators=(",", ":")) + ";\n" + TEMPLATE + "})();\n"
    )
    print(f"[anim] {args.out} frames={len(frames)} yMax={payload['world']['yMax']}")


def _edges_for(n_nodes: int) -> list[list[int]]:
    """creature.triangulated_ladder と同じ辺順序 (2 列 x n_cols ラダー)。"""
    n_cols = n_nodes // 2
    pairs: list[list[int]] = []
    for i in range(n_cols - 1):
        pairs += [[i, i + 1], [n_cols + i, n_cols + i + 1], [i, n_cols + i], [i, n_cols + i + 1]]
    pairs.append([n_cols - 1, 2 * n_cols - 1])
    return pairs


if __name__ == "__main__":
    main()
