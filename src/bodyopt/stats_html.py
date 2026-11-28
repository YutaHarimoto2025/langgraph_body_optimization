"""stats.json から世代統計の HTML レポートを生成する (依存なし・自己完結)。

使い方:
  uv run python -m bodyopt stats-html --run resources/runs/008

出力は stats.json と同じディレクトリの stats.html。数値は stats.json だけを読み、
meta.json があれば実行設定の表示に使う。未来の LangGraph エージェントの判断ログは
stats.json の "decisions"、または同じディレクトリの decisions.json から読む。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

W = 1100
PANEL_H = 184
PANEL_GAP = 14
LEFT = 96
RIGHT = 24
BG = "#232323"
PANEL_BG = "#2b2b2b"
GRID = "#4a4a4a"
FG = "#ededed"
MUTED = "#c9cdd2"
ACCENT = "#9eceff"
DECISION = "#d79bd7"
GENE_TYPES = (
    ("amplitude", "振幅", "N"),
    ("frequency", "周波数", "Hz"),
    ("phase", "位相", "rad"),
    ("stiffness", "剛性 g", "0–1"),
)
GENE_GROUPS = (
    ("top1", "上位1%", ACCENT),
    ("top10", "上位10%", "#e0b685"),
    ("all", "全個体", "#85d6bf"),
)


@dataclass(frozen=True)
class Series:
    key: str
    label: str
    color: str


@dataclass(frozen=True)
class Panel:
    title: str
    series: tuple[Series, ...]
    fixed: tuple[float, float] | None = None
    reference: float | None = None
    band: tuple[str, str] | None = None
    fmt: Callable[[float], str] = lambda v: _fmt(v)


def _fmt(v: float) -> str:
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 10:
        return f"{v:.2f}"
    if a >= 1:
        return f"{v:.3f}"
    if a >= 0.001:
        return f"{v:.4f}"
    return f"{v:.2e}"


def _pct(v: float) -> str:
    return f"{v * 100:.0f}%"


def _esc(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _load_decisions(run_dir: Path, stats: dict[str, Any]) -> list[dict[str, Any]]:
    raw: Any = stats.get("decisions")
    path = run_dir / "decisions.json"
    if raw is None and path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("decisions", [])
    out: list[dict[str, Any]] = []
    for item in raw or []:
        if not isinstance(item, dict) or "generation" not in item:
            continue
        end = item.get("end_generation")
        out.append(
            {
                "generation": int(item["generation"]),
                "end_generation": int(end) if end is not None else None,
                "label": str(item.get("label", "")),
                "detail": str(item.get("detail", "")),
            }
        )
    return sorted(out, key=lambda d: d["generation"])


def _panels(per: dict[str, Any]) -> list[Panel]:
    best = per.get("best_score")
    panels: list[Panel] = []
    if isinstance(best, list) and best:
        score_band = ("score_dist.p10", "score_dist.p90") if isinstance(per.get("score_dist"), dict) else None
        panels.append(
            Panel(
                "評価スコア (重心の移動距離)",
                (Series("best_score", "最良", ACCENT), Series("mean_score", "平均", "#e0b685")),
                band=score_band,
            )
        )
        dist = per.get("best_distance")
        if isinstance(dist, list) and any(abs(a - b) > 1e-6 for a, b in zip(best, dist)):
            distance_band = (
                ("distance_dist.p10", "distance_dist.p90")
                if isinstance(per.get("distance_dist"), dict)
                else None
            )
            panels.append(
                Panel(
                    "重心移動距離",
                    (Series("best_distance", "最良", ACCENT), Series("mean_distance", "平均", "#e0b685")),
                    band=distance_band,
                )
            )
    sigma = per.get("gene_sigma")
    if isinstance(sigma, dict):
        for name, label, unit in GENE_TYPES:
            groups = sigma.get(name)
            if not isinstance(groups, dict):
                continue
            series = tuple(
                Series(f"gene_sigma.{name}.{group}", group_label, color)
                for group, group_label, color in GENE_GROUPS
                if isinstance(groups.get(group), list)
            )
            if series:
                panels.append(Panel(f"{label}のσ (選択個体×全エッジ) [{unit}]", series))
    if isinstance(per.get("valid_rate"), list):
        panels.append(Panel("有効個体の割合", (Series("valid_rate", "有効率", "#8fbe9f"),), (0.0, 1.0), fmt=_pct))
    if isinstance(per.get("best_energy"), list):
        energy_series = [Series("best_energy", "最良", "#d79bd7")]
        if isinstance(per.get("mean_energy"), list):
            energy_series.append(Series("mean_energy", "平均", "#e0b685"))
        panels.append(Panel("駆動仕事 [J]", tuple(energy_series)))
    return panels


def _values(per: dict[str, Any], key: str, n: int) -> list[float | None]:
    raw: Any = per
    for part in key.split("."):
        raw = raw.get(part) if isinstance(raw, dict) else None
    if not isinstance(raw, list):
        return [None] * n
    out: list[float | None] = []
    for i in range(n):
        v = raw[i] if i < len(raw) else None
        out.append(float(v) if isinstance(v, (int, float)) and math.isfinite(v) else None)
    return out


def _ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    if not math.isfinite(lo) or not math.isfinite(hi):
        return []
    if hi <= lo:
        hi = lo + max(abs(lo) * 0.05, 1e-6)
    raw = (hi - lo) / max(n, 1)
    mag = 10.0 ** math.floor(math.log10(raw))
    step = mag
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= m * mag:
            step = m * mag
            break
    start = math.ceil(lo / step) * step
    out: list[float] = []
    v = start
    while v <= hi + step * 1e-6 and len(out) < 12:
        out.append(0.0 if abs(v) < step * 1e-9 else v)
        v += step
    return out


def _segments(xs: list[int], ys: list[float | None], x_of: Callable[[float], float], y_of: Callable[[float], float]) -> str:
    parts: list[str] = []
    pen_up = False
    for x, y in zip(xs, ys):
        if y is None:
            pen_up = False
            continue
        cmd = "M" if not pen_up else "L"
        parts.append(f"{cmd}{x_of(x):.1f},{y_of(y):.1f}")
        pen_up = True
    return " ".join(parts)


def _band_path(
    xs: list[int],
    low: list[float | None],
    high: list[float | None],
    x_of: Callable[[float], float],
    y_of: Callable[[float], float],
) -> str:
    """p10–p90 のような帯を、連続する区間ごとの閉じたパスにする。"""
    parts: list[str] = []
    run: list[tuple[int, float, float]] = []

    def flush() -> None:
        if len(run) < 2:
            run.clear()
            return
        upper = "".join(
            f"{'M' if i == 0 else 'L'}{x_of(x):.1f},{y_of(hi):.1f}"
            for i, (x, _lo, hi) in enumerate(run)
        )
        lower = "".join(f"L{x_of(x):.1f},{y_of(lo):.1f}" for x, lo, _hi in reversed(run))
        parts.append(upper + lower + "Z")
        run.clear()

    for x, lo, hi in zip(xs, low, high):
        if lo is None or hi is None:
            flush()
            continue
        run.append((x, lo, hi))
    flush()
    return "".join(parts)


def _svg(stats: dict[str, Any], decisions: list[dict[str, Any]], note: str) -> str:
    per = stats.get("per_generation")
    if not isinstance(per, dict):
        per = {}
    n = int(stats.get("n_generations") or len(per.get("best_score") or []))
    n = max(n, 1)
    xs = list(range(n))
    panels = _panels(per)
    if not panels:
        return f'<p class="note">{_esc(note)}</p>'

    top = 40 if decisions else 16
    height = top + len(panels) * (PANEL_H + PANEL_GAP) + 52
    plot_w = W - LEFT - RIGHT

    def x_of(g: float) -> float:
        return LEFT if n == 1 else LEFT + g / (n - 1) * plot_w

    svg: list[str] = [
        f'<svg viewBox="0 0 {W} {height}" width="100%" role="img" aria-label="世代統計" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{W}" height="{height}" fill="{BG}"/>',
    ]

    # 判断ログの帯 (期間があるもの)
    for d in decisions:
        if d["end_generation"] is None:
            continue
        x1, x2 = x_of(d["generation"]), x_of(min(d["end_generation"], n - 1))
        svg.append(
            f'<rect x="{x1:.1f}" y="{top}" width="{max(x2 - x1, 1):.1f}" height="{height - top - 42}" '
            f'fill="{DECISION}" opacity="0.07"/>'
        )

    for pi, panel in enumerate(panels):
        y0 = top + pi * (PANEL_H + PANEL_GAP)
        pt = y0 + 34
        pb = y0 + PANEL_H - 10
        svg.append(f'<rect x="{LEFT}" y="{y0}" width="{plot_w}" height="{PANEL_H}" fill="{PANEL_BG}" rx="4"/>')
        svg.append(f'<text x="{LEFT + 8}" y="{y0 + 22}" fill="{FG}" font-size="15" font-weight="700">{_esc(panel.title)}</text>')

        series_vals = {s.key: _values(per, s.key, n) for s in panel.series}
        finite = [v for vals in series_vals.values() for v in vals if v is not None]
        if panel.fixed is not None:
            lo, hi = panel.fixed
        elif finite or panel.reference is not None:
            vals = list(finite) + ([panel.reference] if panel.reference is not None else [])
            lo, hi = min(vals), max(vals)
            pad = (hi - lo) * 0.08 or max(abs(hi) * 0.05, 1e-6)
            lo, hi = lo - pad, hi + pad
        else:
            lo, hi = 0.0, 1.0

        def y_of(v: float) -> float:
            t = 0.0 if hi <= lo else (v - lo) / (hi - lo)
            return pb - t * (pb - pt)

        for tick in _ticks(lo, hi):
            y = y_of(tick)
            if not (pt - 1 <= y <= pb + 1):
                continue
            svg.append(f'<line x1="{LEFT}" y1="{y:.1f}" x2="{LEFT + plot_w}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1" opacity="0.7"/>')
            svg.append(f'<text x="{LEFT - 12}" y="{y + 5:.1f}" fill="{MUTED}" font-size="13" text-anchor="end">{_esc(panel.fmt(tick))}</text>')

        if panel.reference is not None and panel.reference <= hi:
            ry = y_of(panel.reference)
            svg.append(f'<line x1="{LEFT}" y1="{ry:.1f}" x2="{LEFT + plot_w}" y2="{ry:.1f}" stroke="{MUTED}" stroke-width="1" stroke-dasharray="2 4" opacity="0.6"/>')
            svg.append(f'<text x="{LEFT + plot_w - 4}" y="{ry - 5:.1f}" fill="{MUTED}" font-size="12" text-anchor="end">一様 {panel.reference:.3f}</text>')

        # 横軸の目盛り線 (全パネルに薄く引く)
        step = max(1, (n - 1) // 6)
        xticks = sorted(set(list(range(0, n, step)) + [n - 1]))
        for g in xticks:
            svg.append(f'<line x1="{x_of(g):.1f}" y1="{pt}" x2="{x_of(g):.1f}" y2="{pb}" stroke="{GRID}" stroke-width="1" opacity="0.25"/>')

        if panel.band is not None:
            band = _band_path(
                xs,
                _values(per, panel.band[0], n),
                _values(per, panel.band[1], n),
                x_of,
                y_of,
            )
            if band:
                svg.append(f'<path d="{band}" fill="{ACCENT}" opacity="0.13"/>')

        for s in panel.series:
            ys = series_vals[s.key]
            path = _segments(xs, ys, x_of, y_of)
            if not path:
                continue
            svg.append(f'<path d="{path}" fill="none" stroke="{s.color}" stroke-width="2.2" stroke-linejoin="round"/>')
            stride = max(1, n // 70)
            for i, y in enumerate(ys):
                if y is None or (i % stride and i not in (0, n - 1)):
                    continue
                svg.append(
                    f'<circle cx="{x_of(i):.1f}" cy="{y_of(y):.1f}" r="3" fill="{s.color}">'
                    f'<title>世代 {i} {_esc(s.label)}: {_esc(panel.fmt(y))}</title></circle>'
                )

        # 凡例 (右上)
        lx = LEFT + plot_w - 10
        for s in reversed(panel.series):
            text_w = 9 * len(s.label) + 30
            lx -= text_w
            svg.append(f'<line x1="{lx}" y1="{y0 + 14}" x2="{lx + 16}" y2="{y0 + 14}" stroke="{s.color}" stroke-width="3"/>')
            svg.append(f'<text x="{lx + 21}" y="{y0 + 19}" fill="{MUTED}" font-size="13">{_esc(s.label)}</text>')

    # 判断ログの縦線と番号
    for i, d in enumerate(decisions):
        x = x_of(min(max(d["generation"], 0), n - 1))
        tip = f'世代 {d["generation"]}: {d["label"]}' + (f" — {d['detail']}" if d["detail"] else "")
        svg.append(
            f'<line x1="{x:.1f}" y1="{top - 2}" x2="{x:.1f}" y2="{height - 40}" '
            f'stroke="{DECISION}" stroke-width="1.5" stroke-dasharray="5 4" opacity="0.8">'
            f'<title>{_esc(tip)}</title></line>'
        )
        svg.append(f'<circle cx="{x:.1f}" cy="{top - 14}" r="9" fill="{DECISION}"><title>{_esc(tip)}</title></circle>')
        svg.append(f'<text x="{x:.1f}" y="{top - 10}" fill="{BG}" font-size="11" font-weight="700" text-anchor="middle">{i + 1}</text>')

    # 横軸ラベル
    step = max(1, (n - 1) // 6)
    xticks = sorted(set(list(range(0, n, step)) + [n - 1]))
    for g in xticks:
        svg.append(f'<text x="{x_of(g):.1f}" y="{height - 26}" fill="{MUTED}" font-size="13" text-anchor="middle">{g}</text>')
    svg.append(f'<text x="{LEFT + plot_w / 2:.1f}" y="{height - 6}" fill="{MUTED}" font-size="14" text-anchor="middle">世代</text>')
    svg.append("</svg>")
    return "".join(svg)


def _kpi(label: str, value: str, unit: str = "") -> str:
    return (
        f'<div class="kpi"><span>{_esc(label)}</span>'
        f'<strong>{_esc(value)}</strong><em>{_esc(unit)}</em></div>'
    )


def _last_valid(vals: list[float | None]) -> float | None:
    for v in reversed(vals):
        if v is not None:
            return v
    return None


def _meta_rows(meta: dict[str, Any]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if not meta:
        return rows
    git = meta.get("git") or {}
    if git.get("commit"):
        dirty = " (dirty)" if git.get("dirty") else ""
        rows.append(("commit", f'{str(git["commit"])[:10]}{dirty}'))
    for key, label in (
        ("command", "コマンド"),
        ("record_level", "記録レベル"),
        ("elapsed_seconds", "計算時間 [s]"),
        ("peak_rss_mb", "RAM ピーク [MB]"),
        ("jax_version", "JAX"),
    ):
        if meta.get(key) is not None:
            rows.append((label, str(meta[key])))
    sim = meta.get("sim_params") or {}
    for key, label in (
        ("muscle_model", "筋肉モデル"),
        ("dt", "dt [s]"),
        ("episode_seconds", "試行時間 [s]"),
        ("edge_damping_ratio", "減衰比 ζ"),
    ):
        if sim.get(key) is not None:
            rows.append((label, str(sim[key])))
    ga = meta.get("ga_params") or {}
    for key, label in (("pop_size", "集団サイズ"), ("seed", "seed")):
        if ga.get(key) is not None:
            rows.append((label, str(ga[key])))
    return rows


def render_html(
    stats: dict[str, Any],
    run_name: str,
    meta: dict[str, Any] | None = None,
    decisions: list[dict[str, Any]] | None = None,
) -> str:
    meta = meta or {}
    decisions = decisions or []
    per = stats.get("per_generation")
    if not isinstance(per, dict):
        per = {}
    n = int(stats.get("n_generations") or len(per.get("best_score") or []))
    n = max(n, 1)

    best = _values(per, "best_score", n)
    mean = _values(per, "mean_score", n)
    valid = _values(per, "valid_rate", n)
    best_finite = [v for v in best if v is not None and v > -1e5]
    valid_last = _last_valid(valid)
    mean_last = _last_valid(mean)
    has_sigma = isinstance(per.get("gene_sigma"), dict)

    elapsed = meta.get("elapsed_seconds")
    kpis = [
        _kpi("世代数", str(n)),
        _kpi("最良スコア", _fmt(max(best_finite)) if best_finite else "—", "m"),
        _kpi("最終世代の平均", _fmt(mean_last) if mean_last is not None else "—", "m"),
        _kpi("最終世代の有効率", _pct(valid_last) if valid_last is not None else "—"),
    ]
    if has_sigma:
        for name, label, unit in GENE_TYPES:
            v = _last_valid(_values(per, f"gene_sigma.{name}.all", n))
            kpis.append(_kpi(f"最終世代 {label}σ (全個体)", _fmt(v) if v is not None else "—", unit if v is not None else ""))
    kpis.append(_kpi("計算時間", str(elapsed) if elapsed is not None else "—", "s" if elapsed is not None else ""))

    meta_rows = _meta_rows(meta)
    meta_html = ""
    if meta_rows:
        body = "".join(f"<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>" for k, v in meta_rows)
        meta_html = f'<details><summary>実行設定</summary><table class="kv">{body}</table></details>'

    if decisions:
        rows = "".join(
            "<tr>"
            f"<td>{i + 1}</td><td>{d['generation']}</td><td>{_esc(d['label'])}</td><td>{_esc(d['detail'])}</td>"
            "</tr>"
            for i, d in enumerate(decisions)
        )
        decision_html = (
            "<h2>エージェントの意思決定</h2>"
            '<table class="decisions"><thead><tr><th>#</th><th>世代</th><th>判断</th><th>根拠・内容</th></tr></thead>'
            f"<tbody>{rows}</tbody></table>"
        )
    else:
        decision_html = (
            "<h2>エージェントの意思決定</h2>"
            '<p class="note">まだ判断ログはありません。stats.json の <code>decisions</code>、'
            "または同じディレクトリの <code>decisions.json</code> に "
            '<code>[{&quot;generation&quot;: 120, &quot;label&quot;: &quot;突然変異率を上げる&quot;, '
            '&quot;detail&quot;: &quot;10 世代停滞&quot;}]</code> を置くと、この図に縦線と帯が入ります。'
            "<code>end_generation</code> を足すと期間の帯になります。</p>"
        )

    note = (
        "無効個体のスコア (-1e6) は評価スコアと距離のパネルから除外した。"
        if any(v is not None and v <= -1e5 for v in best)
        else ""
    )
    valid_note = (
        '<p class="note">有効個体 = 位置・速度が有限、最大めり込み &lt; 0.10 m、'
        "energy_budget 設定時は駆動仕事 ≤ 予算。無効個体のスコアは -1e6。</p>"
    )
    if has_sigma:
        sigma_note = (
            '<p class="note">σ は選択群の同種の全筋肉・全次元をまとめた標準偏差。'
            "上位1%が1個体のときは、その個体のエッジ間のばらつきを表す。"
            'スコアと距離の帯は有効個体の <span style="white-space:nowrap">p10–p90</span>。</p>'
        )
    else:
        sigma_note = (
            '<p class="warn">遺伝子σが stats.json に無い。この run は summary 記録 (集団を保存しない) か、'
            "機能追加前の形式。summary 記録からは復元できないので、必要なら <code>--record population</code> "
            "で再実行する。</p>"
        )
    chart = _svg(stats, decisions, "表示できる世代統計が stats.json にない。")

    return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(run_name)}｜世代統計</title>
<style>
  :root {{ color-scheme: dark; font-family: system-ui, sans-serif; background: {BG}; color: {FG}; }}
  body {{ max-width: 76rem; margin: auto; padding: 1.5rem 1rem 3rem; line-height: 1.7; font-size: 17px; }}
  h1 {{ font-size: 1.9rem; margin: 0 0 .3rem; }}
  h2 {{ font-size: 1.35rem; margin: 2rem 0 .7rem; border-bottom: 1px solid #666; padding-bottom: .35rem; }}
  a {{ color: {ACCENT}; }}
  code {{ font-family: ui-monospace, Consolas, monospace; font-size: .9em; background: #363636; padding: 0 .25em; border-radius: 3px; }}
  .meta {{ color: {MUTED}; margin: 0 0 1.1rem; font-size: .98rem; }}
  .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(9rem, 1fr)); gap: .6rem; margin: 1rem 0; }}
  .kpi {{ background: {PANEL_BG}; border: 1px solid #444; border-radius: .5rem; padding: .55rem .7rem; }}
  .kpi span {{ display: block; color: {MUTED}; font-size: .88rem; }}
  .kpi strong {{ font-size: 1.55rem; font-variant-numeric: tabular-nums; }}
  .kpi em {{ color: {MUTED}; font-size: .88rem; font-style: normal; margin-left: .25rem; }}
  .card {{ background: {PANEL_BG}; border: 1px solid #444; border-radius: .5rem; padding: .5rem; overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; background: {PANEL_BG}; margin: .6rem 0; font-size: .96rem; }}
  th, td {{ border: 1px solid #555; padding: .5rem .6rem; text-align: left; vertical-align: top; }}
  thead th {{ background: #3d3d3d; }}
  table.kv {{ max-width: 48rem; }}
  table.kv th {{ width: 11rem; color: {MUTED}; font-weight: 500; }}
  details summary {{ cursor: pointer; color: {ACCENT}; margin: .4rem 0; font-size: 1rem; }}
  .note {{ color: {MUTED}; font-size: .96rem; }}
  .warn {{ color: #e0b685; font-size: 1rem; border-left: 3px solid #e0b685; padding-left: .6rem; }}
</style>
</head>
<body>
<header>
  <h1>{_esc(run_name)}｜世代統計</h1>
  <p class="meta">stats.json から生成。横軸は世代 (0 始まり)。判断ログは縦線と番号で示す。</p>
</header>
<section class="kpis">{"".join(kpis)}</section>
<section class="card">{chart}</section>
{f'<p class="note">{_esc(note)}</p>' if note else ''}
{valid_note}
{sigma_note}
{meta_html}
{decision_html}
<footer><p class="note">生成: <code>uv run python -m bodyopt stats-html --run &lt;run dir&gt;</code></p></footer>
</body>
</html>
"""


def write_report(target: str | Path, out: str | Path | None = None) -> Path:
    target = Path(target)
    stats_path = target if target.is_file() else target / "stats.json"
    run_dir = stats_path.parent
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    decisions = _load_decisions(run_dir, stats)
    html = render_html(stats, run_dir.name, meta, decisions)
    dest = Path(out) if out else run_dir / "stats.html"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(html, encoding="utf-8")
    return dest
