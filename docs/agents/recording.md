# 記録・再現・再生 (src/bodyopt/io.py)

## 記録レベル

| レベル | 保存内容 |
|---|---|
| summary | meta.json (再現メタデータ) + stats.json (世代統計・各世代の最良遺伝子) |
| population | summary + population.npz (全世代の遺伝子・スコア・有効フラグ・付随統計) |
| full | population + 最良個体の完全ロールアウト (trajectory_*.json) |

- meta.json には git commit hash・dirty フラグ・JAX バージョン・デバイス (VRAM ピーク含む)・全パラメータ・実行コマンド・elapsed・RSS ピーク、creature の完全な定義 (座標・辺・自然長・質量・剛性・mask) を記録する。
- 軌跡は遺伝子 + commit + params で再現できることを前提に、full では代表個体の生データのみ保存する。
- resources/runs/<name>/ に実行単位で保存し、上書き・自動削除はしない。
- replay は meta.json の creature から身体を復元する (実行時の --n-cols に依存しない)。旧形式 meta にもフォールバックする。

## viewer

- viewer/index.html は three.js で軌跡 JSON を事後再生する (2D 物理を 3D 視点で表示、視点移動・張力色・接触点・重心軌跡)。
- 注意: viewer の ladderEdges は src/bodyopt/creature.py の辺順序を再実装している。接続規則を変えたら viewer 側も更新する (軌跡 JSON に辺情報を含める改修が望ましい)。
- 報告 HTML の動画は scripts/make_report_video.py が軌跡 JSON から WebM (VP9) を生成し、data URI で HTML に埋め込む。VP9 は Chromium 組み込みで preview (webview) でも再生できる。H.264 の <video> は preview で再生できないことがあり、APNG では再生・停止・シークができない。python -m http.server は Range 非対応のため、src 参照ではなく data URI 埋め込みを既定とする。
