# 記録・再現・再生 (src/bodyopt/io.py)

## 記録レベル

| レベル | 保存内容 |
|---|---|
| summary | meta.json (再現メタデータ) + stats.json (世代統計・各世代の最良遺伝子) |
| full | summary + population.npz (各世代の遺伝子・スコア・有効フラグ・付随統計)。既定は summary |


- meta.json には git commit hash・dirty フラグ・JAX バージョン・デバイス (VRAM ピーク含む)・全パラメータ・実行コマンド・elapsed・RSS ピーク、creature の完全な定義 (座標・辺・自然長・質量・剛性・mask) を記録する。
- `git.commit` はその run を走らせたコードを指す唯一の参照である。dirty の run は sha でコードを再現できない。run 前の clean 強制とタグの運用は [@docs/agents/operations.md](operations.md) の「run と commit」に従う。
- 軌跡は保存しない。replay で遺伝子 + commit + params から必要なときに再生成する (full レベルは廃止)。
- resources/runs/<name>/ に実行単位で保存し、上書き・自動削除はしない。
- replay は meta.json の creature から身体を復元する (実行時の --n-cols に依存しない)。旧形式 meta にもフォールバックする。
- replay の軌跡は定性再生用。接触系はカオス的で、同じ遺伝子でも評価の文脈が変わると距離が数十 % 変わるため、記録スコアの再現は保証しない (詳細: [@docs/agents/sim/physics.md](sim/physics.md) の「決定性と再現性の限界」)。
- replay は `--episode` (viewer の「秒」) で保存時より長い区間を計算できる。trajectory JSON には実際の episode 長を記録する。
- replay は population.npz があれば任意の --rank を再生できる。summary 記録には population.npz が無いため、stats.json の各世代の最良遺伝子を使い --rank 0 のみ再生する (それ以外はエラー)。

## stats.json と stats.html

- stats.json の per_generation は best_score・mean_score・valid_rate・best_distance・mean_distance・n_valid・score_dist・distance_dist・energy_dist・gene_sigma・best_energy・mean_energy・best_gene を持つ。
- score_dist・distance_dist は有効個体の p10・p50・p90・std、energy_dist は全個体の p50・p90・std。空の世代は null。
- gene_sigma は遺伝子の種類 (amplitude・frequency・phase) ごとに、選択群 top1・top10・all の標準偏差を世代ごとに持つ。選択群はスコア降順の上位 1%・10%・全個体で、その群の同種の全筋肉・全次元をまとめて np.std する。無効個体も順位どおり含める。上位 1% が 1 個体のときは、その個体のエッジ間のばらつきになる。
- 保存時には常に集団があるため、summary 記録でも gene_sigma は書ける。ただし旧形式の stats.json には無く、population.npz が無い run からは復元できない。
- 有効個体は、位置・速度が有限、最大めり込み < max_penetration (既定 0.10 m)、(設定時) 駆動仕事 ≤ energy_budget。無効個体のスコアは -1e6。
- `uv run python -m bodyopt stats-html --run resources/runs/<name>` で同じディレクトリに stats.html を作る。stats.json だけを読み、meta.json があれば実行設定を併記する。依存は標準ライブラリのみで、SVG を直接書き出す。
- パネルは評価スコア (最良・平均、p10–p90 の帯)、重心移動距離 (スコアと異なる場合のみ)、振幅・周波数・位相の σ (上位 1%・10%・全個体)、有効率、駆動仕事 (最良・平均)。無効個体のスコア (-1e6) はスコアと距離のパネルから除外する。
- stats.html が必要とする値はすべて stats.json に保存時に計算する。summary 記録でも population.npz を読まずにフル表示できる。
- エージェントの判断ログは stats.json の `decisions`、または同じディレクトリの decisions.json に置く。要素は `{"generation": 120, "label": "突然変異率を上げる", "detail": "10 世代停滞"}`。`end_generation` を足すと期間の帯になる。図には縦線と番号が入り、下の表に内容が出る。
- stats.html は再生成するたびに上書きしてよい。数値の正本は stats.json に置き、HTML を直接編集しない。

## viewer

- viewer/index.html は three.js で軌跡 JSON を事後再生する (2D 物理を 3D 視点で表示、視点移動・接地質点の着色・重心軌跡)。操作は右ドラッグ=回転、ホイール=ズーム、中ボタンドラッグ=平行移動。
- パネルは「インポート」「再生」「個体情報」の大見出しで分ける。インポートは replay と軌跡 JSON のパスを小見出しに持つ。個体情報は、固定情報 (label・重心最大移動距離 (エピソード終端)・後端最大移動距離 (エピソード終端)・フレーム数・dt)、その時刻の状態 (時刻・重心 x/y・重心移動距離・後端移動距離・接触質点数・累積駆動仕事)、遺伝子グラフの順に並べる。固定情報の最大値はエピソード終了までで打ち切り、replay を延長した区間は含めない。移動距離は t=0 の重心・後端との差。viewer の重心は単純平均で計算する (現行の身体は全質点の質量が等しいため質量加重と一致)。GA のスコアはエピソード終了フレームの重心変位であり、最大値とは一致しない場合がある。
- 床は y=0 の 40 m 四方の不透明な灰色の平面に、1 m の細線と 5 m の太線を引いた白い格子を描く。5 m 四方の CanvasTexture (1024 px、太線 5 px・細線 2 px) を RepeatWrapping し、線の太さを距離によらず一定に保つ。旧 GridHelper は廃止した。
- 接地中の質点は球そのものをオレンジ (`#ff8c1a`) で塗る。着地の瞬間に広がるリング (波動状のエフェクト)、接触点マーカー、表示の ON/OFF は置かない。
- 辺の色は自然長からのひずみ (L - L_rest)/L_rest で決める。自然長=白灰、圧縮=赤、伸長=青で、ひずみ ±30% で飽和する (STRAIN_FULL)。L_rest は trajectory JSON の rest_lengths (時変自然長) を使う。
- 辺は Line ではなく半径 EDGE_RADIUS (0.01) の円柱メッシュで描く。WebGL では LineBasicMaterial.linewidth が効かず 1 px にしかならないため。円柱は質点球 (半径 0.05) との前後関係も depth buffer で正しく解決される。
- 辺のジオメトリを毎フレーム setFromPoints で書き換える方式にしてはいけない。setFromPoints は既存の position を上書きするだけで boundingSphere を無効化しないため、初期値 (原点・半径 0) のまま視錐台カリングされ、視点によって辺が消える。円柱メッシュなら boundingSphere はジオメトリ由来で、matrixWorld の変換も効く。
- three.js は viewer/vendor/ に同梱する (three.module.js・three.core.js・addons/controls/OrbitControls.js)。CDN を参照しないので、VS Code の Preview First でもそのまま動く (webview の CSP が外部スクリプトを落とすため)。更新は手作業でファイルを差し替える。
- パス入力は絶対パスを受け付ける。webview の base href がファイルシステムのパスをそのまま持つので、ビューアのディレクトリからの相対 URL に写像して fetch する。`bodyopt viewer` 配信時はリポジトリルートの絶対パスをそのまま URL に変換する。相対パスはビューア基準で探し、404 ならリポジトリルート基準でも試す。
- JSON をパス指定で読み込んだ場合も、親ディレクトリ名を run 名としてラベルに補う (例: 008 / gen386 · rank0)。gen/rank はファイル名の `gen<g>_rank<r>` からそのまま取る (replay UI と同じ 0 始まり)。パス読み込みでもカメラは動かさない。
- 床は原点を左手前の角に合わせて x ∈ [0, 40], z ∈ [-40, 0] に置く (three.js は +z がカメラ手前なので、奥は -z)。初期位置 x = 5 のラインを幅 0.1 m のオレンジの太線で描く (スタートライン)。
- 複数個体を同時に読み込める。1 個目を (x, y, z) = (5, 0, -1)、以降を (5, 0, -2), (5, 0, -3), … と手前から 1.0 m 間隔で奥へ並べる。各レーンの xy 面 (z = レーンの位置) に「ケース名 / gen · rank」の 2 行ラベルを 3D の板ポリゴン (CanvasTexture + PlaneGeometry、高さ 0.52 m) で置き、下端を床に接地させる (看板状)。ラベルの x は個体の左 (グループ原点から -1 付近) に揃える。文字は白で黒の細い輪郭 (strokeText 3 px)、視点を回すとラベルも一緒に回る。
- 既定のカメラは床の手前の端 (原点の角の近く) に置き、斜めに奥を見る (VIEW_DIR)。R キーで戻す。読み込みではカメラを動かさない (個体を足すたびに視点が飛ぶのを避ける)。レーンが画面に収まらないときはホイールで引く。
- ラベルの左に × の削除ボタン (別の板ポリゴン) を置き、左クリックの raycast でその個体をビューから削除して残りを等間隔に詰め直す。文字をクリックすると統計表・遺伝子グラフの対象を切り替える。左下の接続状態は緑 (接続済み)・黄 (replay 実行中)・赤 (失敗/未接続) のドットで示し、URL や失敗理由はツールチップに出す。
- `uv run python -m bodyopt viewer` で viewer/ と replay API を配信する (src/bodyopt/viewer_server.py、依存は標準ライブラリのみ)。静的ファイルはリポジトリルート基準で配信する。
- replay は 1 リクエスト内の全個体を `jit(vmap(rollout_full))` でまとめて実行する。個体ごとにプロセスを分ける方式は JAX の起動とコンパイルが個体数分かかるため採らない。コンパイル結果は run ディレクトリ単位でキャッシュし、同じ run の 2 回目以降は 1 個体あたりの実行がほぼ 0 になる (実測: 40 個体の逐次 9.5 s に対しバッチ初回 0.4 s、2 回目 0.04 s)。
- replay は保存時のエピソード長と別の秒数で走らせられる。API の `seconds` (未指定なら保存時のエピソード長) を dt の整数倍に丸め、その長さでロールアウトする。軌跡 JSON には保存時のエピソード長を `episode` として書き、ビューアは `episode` を青い縦線 (スライダー上) で示す。
- 再生時間がエピソード長を超える場合、終了時刻より前は本体だけを表示する。終了時刻に達すると、その時刻に最も近い記録フレームの位置・姿勢・辺の色を同じレーンに固定表示し、本体は指定秒数まで動き続ける。質点の色は固定するフレームの接触状態で決める。終了時刻より前へ戻すと固定表示は消える。
- 延長再生では、スライダーの青い縦線の下に、同じ終了時刻の個体をまとめて操作するチェックボックスを置く。終了時刻に達すると自動でチェックが入る。チェックを外すと静止した個体だけが隠れ、終了時刻より前に戻ってから再び到達するまで、外した状態を保つ。
- replay API: `POST /_api/replay` に `{case, gens, ranks, seconds}` を送ると、replay を実行して `/tmp/bodyopt-viewer/trajectory_<case>_gen<g>_rank<r>.json` に書き、URL を返す。gens/ranks は「開始,終了,間隔」(例: gen `1,100,10` → 1, 11, ..., 91、rank `0,2` → 0..2) で、**rank は CLI と同じ 0 始まり (0 = 最良)**。開始 > 終了なら降順に展開し、読み込み順（手前のレーンから）もその順になる (例: `999,1,100` → 999, 899, ..., 99)。一時ファイルなので自動削除はしない。
- 1 個体の replay に失敗しても他の個体は読み込み、失敗理由を API の errors と UI に返す。summary 記録 (population.npz なし) は rank 0 のみ、gen が保存範囲外ならエラーになる。
- VS Code プレビュー (Preview First) からも replay できる。プレビューは `vscode-webview://` のページに HTML を埋め込むため、同一オリジンには API が無い。ビューアは `/_api/cases` が失敗したら `http://127.0.0.1:8765` に切り替える。API 側は CORS (`Access-Control-Allow-Origin: *`) と OPTIONS を返す。
- プレビューから接続するには、ローカル拡張 `~/.vscode/extensions/local.preview-first-0.1.0/extension.js` の CSP の `connect-src` に `http://127.0.0.1:8765` を追加する必要がある (リポジトリ外・この PC 限定の設定)。拡張を更新したら再適用し、VS Code で Reload Window する。API のポートを変える場合は、ビューアの `API_FALLBACK` と CSP の両方を合わせる。
- replay は evolve と同じ JAX デバイスを使う。他の JAX プロセスが VRAM を占有していると `CUDA_ERROR_OUT_OF_MEMORY` で失敗するので、その場合は `JAX_PLATFORMS=cpu` で viewer を起動する。
- 旧形式の run (遺伝子が 2 成分/筋肉) は現在の physics が 3 成分を前提とするため replay できない。ビューアの問題ではなく、古い run を再生するには当時のコードで replay する必要がある。
- 軌跡 JSON の埋め込みはしない (1 run あたり 1 MB 前後あり、ビューアが特定 run 専用になる)。サーバー不要で単一 HTML に閉じたい場合は docs/report/ の Canvas プレーヤーを使う。
- 注意: viewer の ladderEdges は src/bodyopt/creature.py の辺順序を再実装している。接続規則を変えたら viewer 側も更新する (軌跡 JSON に辺情報を含める改修が望ましい)。復元した辺数が rest_lengths の列数と違う場合は status に警告を出す。
- 報告 HTML の再生データは scripts/embed_report_trajectory.py で生成する。軌跡 JSON から位置・バネ力・接触の有無・時刻を取り出し、同じ run の meta.json から辺の接続を読み込む。結果は HTML の `<script type="application/json" data-trajectory-frames>` に埋め込む。
- 再生時は docs/report/assets/trajectory-player.js が保存時刻に合わせて Canvas を描画する。地面と全フレームの質点が収まる範囲を最初に決め、再生中は視点を固定する。操作は再生・一時停止・水平シーク。VS Code のプレビューで外部サーバーを立てずに使える。
