# 実行・検証・報告の運用

## ディレクトリ

| 場所 | 役割 |
|---|---|
| src/bodyopt/ | シミュレータ・GA・CLI (パッケージ) |
| params/ | ケース非依存の既定値 (default.json) |
| resources/runs/ | 実行ケースごとの生データ (上書き・自動削除しない) |
| tests/ | 静的テスト・tests/regression/ (回帰)・再現性 (slow) |
| docs/agents/ | 実装の詳細な正本 (本ディレクトリ、md) |
| docs/humans/ | 人間向けの要約・進捗の正本 (html) |
| docs/report/ | 数値実験の報告 (html) |
| docs/reviews/ | src 実装の review (html、template.html を複製) |
| viewer/ | three.js 軌跡ビューア |
| scripts/ | 報告用の軌跡データ生成など |

## コマンド

```bash
uv sync                                   # 依存の導入 (jax[cuda13]・Python 3.12)
uv run pyright                            # 型チェック (src + tests + scripts)
uv run pytest -q                          # 静的テスト (pyright の 0 件チェックを含む)
uv run pytest -q -m regression            # 積分法の回帰比較
uv run pytest -q -m slow                  # 再現性 (短い GA run)
uv run python -m bodyopt evolve --generations 10 --seed 0 \
    --record summary --out resources/runs/<name> \
    [--config params/default.json]   # 物理・GA・身体の設定 (--config の既定値)
uv run python -m bodyopt replay --run resources/runs/<name> --generation 9 --rank 0 [--out <dir>]
uv run python -m bodyopt stats-html --run resources/runs/<name>   # 同じディレクトリに stats.html
uv run python -m bodyopt chaos --run resources/runs/<name> --repeats 20 [--generation 99] [--json]
uv run python -m bodyopt viewer --port 8765   # http://127.0.0.1:8765/viewer/ (replay API 付き)
uv run python scripts/embed_report_trajectory.py <trajectory.json> docs/report/<file>.html --stride 6
```

## run と commit

run の再現単位は commit sha とする。 meta.json  の `git.commit` がその run を走らせたコードを指し、run 同士の比較は sha が一致する範囲でだけ意味を持つ。

### commit のタイミング

| タイミング | すること |
|---|---|
| evolve の直前 |  src/   params/   tests/  を commit し、作業ツリーを clean にする |
| 比較 run の間 | commit しない。seed だけ変えて同じ sha で回す |
| chaos・replay | evolve と同じ sha で行う |
| run の直後 | その sha に `git tag run-<name>` を打つ |
| run の後 | docs/report の更新は別 commit にする |

- 規則は「run ごとに commit」ではなく「dirty のまま run しない」である。コードが前回の run から変わっていなければ commit は不要で、その run は既存の sha を指す。
- run の前に commit するときは、コミットメッセージに run 名を最低限入れる (例: `run-009: 接触ばねを柔らかくする`)。同じ sha を複数の run が使う場合は、その commit を必要とした最初の run 名でよい。
- 部分 commit はしない。run の sha はツリー全体を指すため、viewer だけを commit した sha に checkout しても当時の実行状態にはならない。
- tag は gc 対策である。rebase や amend で sha が到達不能になると、run からコードを辿れなくなる。
- 既存の 001〜008 は `dirty: true` で、sha が当時のコードを指していない。比較の基準には使わない。

### run の手順

1. `git status --porcelain` が空か確かめる。空でなければ commit する。
2. `evolve` を回す。
3.  meta.json  の `git.dirty` が `false` か確かめる。
4. `stats-html` を生成する (run が正常終了したら原則として必ず)。
5. `git tag run-<name>` を打つ。
6. `chaos` で最良個体のばらつきを測る。
7. 必要な個体を `replay` して viewer で見る。

run 名は  resources/runs/<name>/  のディレクトリ名にしかない。追跡は「ディレクトリ名 → meta.json の sha」の順で行う。`replay` と `chaos` は meta.json の `git` をまだ照合しないので、手順 3 が現状で唯一の担保である。

## 運用ルール

1. 実装後は「静的テスト → 短い GA run による動的テスト → 合意仕様の docs 正本化」を徹底する。first goal の段階では長い GA を回さない (数分で終わる動作確認まで)。
2. 新しい実験設定 (params・身体・物理・評価式) で run を回したら、結果を採用する前に最良個体で `chaos` を測る。同じ条件の反復シミュレーションで距離の幅が max(5 mm, 平均の 1%) を超えたら、カオスが結果に効いているとみなす。その場合は report に幅と `dt` を明記し、`dt` を細かくする・減衰を見直す・設定間の比較を平均と分布で行う、のいずれかで対処する。旧形式の run は遺伝子長が合わず測れないので、当時のコードで測る。
3. run が正常終了したら、原則として `uv run python -m bodyopt stats-html --run resources/runs/<name>` を実行し、`stats.html` を残す。異常終了や検証用の極小 run では省略してよい。
4. 新機能や数値実験のたびに docs/report/ に HTML 報告を書く。`docs/report/template.html` を複製し、ファイル名は `YYYY-MM-DD_topic.html` とする。見出しは「目的・追加機能・実験設定・資源と時間・主要な結果・分析・todo との照合」で、目次と完全に一致させる。項目ごとの分量は回の重点に合わせて変えてよい。
5. 報告 HTML に軌跡データを埋め込み、Canvas で個体の動きを再生する。再生・一時停止ボタンと水平シークバーを図の下に置く。サーバーや動画コーデックは不要。
6. src の実装 review を定期的に行い docs/reviews/ に置く。ファイル名は `standard_YYYY-MM-DD.html`。書式は docs/reviews/template.html を複製する。対応が完了した review は docs/reviews/archive/ へ移動し、script のパスを ../../open/assets/html-state.js に直す。
7. 人間が操作する HTML は docs/open/assets/html-state.js で状態を保存する (祖先に data-persist、末尾に script)。
8. GA の計算は世代ループ内で CPU 同期を入れない。同期は run の最後の 1 回だけ。
9. 仕様が複雑になったら、本ディレクトリを内容別のサブディレクトリに分割する。AGENTS.md には参照リンクだけを置く。
10. 参照の書き方 (Ctrl+クリックで開けるようにする):
   - md では `[@docs/agents/sim/physics.md](../sim/physics.md)` の形にする (表示は @docs 表記のまま)。
   - html では `<a href="../agents/sim/physics.md">@docs/agents/sim/physics.md</a>` の形にする。VS Code での遷移は拡張が処理するため、`target` 属性は不要。
   - コードブロック・インラインコード・文字列リテラルの中の @docs はリンク化しない。
11. VS Code で preview のリンクをエディタタブで開く:
   - この環境では、既定の「Preview First」(`local.preview-first`) が HTML と Markdown の相対リンクを VS Code のタブで開く。Ctrl+クリック時も、別のエディタグループを含め、開いているタブがあればそのタブへ移動する。拡張を更新したときは VS Code で Reload Window を実行する。
   - 手動で再読み込みするには、プレビューのエディタ右上のリロードアイコンを押す。ファイル保存時は自動で再描画される。
   - `bodyopt doclinks` はこの PC の VS Code ユーザー拡張で、リポジトリには含めない。Preview First を使わない場合は、HTML の右クリックメニューから「bodyopt: HTML プレビューを開く」を選ぶ。VS Code 標準の Markdown プレビューには `markdown.preview.openMarkdownLinks: inEditor` を使う。
