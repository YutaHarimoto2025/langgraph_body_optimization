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
| scripts/ | 報告用の生成ツール (mp4) |

## コマンド

```bash
uv sync                                   # 依存の導入 (jax[cuda13]・Python 3.12)
uv run pyright                            # 型チェック (src + tests + scripts)
uv run pytest -q                          # 静的テスト (pyright の 0 件チェックを含む)
uv run pytest -q -m regression            # 積分法の回帰比較
uv run pytest -q -m slow                  # 再現性 (短い GA run)
uv run python -m bodyopt evolve --generations 10 --seed 0 \
    --record population --out resources/runs/<name>
uv run python -m bodyopt replay --run resources/runs/<name> --generation 9 --rank 0
uv run python scripts/make_report_video.py <trajectory.json> docs/report/assets/<name>.mp4 \
    --stride 6 --fps 30 --label "..." --html docs/report/<file>.html
```

## 運用ルール

1. 実装後は「静的テスト → 短い GA run による動的テスト → 合意仕様の docs 正本化」を徹底する。first goal の段階では長い GA を回さない (数分で終わる動作確認まで)。
2. 新機能や数値実験のたびに docs/report/ に HTML 報告を書く。ファイル名は `YYYY-MM-DD_topic.html` とする。項目は「目的・追加機能・実験設定・RAM/VRAM ピークと計算時間・主要結果・分析と先行研究との差異・todo との照合」。全項目を同じ詳細度で書かず、その回で新規なことに詳細を割く。
3. 個体の動きは mp4 (H.264) を `<video controls loop muted playsinline>` で埋め込む。src パス参照は Range 非対応サーバーで再生できないため、data URI で HTML に埋め込む。
4. src の実装 review を定期的に行い docs/reviews/ に置く。ファイル名は `standard_YYYY-MM-DD.html`。書式は docs/reviews/template.html を複製する。対応が完了した review は docs/reviews/archive/ へ移動し、script のパスを ../../open/assets/html-state.js に直す。
5. 人間が操作する HTML は docs/open/assets/html-state.js で状態を保存する (祖先に data-persist、末尾に script)。
6. GA の計算は世代ループ内で CPU 同期を入れない。同期は run の最後の 1 回だけ。
7. 仕様が複雑になったら、本ディレクトリを内容別のサブディレクトリに分割する。AGENTS.md には参照リンクだけを置く。
