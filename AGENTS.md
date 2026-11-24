# AGENTS.md

## プロジェクト概要

2D の質点・バネ身体（mass-spring creature）の形態と筋肉制御を最適化する実験プロジェクト。最適化ループ自体を LangGraph で構造化し、LLM を「設計者」または「最適化戦略の立案者」として使う。目的は最強の人工生命 simulator ではなく、**数値最適化・LLM 判断・失敗処理・戦略変更をどこまできれいに構造化・観測できるか**の検証にある。

背景・設計案は  docs/open/0912_langgraph_artificial_body_optimization_idea.md  に集約されている。実装判断で迷ったらこの文書を一次ソースとする。

## 現状

first goal（LangGraph・LLM なしの純粋 GA による周期制御最適化）は実装済みで、動作テストまで通っている。実装仕様の正本は  docs/agents/SPEC.md 、人間向けの要約は  docs/humans/spec.html  にある。実験の報告は  docs/report/ に、実装 review は  docs/reviews/ に蓄積する。docs/open/ は履歴・構想の参考資料である。

### 構成

- 身体モデル: 2D mass-spring、`q_i` `m_i`、伸縮エッジ `k_e` `L_e` `a_e` `phi_e` `omega_e`（first goal では `a_e` `phi_e` のみ遺伝子）
- simulator: JAX による独自の質点・バネ実装（重力・地面接触・摩擦）。半陰的オイラー法第一候補、RK4 は比較用
- 最適化: 純粋 GA（トーナメント・一様交叉・ガウス変異・エリート継承）。LLM 監督・LLM による optimizer 設計は first goal 達成後の比較対象
- オーケストレーション: LangGraph は未導入（first goal の範囲外。導入時に docs/humans/spec.html の LangGraph 節を更新する）

## ディレクトリ構成

```
docs/
  open/   アイデア・設計メモ（日付 prefix: MMDD_topic_idea.md）
    assets/html-state.js  HTML の入力状態を保存する共通スクリプト
  agents/   実装仕様の正本（agent 向け・md）。現在は SPEC.md
  humans/   人間向けの仕様と進捗の正本（html）
  report/   数値実験の報告（html、全項目に言及、新規事項を厚く）
  reviews/  src 実装の review（aisin の規約 + 計算効率・アルゴリズム・再現性）
    template.html  書式の雛形（複製して使う）
    archive/       対応が完了した review
src/        Python パッケージ bodyopt（creature・physics・fitness・ga・io・cli）
params/     ケース非依存の既定値（default.json）
resources/  実行ケースごとの生データ（runs/<name>/。上書き・自動削除しない）
tests/      静的テスト・regression/（積分法回帰）・再現性（slow マーカー）
viewer/     three.js による軌跡ビューア（事後再生）
.vscode/    エディタ配色設定
opencode.jsonc  opencode の権限設定
```

## 運用ルール

1. 実装後は「静的テスト → 短い GA run による動的テスト → 合意仕様の docs 正本化」を徹底する。first goal の段階では長い GA を回さない（数分で終わる動作確認まで）。
2. 新機能や数値実験のたびに  docs/report/ に HTML 報告を書く。項目は「目的・追加機能・実験設定・RAM/VRAM ピークと計算時間・主要結果・分析と先行研究との差異・todo との照合」。全項目を同じ詳細度で書かず、その回で新規なことに詳細を割く。個体の動きは mp4 (H.264) を `<video controls loop muted playsinline>` で埋め込む (生成は scripts/make_report_video.py)。
3. src の実装 review を定期的に行い  docs/reviews/ に置く（report とは別ファイル）。書式は  docs/reviews/template.html  を複製する。対応が完了した review は  docs/reviews/archive/  へ移動し、script のパスを  ../../open/assets/html-state.js  に直す。
4. GA の計算は世代ループ内で CPU 同期（`int()`・`float()`・`np.asarray`・`print`）を入れない。同期は run の最後の 1 回だけ。

## ドキュメント規約

- 本文は日本語で書く。
- 表示数式は `$$ ... $$`、インライン数式は `$ ... $` を使う。`\[ ... \]` は使わない。
- 数式では変数に一貫した記号を割り当てる。例: 位置 `q_i`、質量 `m_i`、エッジ剛性 `k_e`、自然長 `L_e`、収縮振幅 `a_e`、位相 `phi_e`、周波数 `omega_e`、設計変数 `theta`。
- ファイル名は `MMDD_topic_idea.md` 形式（例: `0912_langgraph_artificial_body_optimization_idea.md`）。
- 箇条書きは真に並列な列挙に限定し、説明は地の文で書く。

## HTML ドキュメントの操作状態

人間が操作する HTML（チェック、ラジオ、テキスト入力、選択）を作るときは、状態の保存を必須とする。共通スクリプト  docs/open/assets/html-state.js  を読み込めば、`data-persist` を付けた要素の状態がブラウザの localStorage に保存され、再読み込み時に復元される。`file://` で開いた場合も同じように動く。

- 入力の祖先（通常は `form`）に `data-persist` を付ける。無い場合は `form` が対象になる。
- HTML の末尾に `<script src="assets/html-state.js" defer></script>` を置く。`src` は HTML からの相対パスで書く。
- reset ボタンは保存内容も消す。状態だけ初期化したい場合はページを再読み込みする。
- 保存先はブラウザのプロファイルであり、別ブラウザ・別端末・シークレットウィンドウとは共有されない。git にも入らない。
- 記録をリポジトリに残す必要があるときは、状態を JSON で書き出すボタンを HTML 側に追加する。保存処理を各 HTML に複製しない。

## 開発方針

- 目的は探索の構造化と観測であり、物理の高忠実度化ではない。simulator は単純さを優先する。
- 評価関数は最初は移動距離 `f = x_COM(T) - x_COM(0)` のみとし、reward hacking が観測されてからエネルギー・破損・転倒の罰則を追加する。
- 拡張は  docs/open/0912_langgraph_artificial_body_optimization_idea.md  の Level 1〜6 の順で段階的に行う。

## セキュリティ・権限

- `.env`、`*.pem`、`*.key`、`*.sqlite` 等は読み書き不可（opencode.jsonc で deny 済み）。
- `.git/` 配下は直接編集せず、git コマンド経由で操作する。

## 開発コマンド

```bash
uv sync                                   # 依存の導入 (jax[cuda13]・Python 3.12)
uv run pyright                            # 型チェック (src + tests)
uv run pytest -q                          # 静的テスト
uv run pytest -q -m regression            # 積分法の回帰比較
uv run pytest -q -m slow                  # 再現性 (短い GA run)
uv run python -m bodyopt evolve --generations 10 --seed 0 \
    --record population --out resources/runs/<name>     # GA (headless)
uv run python -m bodyopt replay --run resources/runs/<name> \
    --generation 9 --rank 0                              # 保存個体の再ロールアウト
```

実行の生データは  resources/runs/<name>/  に meta.json (commit hash・seed・全パラメータ) とともに保存され、上書きしない。first goal の段階では数分で終わる run までを検証範囲とする。
