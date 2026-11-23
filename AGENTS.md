# AGENTS.md

## プロジェクト概要

2D の質点・バネ身体（mass-spring creature）の形態と筋肉制御を最適化する実験プロジェクト。最適化ループ自体を LangGraph で構造化し、LLM を「設計者」または「最適化戦略の立案者」として使う。目的は最強の人工生命 simulator ではなく、**数値最適化・LLM 判断・失敗処理・戦略変更をどこまできれいに構造化・観測できるか**の検証にある。

背景・設計案は  docs/open/0912_langgraph_artificial_body_optimization_idea.md  に集約されている。実装判断で迷ったらこの文書を一次ソースとする。

## 現状

アイデア段階であり、実装コード・ビルド設定・テストはまだ存在しない。したがって lint / typecheck / test のコマンドは未確定である。コードを追加する際は、その PR 内で実行方法（`uv` / `pytest` 等）を決め、このファイルの「開発コマンド」を更新する。

### 予定している構成（未確定）

- 身体モデル: 2D mass-spring、`q_i` `m_i`、伸縮エッジ `k_e` `L_e` `a_e` `phi_e` `omega_e`
- simulator: 独自の簡易質点・バネ実装（重力・地面接触・摩擦）
- 最適化: GA を基本、LLM 監督・LLM による optimizer 設計を比較
- オーケストレーション: LangGraph、LangSmith Studio で trace

## ディレクトリ構成

```
docs/
  open/   アイデア・設計メモ（日付 prefix: MMDD_topic_idea.md）
  agents/ （用途未定）
.vscode/  エディタ配色設定
opencode.jsonc  opencode の権限設定
```

## ドキュメント規約

- 本文は日本語で書く。
- 表示数式は `$$ ... $$`、インライン数式は `$ ... $` を使う。`\[ ... \]` は使わない。
- 数式では変数に一貫した記号を割り当てる。例: 位置 `q_i`、質量 `m_i`、エッジ剛性 `k_e`、自然長 `L_e`、収縮振幅 `a_e`、位相 `phi_e`、周波数 `omega_e`、設計変数 `theta`。
- ファイル名は `MMDD_topic_idea.md` 形式（例: `0912_langgraph_artificial_body_optimization_idea.md`）。
- 箇条書きは真に並列な列挙に限定し、説明は地の文で書く。

## 開発方針

- 目的は探索の構造化と観測であり、物理の高忠実度化ではない。simulator は単純さを優先する。
- 評価関数は最初は移動距離 `f = x_COM(T) - x_COM(0)` のみとし、reward hacking が観測されてからエネルギー・破損・転倒の罰則を追加する。
- 拡張は  docs/open/0912_langgraph_artificial_body_optimization_idea.md  の Level 1〜6 の順で段階的に行う。

## セキュリティ・権限

- `.env`、`*.pem`、`*.key`、`*.sqlite` 等は読み書き不可（opencode.jsonc で deny 済み）。
- `.git/` 配下は直接編集せず、git コマンド経由で操作する。

## 開発コマンド

未定（実装開始時に追記する）。
