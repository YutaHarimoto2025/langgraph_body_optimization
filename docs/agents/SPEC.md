# bodyopt 実装正本 (agent 向け)

対象: src/ のシミュレータ・GA・記録。人間向けの要約は docs/humans/spec.html。
この文書が実装の詳細な正本で、実装と乖離したらどちらを直すか利用者に確認する。
履歴 (docs/open/) は参考資料であり、本正本が優先する。

## 1. first goal の範囲

LangGraph・LLM・feedback 制御 (脳) を導入せず、**固定形状の 2D 質点・バネ身体に対して周期制御のみを遺伝子とする純粋 GA** が動くことを目標とする。完了条件は次のとおり。

1. 静的テスト・積分法回帰テスト・短い GA run での再現性テストが通る。
2. 数分で終わる GA run (既定 10 世代 × 20 個体) で初期世代から best score が改善する。
3. 保存した個体を同じ物理コードで再ロールアウトして再生できる。

最適化対象の拡張 (形状・トポロジー・ノード数) は first goal 達成後に段階的に行う (docs/open/0912_...md の Level 1→6)。

## 2. 身体モデル

- 三角補強ラダー: 上下 2 列 × 3 点 = 6 質点、横 4 + 縦 3 + 対角 2 = 9 辺 (src/bodyopt/creature.py の triangulated_ladder)。
- 列間隔 0.5 m、行間隔 0.4 m、下列は床 y=0 から 0.05 m 上。全質点 1 kg、全辺剛性 100 N/m。
- 基準自然長は初期辺長に一致 (初期張力ゼロ)。
- 全 9 辺を能動筋肉として扱う (muscle_mask)。受動バネとの区別は mask で表現し、駆動する辺の指定は設定で変えられる。

## 3. 遺伝子

$$\theta = (\{a_e\}_{e \in M},\ \{\phi_e\}_{e \in M})$$

- 振幅 $a_e \in [0, 0.20]$ (MAX_AMPLITUDE、常に課すハード上限)。
- 位相 $\phi_e \in [0, 2\pi)$ (周期折返し)。
- 周波数は全筋肉共通 $\omega = 2\pi$ rad/s (1 Hz) で遺伝子外。個別周波数・剛性・質量は first goal の対象外。
- 自然長の時間発展:

$$\ell_e^{\mathrm{rest}}(t) = L_e (1 + a_e \sin(\omega t + \phi_e))$$

## 4. 物理 (src/bodyopt/physics.py)

### 4.1 積分

第一候補は半陰的オイラー法 (既定 dt = 5 ms、episode 5 秒 = 1000 ステップ)。RK4 を比較候補として同ファイルに実装し、tests/regression/test_integration_scheme.py で次を検証する。

- 同一遺伝子で dt 5 ms ↔ 2.5 ms、semi_implicit_euler ↔ RK4 の distance 差が 0.05 m 未満。
- 有効判定と代表 3 個体の順位が一致。
- 許容差は暫定値であり、差が出たら刻み細化を第一候補とする。

### 4.2 床接触と摩擦

- めり込みに対するペナルティ反力 (法線は常に +y、2D 平面床の特殊ケース):

$$F_i^{N} = \max(0,\ k_c \max(0, -y_i) - c_c \min(0, \dot y_i))$$

- 滑らかなクーロン摩擦 ($\operatorname{sgn}$ を $\tanh$ で近似、静止摩擦は再現しない):

$$F_i^{f} = -\mu F_i^{N} \tanh(\dot x_i / v_\epsilon), \quad v_\epsilon = 0.05\ \mathrm{m/s}$$

### 4.3 3D 拡張を見据えた一般性

- 質点座標は `[N, dim]` で書き、バネ力に 2D 固有の処理を入れない。
- 接触は「床の判定 + 法線反力」と「接線摩擦」を分けて実装する。3D 化では床を符号付き距離関数 (SDF) + 法線に差し替え、摩擦は接平面へ射影する。
- 自己衝突・凹凸地面・坂は first goal の対象外。

### 4.4 計算効率 (5-4 の合意)

- CPU 版シミュレータは作らない。単体物理を jax で書き、集団軸は vmap、時間軸は lax.scan、世代軸も run_ga 内で 1 つの jit + scan に載せる。
- 世代ループ内の CPU 同期 (int()・float()・np.asarray・print) を禁止する。run_ga の戻り値を tree.map(np.asarray) する最後の 1 回だけ同期する。
- full 記録で GPU メモリが不足する場合のみ、複数世代ずつまとめて転送・保存する (初回規模では発生しない)。

## 5. 評価 (src/bodyopt/fitness.py)

$$f = x_{\mathrm{COM}}(T) - x_{\mathrm{COM}}(0), \quad x_{\mathrm{COM}}(t) = \frac{1}{M} \sum_i m_i x_i(t)$$

- 無効条件: 非有限値、最大めり込み > 0.10 m、(オプション) 駆動仕事が予算超過。
- 無効個体のスコアは INVALID_SCORE = -1e6 (有効個体より常に下位)。距離の罰則への混合はしない。
- 駆動仕事 (エネルギー消費の代理指標、常に記録):

$$E_{\mathrm{drive}} = \int_0^T \sum_{e \in M} \max\!\left(0,\ -F_e(t)\, L_e\, a_e\, \omega \cos(\omega t + \phi_e)\right) dt$$

張った筋肉をさらに縮めるときだけ正の仕事と数え、回生は数えない。energy_budget (None = 無制限) を設定した場合だけ無効判定に使う。first goal は無制限で走らせ、E_drive を距離と並べて報告する。

## 6. GA (src/bodyopt/ga.py)

- 集団 20、エリート 2 (継承・再評価なし)、トーナメント選択 k=3、一様交叉 (遺伝子単位スイッチ確率 0.8)、突然変異率 0.10/遺伝子、変異 sigma は範囲 × 0.05。
- 再現性の限界: バネ力の組立に使う scatter-add は GPU 上で atomic 演算になり、加算順序が実行ごとに変わるため float32 最下位ビットが非決定になり得る。同じ seed・同じ実行環境では相対 1e-5 で一致する (tests/test_reproducibility.py) をもって再現とする。ビット一致を要求する場面では CPU 実行か決定的スキャッタの採用を別途検討する。
- 構成は trajectory-solver-generic の brkga.py を参考に「評価は注入関数・集団更新は JAX 演算」の分離を踏襲。BRKGA の biased crossover は採用しない (first goal は合意どおり標準 GA)。
- evolve_population は (pop, fit, key) → (pop, key) の純関数。island model への拡張は、この関数を島ごとに vmap し移住演算を追加する形で行う (初回から実装しない)。
- 世代ループは run_ga 内で lax.scan。初期集団・交叉・変異すべて jax.random のキー分割で行い、Python 側の乱数を使わない。

## 7. 記録 (src/bodyopt/io.py) とディレクトリ

| 記録レベル | 保存内容 |
|---|---|
| summary | meta.json (再現メタデータ) + stats.json (世代統計・各世代の最良遺伝子) |
| population | summary + population.npz (全世代の遺伝子・スコア・有効フラグ・付随統計) |
| full | population + 最良個体の完全ロールアウト (trajectory_*.json) |

- meta.json には git commit hash・dirty フラグ・JAX バージョン・デバイス (VRAM ピーク含む)・全パラメータ・実行コマンド・elapsed・RSS ピークを記録する。
- 軌跡は遺伝子 + commit + params で再現できることを前提に、full では代表個体の生データのみ保存する。
- resources/runs/<name>/ に実行単位で保存し、上書き・自動削除はしない。
- 注意: viewer/index.html の ladderEdges は src/bodyopt/creature.py の辺順序を再実装したものである。creature.py の接続規則を変えたら viewer 側も同じ順序に更新すること (軌跡 JSON に辺情報を含める改修が望ましい)。

```
src/bodyopt/     シミュレータ・GA・CLI ( パッケージ )
params/          ケース非依存の既定値 (default.json)
resources/runs/  実行ケースごとの生データ
tests/           静的テスト・tests/regression/ (回帰)・再現性 (slow)
docs/agents/     本正本 (md)
docs/humans/     人間向けの要約・進捗 (html)
docs/report/     数値実験の報告 (html、全項目に言及、新規事項を厚く)
docs/open/       履歴・構想 (参考資料、正本ではない)
```

## 8. 実行コマンド

```bash
uv sync                                   # 依存の導入 (jax[cuda13])
uv run pytest -q                          # 静的テスト
uv run pytest -q -m regression            # 積分法回帰
uv run pytest -q -m slow                  # 再現性 (短い GA run)
uv run python -m bodyopt evolve --generations 10 --seed 0 \
    --record population --out resources/runs/first_try
uv run python -m bodyopt replay --run resources/runs/first_try \
    --generation 9 --rank 0
```

first goal では長い GA を回さない。数分で終わる動作確認 (既定 10 世代 × 20 個体 × 5 秒 episode) までを検証範囲とする。

## 9. 運用ルール (合意)

1. 実装後は静的テスト → 短い GA run による動的テスト → 合意仕様の docs 正本化を徹底する。
2. 新機能や数値実験のたびに docs/report/ に HTML 報告を書く (項目: 目的・追加機能・実験設定・RAM/VRAM ピークと計算時間・主要結果・分析と先行研究との差異・todo との照合)。全項目を同じ詳細度で書かず、その回で新規なことに詳細を割く。
3. src の実装 review を定期的に行う。書式は aisin の docs/reviews テンプレート (C/R 番号・層バッジ・自動修正可/利用者判断) に、計算効率・アルゴリズム改善・再現性チェック (軽い run) を加える。
4. review は docs/report/ とは別ファイルとする。

## 10. 拡張ロードマップ (first goal 後)

- 形状探索 (初期座標を遺伝子に追加、接続固定) → トポロジー → ノード数可変 (docs/open/0912 の Level 2-3)。
- LangGraph 導入: 固定 GA を基準に、停滞診断・戦略変更の分岐を追う。
- feedback 制御: 周期制御を基準に、状態入力の制御器 (Rojas 2025 の attention 共有制御を参考) を比較。
- Three.js ビューアの強化 (分析パネル・複数個体比較・shadcn/ui)。
