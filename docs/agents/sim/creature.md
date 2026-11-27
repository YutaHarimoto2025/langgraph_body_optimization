# 身体モデルと遺伝子

## 身体

- 両対角ラダー: 上下 2 列 × 3 点 = 6 質点、横 4 + 縦 3 + 対角 4 = 11 辺 (src/bodyopt/creature.py の triangulated_ladder)。各セルに両方向の対角を張る (run 006 から。以前は片方向のみの 9 辺)。
- 列間隔 0.5 m、行間隔 0.4 m、下列は床 y=0 から 0.05 m 上。全質点 1 kg、全辺剛性 500 N/m。無駆動時の重力による圧縮は辺長の 5% 程度 (上段は 0.45 → 0.37 m) で、形は保たれる。
- 基準自然長は初期辺長に一致 (初期張力ゼロ)。
- 全 11 辺を能動筋肉として扱う (muscle_mask)。受動バネとの区別は mask で表現し、駆動する辺の指定は設定で変えられる。muscle_idx は構築時に確定する (JIT 内で nonzero を呼ばないため)。

## 遺伝子

$$\theta = (\{a_e\}_{e \in M},\ \{f_e\}_{e \in M},\ \{\phi_e\}_{e \in M})$$

- $a_e$ は筋肉の振幅。単位と上限は `sim.muscle_model` に依存する (src/bodyopt/models.py の `gene_hi`)。
  - `active_force` (既定): 最大能動張力 $A_e \in [0, 100]$ N。
  - `rest_length`: 自然長の振幅比 $a_e \in [0, 0.20]$。
- $f_e$ は辺ごとの周波数 $\in [0.1, 10]$ Hz。
- 位相 $\phi_e \in [0, 2\pi)$ (周期折返し)。
- 自然長と能動張力の時間発展 (モデル別、$\omega_e = 2\pi f_e$):

$$\ell_e^{\mathrm{rest}}(t)=L_e\,(1+a_e\sin(\omega_e t+\phi_e)), \qquad F_e^{A}(t)=A_e\,\frac{1+\sin(\omega_e t+\phi_e)}{2}$$

- 辺ごとに周波数が違うため、共通周期は存在しない。周期性の評価では、周波数比の分布と自己相関を使う (docs/agents/sim/physics.md)。

- モデルは params/default.json の `sim.muscle_model` で切り替える。遺伝子の意味が変わるため、モデルを変えた run 同士は数値を直接比較しない。

初回の対照として、形状固定・振幅と位相のみ探索を維持する (Rojas 2016 の「身体を固定し制御のみ進化」と同じ切り分け)。
