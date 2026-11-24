# 身体モデルと遺伝子

## 身体

- 三角補強ラダー: 上下 2 列 × 3 点 = 6 質点、横 4 + 縦 3 + 対角 2 = 9 辺 (src/bodyopt/creature.py の triangulated_ladder)。
- 列間隔 0.5 m、行間隔 0.4 m、下列は床 y=0 から 0.05 m 上。全質点 1 kg、全辺剛性 100 N/m。
- 基準自然長は初期辺長に一致 (初期張力ゼロ)。
- 全 9 辺を能動筋肉として扱う (muscle_mask)。受動バネとの区別は mask で表現し、駆動する辺の指定は設定で変えられる。muscle_idx は構築時に確定する (JIT 内で nonzero を呼ばないため)。

## 遺伝子

$$\theta = (\{a_e\}_{e \in M},\ \{\phi_e\}_{e \in M})$$

- 振幅 $a_e \in [0, 0.20]$ (MAX_AMPLITUDE、常に課すハード上限)。
- 位相 $\phi_e \in [0, 2\pi)$ (周期折返し)。
- 周波数は全筋肉共通 $\omega = 2\pi$ rad/s (1 Hz) で遺伝子外。個別周波数・剛性・質量は first goal の対象外。
- 自然長の時間発展:

$$\ell_e^{\mathrm{rest}}(t) = L_e (1 + a_e \sin(\omega t + \phi_e))$$

初回の対照として、形状固定・振幅と位相のみ探索を維持する (Rojas 2016 の「身体を固定し制御のみ進化」と同じ切り分け)。
