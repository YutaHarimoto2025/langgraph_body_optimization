# LangGraph による人工身体の最適化（アイデアメモ）

2D の質点・バネ身体を物理シミュレータ上で動かし、その形態と筋肉制御を最適化する。最適化のループ自体を LangGraph で組み、LLM を「設計者」あるいは「最適化戦略の立案者」として使う。

---

## 2. 基本となる身体モデル

身体は 2D のグラフ構造として表す。

$$
G=(V,E),\qquad V=\{1,\dots,N\},\qquad E\subseteq V\times V
$$

### ノード

各ノード $i\in V$ を質点として扱い、状態を位置・速度・質量で持つ。

$$
q_i=(x_i,y_i)\in\mathbb{R}^2,\qquad \dot q_i=(v_{x,i},v_{y,i}),\qquad m_i>0
$$

候補となるパラメータ:

- 位置 $q_i$
- 質量 $m_i$
- 必要なら接触特性（摩擦係数 $\mu_i$、反発係数など）

### エッジ

ノード間を伸縮可能なエッジで接続する。エッジは構造材・バネ・筋肉を兼ねる単純なモデルとする。

エッジ $e=(i,j)\in E$ の長さを

$$
\ell_e=\|q_i-q_j\|_2
$$

とすると、バネ力をスカラーで

$$
F_e=k_e\left(\ell_e-\ell_e^{\mathrm{rest}}\right)
$$

と書ける。力の向きはエッジ方向の単位ベクトル

$$
\hat u_{ij}=\frac{q_i-q_j}{\|q_i-q_j\|_2}
$$

で与えられ、ノードに働く力は

$$
\mathbf F_{ij}=-F_e\,\hat u_{ij}
$$

となる。筋肉として使う場合は自然長を周期的に変化させる。

$$
\ell_e^{\mathrm{rest}}(t)=L_e\left(1+a_e\sin(\omega_e t+\phi_e)\right)
$$

候補パラメータ:

- 接続するノード $i,j$
- 基準長 $L_e$
- 剛性 $k_e$
- 収縮振幅 $a_e$
- 位相 $\phi_e$
- 周波数 $\omega_e$

最初はすべてのエッジを同一材質・同一剛性（$k_e=k$）としてもよい。

---

## 3. 物理環境

最初はできるだけ単純にする。

候補:

- 2D
- 重力あり（$g=(0,-g)$）
- 水平な地面（$y=0$）
- 接触あり
- 摩擦あり
- 1 episode 数秒程度

身体の運動は質点ごとの運動方程式で決まる。

$$
m_i\ddot q_i=m_i g+\sum_{j:(i,j)\in E}\mathbf F_{ij}+\mathbf F_i^{\mathrm{contact}}
$$

地面との接触は、めり込み量に比例する反力（ペナルティ法）で近似できる。ノードが地面より下にあるとき $y_i<0$ なので、

$$
F_i^{N}=k_c\max(0,-y_i)-c_c\min(0,\dot y_i)
$$

を法線方向の反力とし、接線方向にはクーロン摩擦

$$
F_i^{f}=-\mu\,F_i^{N}\,\operatorname{sgn}(\dot x_i)
$$

を加える。実装は半陰的オイラー法で離散化すれば十分である。

$$
v_i^{(t+1)}=v_i^{(t)}+\frac{\Delta t}{m_i}F_i^{(t)},\qquad q_i^{(t+1)}=q_i^{(t)}+\Delta t\,v_i^{(t+1)}
$$

最低限、「筋肉の周期運動 → 地面との接触 → 身体の並進」が生じればよい。本格的なロボットシミュレータは不要で、独自の質点・バネ simulator でも成立する。

将来的には坂道・凹凸地面・水中・低重力・障害物・外乱を追加できる。これらは重力ベクトル $g$、接触判定、外乱項 $F_i^{\mathrm{ext}}$ の変更として統一的に扱える。

---

## 4. 最初の目的関数

最初は単純に右方向への移動距離（重心の変位）を最大化する。

$$
f=x_{COM}(T)-x_{COM}(0)
$$

重心は質量で重み付けした平均である。

$$
x_{COM}(t)=\frac{1}{M}\sum_{i\in V}m_i x_i(t),\qquad M=\sum_{i\in V}m_i
$$

単純な移動距離だけでは「身体を吹き飛ばす」「激しく跳ねる」といった意図しない解が発生する。必要になれば、罰則項を加えた評価関数にする。

$$
J=d-\lambda_E E-\lambda_B P_{broken}-\lambda_U P_{unstable}
$$

ここで $d=f$ は移動距離、$E$ は消費エネルギー、$P_{broken},P_{unstable}$ は破損・転倒のペナルティ、$\lambda$ は各項の重みである。エネルギー消費は筋肉仕事の時間積分として定義できる。

$$
E=\int_0^T\sum_{e\in E}\left|F_e(t)\,\dot\ell_e(t)\right|dt
$$

ただし最初から複雑な reward を作り込まず、reward hacking が起きたら制約を追加する方が面白い。

---

## 5. 設計変数の候補

初期段階では以下の一部だけを使う。設計変数を形態と制御に分けてベクトルで表す。

$$
\theta=(\theta_{\mathrm{morph}},\theta_{\mathrm{ctrl}})
$$

### Morphology

$$
\theta_{\mathrm{morph}}=\left(\{q_i\}_{i\in V},\ \{L_e\}_{e\in E},\ \{k_e\}_{e\in E},\ \{m_i\}_{i\in V},\ E_{\mathrm{topology}}\right)
$$

- ノード位置 $q_i$
- ノード数 $N$
- エッジ接続 $E_{\mathrm{topology}}$
- エッジ長 $L_e$
- 剛性 $k_e$
- 質量 $m_i$

### Muscle / Controller

$$
\theta_{\mathrm{ctrl}}=\left(\{a_e\},\ \{\phi_e\},\ \{\omega_e\}\right)
$$

- 収縮振幅 $a_e$
- 位相 $\phi_e$
- 周波数 $\omega_e$

初期版ではノード数固定・周波数共通とし、ノード位置、エッジ topology、amplitude、phase 程度に絞ると扱いやすい。すなわち $\omega_e=\omega$ と固定し、探索対象を

$$
\theta=\left(\{q_i\},\ E_{\mathrm{topology}},\ \{a_e\},\ \{\phi_e\}\right)
$$

に限定する。最適化問題は次式で書ける。

$$
\theta^*=\arg\max_{\theta\in\Theta}\ J(\theta)
$$

---

## 6. 試してみたい Agent の役割

### A. Agent が身体を直接設計する

LLM が身体そのものを提案する。これは履歴 $\mathcal H_m$（過去の設計・評価・失敗の要約）から次の設計への写像と見なせる。

$$
\theta^{(m+1)}=\mathcal L\!\left(\mathcal H_m\right),\qquad
\mathcal H_m=\{(\theta^{(k)},J^{(k)},\mathrm{analysis}^{(k)})\}_{k\le m}
$$

```text
LLM Designer
    ↓
Creature Definition
    ↓
Validation
    ↓
Physics Simulation
    ↓
Fitness
    ↓
Behavior Analysis
    ↓
LLM Reflection
    ↓
Next Creature
```

LLM が前回の結果を読んで、後脚が滑っている、接地点が少ない、前後の筋肉位相が合っていない、重心位置が悪い、などの仮説を立て、次の身体を提案する。LLM にとって設計変数を直接操作する能力がどの程度あるかを見られる。

---

### B. Agent が既存 optimizer を監督する

身体自体の探索は GA などに任せる。GA の1世代は選択・交叉・突然変異の合成として書ける。

$$
P^{(t+1)}=\mathrm{Mutate}_{\lambda^{(t)}}\circ\mathrm{Crossover}\circ\mathrm{Select}\!\left(P^{(t)}\right)
$$

LLM は数世代ごとに探索状況だけを見て、ハイパーパラメータを更新する。

$$
\lambda^{(t+1)}=\mathcal L\!\left(s^{(t)}\right)
$$

```text
GA
 ↓
数世代実行
 ↓
Statistics
 ↓
LLM Analyst
 ↓
Strategy Update
 ↓
GA
```

LLM に渡す情報 $s^{(t)}$ の例:

- best fitness
- average fitness
- feasible ratio
- population diversity
- 改善量
- mutation rate
- topology diversity
- failure type の分布

LLM が変更する候補 $\lambda$:

- mutation rate
- crossover rate
- selection pressure
- topology mutation 強度
- population restart
- elite 数
- morphology / controller の探索比率

これは「LLM = semantic / strategic controller」「GA = numerical search」と役割を分離する方式である。

---

### C. Agent が optimizer 自体を設計する

より実験的な案。LLM が GA のパラメータを変更するだけでなく、

- 新しい mutation operator
- selection 方法
- morphology と controller の交互最適化
- 局所探索
- restart strategy
- surrogate model
- CMA-ES / BO への切替

など、探索アルゴリズムそのものを提案する。探索アルゴリズムを写像 $A$ として抽象化する。

$$
A:\ P^{(t)}\mapsto P^{(t+1)},\qquad A=\mathcal L(\mathrm{benchmark})
$$

さらに進めるなら、

```text
Optimizer Agent
      ↓
optimizer.py を生成
      ↓
Sandbox Test
      ↓
Benchmark
      ↓
Baseline と比較
      ↓
Accept / Reject
```

として、Agent に optimizer のコードを作らせることもできる。

---

## 7. 比較すると面白い方式

同じ simulator、同じ評価 budget を使い、次を比較する。

1. LLM Direct Designer
2. 固定 GA
3. LLM-supervised GA
4. LLM-generated Optimizer

全方式で simulation 回数を揃える。

$$
N_{\mathrm{sim}}=1000
$$

方式 $a$ の性能を、与えられた計算予算 $N_{\max}$ の下での最良 fitness として比較する。

$$
J_a^*=\max_{\theta}\ J(\theta)\quad \text{s.t.}\quad N_{\mathrm{sim}}(a)\le N_{\max}
$$

比較指標候補:

- best fitness
- simulation evaluations
- wall time
- LLM token 使用量
- feasible solution rate
- behavior diversity
- morphology diversity
- robustness

特に、

> LLM は「設計者」として使う方がよいのか、それとも「最適化アルゴリズム設計者」として使う方がよいのか

という比較ができると面白い。

---

## 8. LangGraph で試したい構造

LangGraph の graph は状態 $s$ と遷移関数 $\delta$ の組として書ける。

$$
s_{t+1}=\delta(s_t),\qquad
\delta(s)=
\begin{cases}
\mathrm{DesignCreature}(s) & \text{if } \mathrm{cond}(s)\\
\mathrm{END} & \text{otherwise}
\end{cases}
$$

### Direct Designer

```text
START
  ↓
DesignCreature
  ↓
Validate
  ↓
Simulation
  ↓
Evaluate
  ↓
AnalyzeBehavior
  ↓
Improve?
 ├─ yes → DesignCreature
 └─ no  → END
```

### LLM-supervised Optimizer

```text
START
  ↓
InitializePopulation
  ↓
RunOptimizer
  ↓
EvaluatePopulation
  ↓
Stagnation?
 ├─ no  → RunOptimizer
 └─ yes
      ↓
  LLMDiagnose
      ↓
  SelectStrategy
   ├─ increase mutation
   ├─ partial restart
   ├─ topology exploration
   └─ continue
      ↓
  RunOptimizer
```

停滞判定は、直前 $w$ 世代の改善量が閾値以下かで定義できる。

$$
\Delta J_t=J_t^*-J_{t-w}^*<\epsilon\ \Rightarrow\ \text{stagnation}
$$

### Optimizer Designer

```text
START
  ↓
GenerateOptimizer
  ↓
ValidateCode
  ↓
Benchmark
  ↓
CompareBaseline
  ↓
Improved?
 ├─ yes → KeepOptimizer
 └─ no  → AnalyzeFailure
             ↓
        GenerateOptimizer
```

LangSmith Studio では、

- 実際にどの branch を通ったか
- 何回 restart したか
- どの strategy が選択されたか
- LLM がどの判断をしたか
- optimizer のどの変更が改善につながったか

を trace できる。

---

## 9. 段階的な拡張案

### Level 1: 固定 topology

- ノード数固定
- エッジ接続固定
- node geometry + muscle phase / amplitude のみ最適化

$$
\theta=\left(\{q_i\},\ \{a_e\},\ \{\phi_e\}\right)
$$

まず物理 simulator と LangGraph を成立させる。

### Level 2: Topology optimization

- edge の追加・削除を許可
- 生物の形そのものを探索

$$
E^{(t+1)}=\left(E^{(t)}\setminus\{e\}\right)\cup\{e'\},\qquad e\in E^{(t)}
$$

### Level 3: ノード数可変

- node add
- node remove
- edge split

などを mutation として追加する。ノード数も設計変数になる。

$$
N^{(t+1)}=N^{(t)}\pm 1
$$

### Level 4: Morphology-Control Co-design

身体と筋肉制御を同時に最適化する。

$$
\theta^*=(\theta_{\mathrm{morph}}^*,\theta_{\mathrm{ctrl}}^*)
=\arg\max_{\theta_{\mathrm{morph}},\theta_{\mathrm{ctrl}}}\ J(\theta_{\mathrm{morph}},\theta_{\mathrm{ctrl}})
$$

ただし探索が難しくなるため、交互最適化も試す。

$$
\theta_{\mathrm{morph}}^*=\arg\max_{\theta_{\mathrm{morph}}}J(\theta_{\mathrm{morph}},\theta_{\mathrm{ctrl}}^*),\qquad
\theta_{\mathrm{ctrl}}^*=\arg\max_{\theta_{\mathrm{ctrl}}}J(\theta_{\mathrm{morph}}^*,\theta_{\mathrm{ctrl}})
$$

```text
Morphology optimization
        ↓
Controller optimization
        ↓
Morphology optimization
```

### Level 5: Feedback controller

周期関数だけでなく、観測 $s_t$ から操作 $u_t$ への feedback controller にする。

$$
u_t=f_\theta(s_t)
$$

入力候補:

$$
s_t=\left(\{\mathrm{contact}_i\},\ \{q_i\},\ \{\dot q_i\},\ \Theta_{\mathrm{body}}\right)
$$

- 接触
- node position
- velocity
- body orientation $\Theta_{\mathrm{body}}$

### Level 6: 複数環境

同じ身体を $K$ 個の環境で評価する。単一環境への過適合を避け、robustness を次式で評価する。

$$
J_{\mathrm{robust}}(\theta)=\frac{1}{K}\sum_{k=1}^{K}J_k(\theta)
\quad\text{または}\quad
J_{\mathrm{robust}}(\theta)=\min_{k}J_k(\theta)
$$

- 平地
- 坂道
- 凹凸
- 低摩擦

---

## 10. 面白そうな発展方向

### 人工進化

世代を重ねることで、歩行・跳躍・這行・転がる・尺取虫型など、明示的に指定していない運動様式が出現する可能性がある。

### Novelty Search

距離だけではなく、行動の新規性を評価する。behavior descriptor を

$$
b=\left(\overline{h},\ n_{contact},\ \Theta_{rot},\ T_{period},\ \overline{v}_x\right)
$$

などにして、似ていない動きを積極的に探索する。新規性は k 近傍までの平均距離として定義できる。

$$
\mathrm{nov}(b)=\frac{1}{k}\sum_{b'\in \mathrm{kNN}(b)}\|b-b'\|
$$

探索の評価を目的関数と新規性の混合にする。

$$
J_{\mathrm{NS}}(\theta)=J(\theta)+\beta\,\mathrm{nov}(b(\theta))
$$

### Open-ended Evolution

単一の「最良個体」を探すだけでなく、多様な身体・多様な移動様式・多様な環境への適応を継続的に生成する方向にも拡張できる。

### Agent に環境を作らせる

身体だけでなく、curriculum を自動生成する。

```text
Environment Agent
       ↓
新しい環境
       ↓
Creature Optimizer
       ↓
適応
```

環境をパラメータ $\phi$（重力、摩擦、地形）で表し、適応度 $J(\theta,\phi)$ に対する curriculum を生成する。

$$
\phi^{(m+1)}=\mathcal L\!\left(\{(J(\theta^{(k)},\phi^{(k)}))\}_{k\le m}\right)
$$

---

## 11. 最初の実装イメージ

最初は以下くらいが扱いやすそう。

```text
Dimension:
    2D

Nodes:
    6〜10

Edges:
    8〜20

Physics:
    mass-spring
    gravity
    ground collision
    friction

Muscle:
    sinusoidal rest-length control

Episode:
    約5秒

Primary objective:
    右方向へのCOM移動距離

Optimization:
    GA

LangGraph:
    GA loop
    stagnation detection
    LLM diagnosis
    strategy update

Visualization:
    creature animation
    fitness history
    LangSmith Studio trace
```

この段階では「最適な人工生命 simulator」を作ることより、

> LangGraph を使うことで、数値最適化・LLM判断・失敗処理・戦略変更をどの程度きれいに構造化・観測できるか

を見ることを主目的とする。

---

## 12. この題材で確認したい問い

最終仕様を固定する前に、以下を実験できると面白い。各問いは、比較すべき仮説として書ける。

- LLM は身体を直接設計できるか: $\mathcal L$ による探索が GA より少ない $N_{\mathrm{sim}}$ で高い $J$ に到達するか
- LLM は数値 optimizer より少ない試行で有望な形態を発見できるか: $J_{\mathrm{LLM}}(N_{\mathrm{sim}})$ と $J_{\mathrm{GA}}(N_{\mathrm{sim}})$ の比較
- GA の停滞を LLM は正しく診断できるか: 診断 → 戦略変更 → 改善の因果
- LLM に optimizer の戦略変更を任せると改善するか
- optimizer のコード自体を Agent に改良させられるか
- morphology と controller は同時探索すべきか、分離すべきか
- LLM の介入頻度はどの程度が適切か: 介入間隔 $\tau$ を変えた比較
- Agent が選んだ graph 経路と性能改善に相関があるか
- 単一の best fitness だけでなく、多様な locomotion を発見できるか

このあたりを自由に試せる実験場として作るのが、このアイデアの中心。
