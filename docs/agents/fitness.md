# 評価関数 (src/bodyopt/fitness.py)

$$f = x_{\mathrm{COM}}(T) - x_{\mathrm{COM}}(0), \quad x_{\mathrm{COM}}(t) = \frac{1}{M} \sum_i m_i x_i(t)$$

- 無効条件: 非有限値、最大めり込み > 0.10 m、(オプション) 駆動仕事が予算超過。
- 無効個体のスコアは INVALID_SCORE = -1e6 (有効個体より常に下位)。距離の罰則への混合はしない。
- 駆動仕事 (エネルギー消費の代理指標、常に記録する):

$$E_{\mathrm{drive}} = \int_0^T \sum_{e \in M} \max\!\left(0,\ -F_e(t)\, L_e\, a_e\, \omega \cos(\omega t + \phi_e)\right) dt$$

張った筋肉をさらに縮めるときだけ正の仕事と数え、回生は数えない。energy_budget (None = 無制限) を設定した場合だけ無効判定に使う。first goal は無制限で走らせ、E_drive を距離と並べて報告する。

EvalStats は vmap 出力に使うため pytree として登録する。分析用の付随統計 (distance・drive_energy・max_penetration・contact_rate) は GA の選択に使わず、記録と診断にだけ使う。

## 既知の失敗モード

変位のみの評価では「跳んで飛ぶ」解が最適になる (報告 2026-11-23: 滞空率 64%、重心最高 5.8 m)。対処 (滞空率罰則・エネルギー予算・物理調整) は利用者判断待ち。評価式を変える場合は、比較方式間で同じ式に固定する。
