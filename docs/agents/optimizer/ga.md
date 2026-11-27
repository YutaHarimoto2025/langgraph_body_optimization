# 固定 GA (src/bodyopt/ga.py)

- 既定は集団 100 (params/default.json)、エリート 2 (継承・再評価なし)、トーナメント選択 k=3、一様交叉 (遺伝子単位スイッチ確率 0.8)、突然変異率 0.10/遺伝子、変異 sigma は範囲 × 0.05。位相は周期折返し、振幅と周波数は範囲内にクリップ。
- 遺伝子は [振幅] + [周波数] + [位相] の 3 ブロック。範囲は振幅 0〜100 N、周波数 0.1〜1 Hz、位相 0〜2π。
- 移民: 毎世代、適応度の下位 20% (GAParams.immigration_rate) をランダム個体で置き換える (inject_worst)。置換した個体は次世代で評価される。置換前のスコアは下位のまま残るので、その世代の選択・エリートには影響しない。
- 構成は trajectory-solver-generic の brkga.py を参考に「評価は注入関数・集団更新は JAX 演算」の分離を踏襲。BRKGA の biased crossover は採用しない (first goal は合意どおり標準 GA)。
- evolve_population は (pop, fit, key) → (pop, key) の純関数。island model への拡張は、この関数を島ごとに vmap し移住演算を追加する形で行う (初回から実装しない)。
- 世代ループは run_ga 内で lax.scan。初期集団・交叉・変異すべて jax.random のキー分割で行い、Python 側の乱数を使わない。

## 再現性の限界

バネ力の組立に使う scatter-add は GPU 上で atomic 演算になり、加算順序が実行ごとに変わるため float32 最下位ビットが非決定になり得る。

- 周波数上限 1 Hz・減衰 ζ=0.1 なら、同じ seed の短い GA run は 1e-4 以内で一致する (tests/test_reproducibility.py)。
- 周波数を 3 Hz 以上にすると遺伝子依存で感度が上がり、10 Hz では同じ seed でも選択が反転して集団ごと変わる (3 世代でスコア差 1.9 m)。
- 個体単体では 1 Hz でも 20 回の距離幅が 0.002 m 程度残る。run 間の比較は平均・分散で行う。
- run 間の比較は平均・分散で行う。ビット一致が必要な場面では CPU 実行か決定的スキャッタを検討する。

## 改善候補 (未導入)

- エリートの重複排除 (brkga の elite de-dup)。停滞が観測されたら導入する。
- トーナメント両親の selfing 防止。停帯観測時に再評価する。
