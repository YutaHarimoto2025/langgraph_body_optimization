# 固定 GA (src/bodyopt/ga.py)

- 集団 20、エリート 2 (継承・再評価なし)、トーナメント選択 k=3、一様交叉 (遺伝子単位スイッチ確率 0.8)、突然変異率 0.10/遺伝子、変異 sigma は範囲 × 0.05。位相は周期折返し、振幅は範囲内にクリップ。
- 構成は trajectory-solver-generic の brkga.py を参考に「評価は注入関数・集団更新は JAX 演算」の分離を踏襲。BRKGA の biased crossover は採用しない (first goal は合意どおり標準 GA)。
- evolve_population は (pop, fit, key) → (pop, key) の純関数。island model への拡張は、この関数を島ごとに vmap し移住演算を追加する形で行う (初回から実装しない)。
- 世代ループは run_ga 内で lax.scan。初期集団・交叉・変異すべて jax.random のキー分割で行い、Python 側の乱数を使わない。

## 再現性の限界

バネ力の組立に使う scatter-add は GPU 上で atomic 演算になり、加算順序が実行ごとに変わるため float32 最下位ビットが非決定になり得る。同じ seed・同じ実行環境では相対 1e-5 で一致する (tests/test_reproducibility.py) をもって再現とする。ビット一致を要求する場面では CPU 実行か決定的スキャッタの採用を別途検討する。

## 改善候補 (未導入)

- エリートの重複排除 (brkga の elite de-dup)。停滞が観測されたら導入する。
- トーナメント両親の selfing 防止。停帯観測時に再評価する。
