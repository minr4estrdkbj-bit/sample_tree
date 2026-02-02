### 概要
元の sample_tree.jl の分割ロジック周りの不具合を修正し、木のノードを明示的な構造体で表現するようにしました。また、学習済み木で予測する predict 関数を追加しまし���.

### 変更点:
- find_best_split: 閾値候補の評価を安定化（隣接値の中点を採用、同値スキップ）
- split_by_thre!: in-place で左→右の並べ替えを行うよう修正
- TreeNode 構造体を導入してノード情報を明確化
- fit: 幅優先でノードを展開、葉の平均値を保持
- predict/predict_single を追加（学習済み木での予測）

### 動作確認方法:
リポジトリの sample_tree.jl を置き換え、下記のスニペットで動作確認してください。

例:
X = randn(200,3); y = X[:,1]*2 .- X[:,2]*0.5 .+ 0.1*randn(200)
tree = fit(X,y,min_size=5,max_depth=6,sample_size=50)
preds = predict(tree, randn(10,3))

### 注意事項:
簡易な回帰木実装です。分類対応、剪定、性能最適化などは別途対応可能です.