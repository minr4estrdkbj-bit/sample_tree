using Random
using Statistics

mutable struct TreeNode
    start::Int       # start index in the (reordered) training arrays
    n::Int           # number of rows in this node
    depth::Int       # node depth (root = 1)
    col::Int         # splitting column (0 means leaf)
    thre::Float64    # threshold value for split
    left::Int        # index of left child in nodes array (0 if none)
    right::Int       # index of right child in nodes array (0 if none)
    mean::Float64    # prediction value for leaf (mean of y)
end

# ランダムサンプリング（行の view と sampled y のコピーを返す）
function sample_rows(X::AbstractMatrix, y::AbstractVector, sample_size::Int)
    s = min(size(X,1), sample_size)
    idx = randperm(size(X,1))[1:s]
    return view(X, idx, :), y[idx]
end

# 与えられたサンプルから各特徴ごとに最良の閾値（SSE最小）を探す
# 戻り値: best_col (0 if no valid split), best_thre
function find_best_split(Xs, ys)
    n = length(ys)
    p = size(Xs,2)
    if n <= 1
        return 0, 0.0
    end

    best_sse = Inf
    best_col = 0
    best_thre = 0.0

    for j in 1:p
        # 値を連続配列に取り出す（サンプルが view でも問題ないように）
        colvals = collect(Xs[:, j])
        order = sortperm(colvals)
        y_sorted = ys[order]
        x_sorted = colvals[order]

        # 累積和 (left side)
        sum_y_l = cumsum(y_sorted)
        sumq_y_l = cumsum(y_sorted .^ 2)
        total_y = sum_y_l[end]
        total_q = sumq_y_l[end]

        # k = left の要素数 (1 .. n-1)
        for k in 1:(n-1)
            # 同値で分割できない場合はスキップ（閾値が同じ）
            if x_sorted[k] == x_sorted[k+1]
                continue
            end

            n_l = k
            n_r = n - k

            sum_l = sum_y_l[k]
            sumq_l = sumq_y_l[k]
            mean_l = sum_l / n_l
            sse_l = sumq_l - n_l * mean_l^2

            sum_r = total_y - sum_l
            sumq_r = total_q - sumq_l
            mean_r = sum_r / n_r
            sse_r = sumq_r - n_r * mean_r^2

            sse = sse_l + sse_r

            if sse < best_sse
                best_sse = sse
                best_col = j
                # 閾値は隣接値の中点を取る（安全な選択）
                best_thre = (x_sorted[k] + x_sorted[k+1]) / 2
            end
        end
    end

    return best_col, best_thre
end

# 指定の領域（train region 用の行列/ベクトル）をサンプルを使って分割し、
# 元の領域を左サブセット→右サブセットの順に並べ替える（in-place）。
# 戻り値: size_l, best_col, thre
function split_by_thre!(X_region::AbstractMatrix, y_region::AbstractVector, sample_size::Int)
    Xs, ys = sample_rows(X_region, y_region, sample_size)
    best_col, thre = find_best_split(Xs, ys)
    if best_col == 0
        return 0, 0, 0.0
    end
    is_left = X_region[:, best_col] .< thre
    size_l = count(is_left)
    # 領域を左→右の順に並べ替え（in-place 書き換え）
    X_region .= vcat(X_region[is_left, :], X_region[.!is_left, :])
    y_region .= vcat(y_region[is_left], y_region[.!is_left])
    return size_l, best_col, thre
end

# fit: データをコピーして木を構築。返り値は Vector{TreeNode}
# 引数:
#   X_, y_ : training data
#   min_size: 末端ノードとする最小サンプル数（strict > min_size で分割を試行）
#   max_depth: 深さ上限
#   sample_size: 各ノードで閾値探索に使うサンプル数
function fit(X_, y_, min_size::Int=5, max_depth::Int=10, sample_size::Int=100)
    X = copy(X_)
    y = copy(y_)
    nodes = Vector{TreeNode}()
    push!(nodes, TreeNode(1, size(X,1), 1, 0, 0.0, 0, 0, mean(y)))
    ptr = 1
    while ptr <= length(nodes)
        node = nodes[ptr]
        # ノードの領域を view として取得
        start = node.start
        nnode = node.n
        sub_x = view(X, start:start+nnode-1, :)
        sub_y = view(y, start:start+nnode-1)

        if nnode > min_size && node.depth < max_depth
            size_l, best_col, thre = split_by_thre!(sub_x, sub_y, sample_size)
            # split できなければ葉にする
            if size_l <= 0 || size_l >= nnode
                nodes[ptr].col = 0
                nodes[ptr].thre = 0.0
                nodes[ptr].mean = mean(sub_y)
            else
                nodes[ptr].col = best_col
                nodes[ptr].thre = thre
                # 左子・右子ノードを追加
                left_idx = length(nodes) + 1
                right_idx = length(nodes) + 2
                push!(nodes, TreeNode(start, size_l, node.depth+1, 0, 0.0, 0, 0, mean(view(y, start:start+size_l-1))))
                push!(nodes, TreeNode(start+size_l, nnode - size_l, node.depth+1, 0, 0.0, 0, 0, mean(view(y, start+size_l:start+nnode-1))))
                nodes[ptr].left = left_idx
                nodes[ptr].right = right_idx
            end
        else
            # 葉にする
            nodes[ptr].col = 0
            nodes[ptr].thre = 0.0
            nodes[ptr].mean = mean(sub_y)
        end
        ptr += 1
    end

    return nodes
end

# 単一サンプルの予測
function predict_single(nodes::Vector{TreeNode}, x::AbstractVector)
    idx = 1
    while true
        node = nodes[idx]
        # col == 0 を葉の指標とする
        if node.col == 0 || node.left == 0
            return node.mean
        end
        if x[node.col] < node.thre
            idx = node.left
        else
            idx = node.right
        end
    end
end

# 複数サンプルの予測 (行列入力)
function predict(nodes::Vector{TreeNode}, Xnew::AbstractMatrix)
    n = size(Xnew, 1)
    preds = Vector{Float64}(undef, n)
    for i in 1:n
        preds[i] = predict_single(nodes, Xnew[i, :])
    end
    return preds
end
