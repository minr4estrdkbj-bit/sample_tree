using Random
using Statistics
using Test

# load implementation
include("../sample_tree.jl")

Random.seed!(1234)

@testset "sample_tree basic tests" begin
    # 生成データ: y = 2*x1 - 0.5*x2 + noise
    n_train = 200
    n_test = 50
    p = 3
    X = randn(n_train, p)
    y = X[:,1] * 2.0 .- X[:,2] * 0.5 .+ 0.05 * randn(n_train)

    # 学習
    nodes = fit(X, y, min_size=5, max_depth=6, sample_size=100)

    @test length(nodes) >= 1
    @test all(!isnan(n.mean) for n in nodes)

    # 予測 (バッチ)
    Xnew = randn(n_test, p)
    preds = predict(nodes, Xnew)
    @test length(preds) == n_test

    # 予測 (単一)
    s = predict_single(nodes, Xnew[1, :])
    @test isa(s, Float64)

    # 性能指標: MSE が十分小さいこと（緩めの閾値）
    ytrue = Xnew[:,1] * 2.0 .- Xnew[:,2] * 0.5
    mse = mean((preds .- ytrue).^2)
    @test mse < 0.5
end