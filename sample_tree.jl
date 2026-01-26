using Random

@views function sample(train_x, train_y, sample_size)
    sample_size = min(size(train_x,1), sample_size)
    idx = randperm(size(train_x,1))[1:sample_size]
    train_x_sample =  train_x[idx,:]
    train_y_sample =  train_y[idx]
    return train_x_sample, train_y_sample
end
@views function find_thre(train_x_sample, train_y_sample)
    sample_size = size(train_x_sample,1)
    sorted_id = fill(0, size(train_x_sample))
    sorted_y = fill(0.0, size(train_x_sample))
    for i in 1:size(train_x,2)
        sorted_id[:,i] .= sortperm( train_x_sample[:,i])
        sorted_y[:,i] .=  train_y_sample[sorted_id[:,i]]
    end
    n_l = 1:sample_size
    sum_y_l = cumsum(sorted_y, dims=1)
    mean_l = sum_y_l ./ n_l
    sumq_y_l = cumsum(sorted_y.^2, dims=1)
    n_r = sample_size-1:-1:0
    sum_y_r = sum_y_l[end,:]' .- sum_y_l
    mean_r = sum_y_r ./ n_r
    sumq_y_r = sumq_y_l[end,:]' .- sumq_y_l
    sse_l = sumq_y_l .- n_l .* mean_l .^ 2
    sse_r = sumq_y_r .- n_r .* mean_r .^ 2
    sse = sse_l + sse_r
    best = argmin(sse[1:end-1,:])
    best_col = best[2]
    best_id = sorted_id[best]
    thre = train_x_sample[best_id,best_col]
    return best_col, thre
end

@views function split_by_thre!(train_x, train_y,sample_size)
    train_x_sample, train_y_sample = sample(train_x,train_y,sample_size)
    best_col, thre = find_thre(train_x_sample, train_y_sample)
    is_l = ( train_x[:,best_col]) .< thre
    is_r = .!is_l
    train_x .= vcat(train_x[is_l,:], train_x[is_r,:])
    train_y .= vcat(train_y[is_l], train_y[is_r])
    size_l = sum(is_l)
    return size_l,best_col,thre
end

@views function fit(train_x_, train_y_, min_size, max_depth, sample_size)
    train_x = copy(train_x_)
    train_y = copy(train_y_)
    ptr = 1
    tree_strat_id = Int64[1]
    tree_n = Int64[size(train_x,1)]
    tree_depth = Int64[1]
    tree_col = Int64[]
    tree_thre = Float64[]
    tree_child= Int64[]
    while ptr <= length(tree_strat_id)
        sub_train_x =  train_x[tree_strat_id[ptr]:tree_strat_id[ptr]+tree_n[ptr]-1,:]
        sub_train_y =  train_y[tree_strat_id[ptr]:tree_strat_id[ptr]+tree_n[ptr]-1]
        if min_size < size(sub_train_x,1) && tree_depth[ptr] < max_depth
            size_l, best_col, thre = split_by_thre!(sub_train_x, sub_train_y,sample_size)
            push!(tree_strat_id, tree_strat_id[ptr], tree_strat_id[ptr] + size_l)
            push!(tree_n, size_l, size(sub_train_x,1)-size_l)
            push!(tree_depth, tree_depth[ptr]+1, tree_depth[ptr]+1)
            push!(tree_col, best_col)
            push!(tree_thre, thre)
            push!(tree_child, length(tree_strat_id))
        end
        ptr += 1
    end
    return Dict(
        "strat_id"=>tree_strat_id, 
        "n"=>tree_n, 
        "depth"=>tree_depth, 
        "col"=>tree_col, 
        "thre"=>tree_thre, 
        "child"=>tree_child
    )
end
