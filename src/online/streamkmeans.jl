"""
    StreamKmeans(k_target::Int)

Online clustering algorithm [1] to select inducing points in a streaming setting.
Reference :
[1] Liberty, E., Sriharsha, R. & Sviridenko, M. An Algorithm for Online K-Means Clustering. arXiv:1412.5721 [cs] (2015).
"""
mutable struct StreamKmeans{T} <: OnIPSA
    k_target::Int
    k_efficient::Int
    f::T
    q::Int
end

StreamKmeans(k_target::Int) = StreamKmeans(k_target, 0, 0.0, 0)

function initZ(rng::AbstractRNG, alg::StreamKmeans, X::AbstractVector, kernel=nothing)
    length(X) > 10 ||
        throw(ArgumentError("The first batch of data should be bigger than 10 samples"))
    k_efficient = max(1, ceil(Int, (alg.k_target - 15) / 5))
    alg.k_efficient = k_efficient + 10 > length(X) ? 0 : k_efficient
    samp = sample(rng, 1:length(X), alg.k_efficient + 10; replace=false)
    Z = X[samp]
    w = zeros(k)
    for i in 1:k
        w[i] = 0.5 * find_nearest_center(Z[i], Z[1:end .!= i], kernel)[2]
    end
    alg.f = sum(sort(w)[1:10]) #Take the 10 smallest values
    alg.q = 0
    return Z
end

function add_point!(
    rng::AbstractRNG,
    Z::AbstractVector,
    alg::StreamKmeans,
    X::AbstractVector;
    kernel=nothing,
)
    b = length(X) # Size of the input data
    for i in 1:b
        val = find_nearest_center(X[i], Z, kernel)[2]
        if val > (alg.f * rand(rng))
            # new_centers = vcat(new_centers,X[i,:]')
            push!(Z, X[i])
            alg.q += 1
        end
        if alg.q >= alg.k_efficient
            alg.q = 0
            alg.f *= 10
        end
    end
    return Z
end
