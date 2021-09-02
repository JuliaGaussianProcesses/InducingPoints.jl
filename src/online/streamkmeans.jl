"""
    StreamKmeans(m_target::Int)

Online clustering algorithm [1] to select inducing points in a streaming setting.
Reference :
[1] Liberty, E., Sriharsha, R. & Sviridenko, M. An Algorithm for Online K-Means Clustering. arXiv:1412.5721 [cs] (2015).
"""
mutable struct StreamKmeans{T} <: OnIPSA
    m_target::Int
    m_efficient::Int
    f::T
    q::Int
end

StreamKmeans(m_target::Int) = StreamKmeans(m_target, 0, 0.0, 0)

function initZ(rng::AbstractRNG, alg::StreamKmeans, X::AbstractVector; kernel=nothing, kwargs...)
    length(X) > 10 ||
        throw(ArgumentError("The first batch of data should be bigger than 10 samples"))
    m_efficient = max(1, ceil(Int, (alg.m_target - 15) / 5))
    alg.m_efficient = m_efficient + 10 > length(X) ? 0 : m_efficient
    samp = sample(rng, 1:length(X), alg.m_efficient + 10; replace=false)
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
    kwargs...
)
    b = length(X) # Size of the input data
    for i in 1:b
        val = find_nearest_center(X[i], Z, kernel)[2]
        if val > (alg.f * rand(rng))
            # new_centers = vcat(new_centers,X[i,:]')
            push!(Z, X[i])
            alg.q += 1
        end
        if alg.q >= alg.m_efficient
            alg.q = 0
            alg.f *= 10
        end
    end
    return Z
end
