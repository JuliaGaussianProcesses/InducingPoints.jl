mutable struct UniformSampling{S,TZ<:AbstractVector{S}} <: OffIP{S,TZ}
    k::Int
    Z::TZ
end

function uniformsampling(X::AbstractMatrix, k::Int)
    @assert size(X, 1) >= alg.k "Input data not big enough given $k"
    samp = sample(1:size(X, 1), alg.k, replace = false)
    alg.Z = X[samp, :]
    UniformSampling
end
