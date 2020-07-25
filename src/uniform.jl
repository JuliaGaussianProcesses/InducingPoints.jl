mutable struct UniformSampling{T,M<:AbstractMatrix{T}} <: AIP{T,M}
    k::Int64
    Z::M
    function UniformSampling(nInducingPoints::Integer)
        return new{Float64,Matrix{Float64}}(nInducingPoints)
    end
end

function init!(alg::UniformSampling, X, y, kernel)
    @assert size(X, 1) >= alg.k "Input data not big enough given $k"
    samp = sample(1:size(X, 1), alg.k, replace = false)
    alg.Z = X[samp, :]
end
