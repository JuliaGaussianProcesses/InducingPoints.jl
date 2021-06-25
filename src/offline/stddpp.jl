"""
    StdDPP(X::AbstractMatrix, kernel::Kernel; obsdim::Int = 1)
    StdDPP(X::AbstractVector, kernel::Kernel)

Standard DPP (Determinantal Point Process) sampling given `kernel`.
The size of the returned `Z` is not fixed
"""
struct StdDPP{K<:Kernel} <: OffIPSA
    kernel::K
    function StdDPP(kernel::K) where {K<:Kernel}
        new{K}(kernel)
    end
end

function inducingpoints(rng::AbstractRNG, alg::StdDPP, X::AbstractVector; kwargs...)
    K = Symmetric(kernelmatrix(alg.kernel, X) + jitt * I)
    dpp = DPP(K)
    samp = rand(rng, dpp, 1)[1]
    return X[samp]
end