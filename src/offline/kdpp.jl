"""
    kDPP(m::Int, kernel::Kernel)

k-DPP (Determinantal Point Process) will return a subset of `X` of size `m`,
according to DPP probability
"""
struct kDPP{K<:Kernel} <: OffIPSA
    m::Int
    kernel::K
    function kDPP(m::Int, kernel::K) where {K<:Kernel}
        m > 0 || throw(ArgumentError("The number of inducing points m should be positive"))
        return new{K}(m, kernel)
    end
end

function incudingpoints(rng::AbstractRNG, alg::kDPP, X::AbstractVector; kwargs...)
    return kddp_ip(rng, X, alg.m, alg.kernel)
end

Base.show(io::IO, ::kDPP) = print(io, "k-DPP selection of inducing points")

function kdpp_ip(rng::AbstractRNG, X::AbstractVector, m::Int, kernel::Kernel)
    return X[rand(rng, DPP(kernel, X)(m))] # Sample m indices from a DPP
end
