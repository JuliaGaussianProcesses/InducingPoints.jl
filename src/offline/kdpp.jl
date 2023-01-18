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

function inducingpoints(rng::AbstractRNG, alg::kDPP, X::AbstractVector; kernel=nothing, kwargs...)
    kernel = if isnothing(kernel)
        @warn "The API for kDPP changes in the next breaking release. Please pass the kernel as a keyword argument."
        alg.kernel
    else
        kernel
    end
    if alg.m >= length(X)
        return edge_case(alg.m, length(X), X)
    end
    return kdpp_ip(rng, X, alg.m, kernel)
end

Base.show(io::IO, ::kDPP) = print(io, "k-DPP selection of inducing points")

function kdpp_ip(rng::AbstractRNG, X::AbstractVector, m::Int, kernel::Kernel)
    return X[rand(rng, DPP(kernel, X)(m))] # Sample m indices from a DPP
end
