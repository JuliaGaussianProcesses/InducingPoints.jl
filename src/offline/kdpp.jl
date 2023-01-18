"""
    kDPP(m::Int)

k-DPP (Determinantal Point Process) will return a subset of `X` of size `m`,
according to DPP probability.
The kernel is passed as a keyword argument to [`inducingpoints`](@ref).
"""
struct kDPP <: OffIPSA
    m::Int
    function kDPP(m::Int)
        m > 0 || throw(ArgumentError("The number of inducing points m should be positive"))
        return new(m)
    end
end

function inducingpoints(
    rng::AbstractRNG, alg::kDPP, X::AbstractVector; kernel::Kernel, kwargs...
)
    if alg.m >= length(X)
        return edge_case(alg.m, length(X), X)
    end
    return kdpp_ip(rng, X, alg.m, kernel)
end

Base.show(io::IO, alg::kDPP) = print(io, "k-DPP selection of $(alg.m) inducing points")

function kdpp_ip(rng::AbstractRNG, X::AbstractVector, m::Int, kernel::Kernel)
    return X[rand(rng, DPP(kernel, X)(m))] # Sample m indices from a DPP
end
