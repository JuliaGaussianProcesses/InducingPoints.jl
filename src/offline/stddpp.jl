@doc raw"""
    StdDPP()

Standard DPP (Determinantal Point Process) sampling.
The size of the returned `Z` is not fixed (but is not allowed to be empty unlike in a classical DPP).
The kernel is passed as a keyword argument to [`inducingpoints`](@ref).
"""
struct StdDPP <: OffIPSA end

function inducingpoints(
    rng::AbstractRNG, alg::StdDPP, X::AbstractVector; kernel::Kernel, kwargs...
)
    dpp = DPP(kernel, X)
    samp = rand(rng, dpp)
    while isempty(samp) # Sample from the DPP until there is a non-empty set
        samp = rand(rng, dpp)
    end
    return X[samp]
end
