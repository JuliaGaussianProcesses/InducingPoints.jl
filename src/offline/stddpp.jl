@doc raw"""
    StdDPP(kernel::Kernel)

Standard DPP (Determinantal Point Process) sampling given `kernel`.
The size of the returned `Z` is not fixed (but is not allowed to be empty unlike in a classical DPP).
"""
struct StdDPP{K<:Kernel} <: OffIPSA
    kernel::K
end

function inducingpoints(rng::AbstractRNG, alg::StdDPP, X::AbstractVector; kernel=nothing, kwargs...)
    kernel = if isnothing(kernel)
        @warn "The API for StdDPP changes in the next breaking release. Please pass the kernel as a keyword argument."
        alg.kernel
    else
        kernel
    end
    dpp = DPP(kernel, X)
    samp = rand(rng, dpp)
    while isempty(samp) # Sample from the DPP until there is a non-empty set
        samp = rand(rng, dpp)
    end
    return X[samp]
end
