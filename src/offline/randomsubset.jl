@doc raw"""
    RandomSubset(m::Int)

## Arguments
- `m::Int`: Number of inducing points

Uniform sampling of a subset of `m` points of the data.
"""
struct RandomSubset <: OffIPSA
    m::Int
    function RandomSubset(m::Int)
        m > 0 || throw(ArgumentError("The number of inducing points m should be positive"))
        return new(m)
    end
end

"""
     inducingpoints([rng::AbstractRNG], alg::RandomSubset, X::AbstractVector; [weights::Vector=nothing])
     inducingpoints([rng::AbstractRNG], alg::RandomSubset, X::AbstractMatrix; obsdim=1, [weights::Vector=nothing])

Select inducing points by taking a Random Subset. Optionally accepts a weight vector for the inputs `X`.
"""
function inducingpoints(
    rng::AbstractRNG, alg::RandomSubset, X::AbstractVector; weights=nothing, kwargs...
)
    nSamples = length(X)
    alg.m > nSamples && error(
        "Number of inducing points larger than the input collection size ($(alg.m) > $(length(X))",
    )
    alg.m == nSamples && return X

    samp = if isnothing(weights)
        sample(rng, 1:nSamples, alg.m; replace=false)
    else
        wsample(rng, 1:nSamples, weights, alg.m; replace=false)
    end
    return X[samp]
end
