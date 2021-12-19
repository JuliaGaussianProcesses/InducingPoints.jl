"""
    KmeansAlg(m::Int, metric::SemiMetric=SqEuclidean(); nMarkov = 10, tol = 1e-3)

k-Means [1] initialization on the data `X` taking `m` inducing points.
The seeding is computed via [2], `nMarkov` gives the number of MCMC steps for the seeding.

## Arguments
- `k::Int` : Number of inducing points
- `metric::SemiMetric` : Metric used to compute the distances for the k-means algorithm

## Keyword Arguments
- `nMarkov::Int` : Number of random steps for the seeding
- `tol::Real` : Tolerance for the `kmeans` algorithm

[1] Arthur, D. & Vassilvitskii, S. k-means++: The advantages of careful seeding. in Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms 1027–1035 (Society for Industrial and Applied Mathematics, 2007).
[2] Bachem, O., Lucic, M., Hassani, S. H. & Krause, A. Fast and Provably Good Seedings for k-Means. Advances in Neural Information Processing Systems 29 55--63 (2016) doi:10.1109/tmtt.2005.863818.
"""
struct KmeansAlg{Tm,T} <: OffIPSA
    m::Int
    metric::Tm
    nMarkov::Int
    tol::T
    function KmeansAlg(
        m::Int, metric::SemiMetric=SqEuclidean(); nMarkov::Int=10, tol::T=1e-3
    ) where {T<:Real}
        m > 0 || throw(ArgumentError("The number of inducing points m should be positive"))
        tol > 0 || throw(ArgumentError("The tolerance tol should be positive"))
        return new{typeof(metric),T}(m, metric, nMarkov, tol)
    end
end

function inducingpoints(
    rng::AbstractRNG, alg::KmeansAlg, X::AbstractVector; weights=nothing, kwargs...
)
    if alg.m >= length(X)
        return edge_case(alg.m, length(X), X)
    end
    return kmeans_ip(
        rng, X, alg.m, alg.metric; nMarkov=alg.nMarkov, weights=weights, tol=alg.tol
    )
end

function Base.show(io::IO, alg::KmeansAlg)
    return print(io, "k-Means Selection of Inducing Points (m : $(alg.m))")
end

#Return K inducing points from X, m being the number of Markov iterations for the seeding
function kmeans_ip(
    rng::AbstractRNG,
    X::AbstractVector{T},
    nC::Int,
    metric::SemiMetric;
    nMarkov::Int=10,
    weights=nothing,
    tol=1e-3,
) where {T}
    C = kmeans_seeding(rng, X, nC, metric, nMarkov)
    C = reduce(hcat, C)
    kmeans!(reduce(hcat, X), C; weights=weights, tol=tol, distance=metric)
    if T isa Real
        return C
    else
        return ColVecs(C)
    end
end

"""Fast and efficient seeding for KMeans based on [Fast and Provably Good Seeding for k-Means](https://las.inf.ethz.ch/files/bachem16fast.pdf)"""
function kmeans_seeding(
    rng::AbstractRNG,
    X::AbstractVector, # Input data
    nC::Integer, # Number of centroids
    metric::SemiMetric, # Metric used
    nMarkov::Integer, # Number of Markov iterations
)
    nsamples = length(X) # Number of input samples
    # Preprocessing, 
    init = rand(rng, 1:nsamples) # Sample first random center
    C = collect.(X[init:init]) # Initialize the collection of centroids
    q = vec(pairwise(metric, X, C)) # Create the pairwise values between the data and the first centroid
    q = Weights(q / sum(q) .+ 1.0 / (2 * nsamples), 1) # Create weights to work with
    for _ in 2:nC
        x = X[sample(rng, q)] # weighted sampling,
        mindist = mindistance(metric, x, C) # Find the closest centroid to the random sample x
        for _ in 2:nMarkov # Iterate over nMarkov iterations
            y = X[sample(rng, q)] # Draw a new sample 
            dist = mindistance(metric, y, C) # Find the closest centroid to the random sample y
            if (dist / mindist) > rand(rng) # Acceptance step to see if y is better than x
                x = y
                mindist = dist
            end
        end
        push!(C, x isa Real ? fill(x) : x)
    end
    return C
end
