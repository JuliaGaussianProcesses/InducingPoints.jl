module InducingPoints

import Base: rand, show

using AbstractGPs: AbstractGPs
using Clustering: kmeans!
using DataStructures
using DeterminantalPointProcesses: DPP
using Distances
using KernelFunctions
using KernelFunctions: ColVecs, RowVecs, vec_of_vecs
using LinearAlgebra#: Symmetric, Eigen, eigen, eigvals, I, logdet, diag, norm
using Random: rand, bitrand, AbstractRNG, MersenneTwister, GLOBAL_RNG
using StatsBase: Weights, sample, wsample

export AbstractInducingPointsSelectionAlg

## Generic functions

export inducingpoints
export initZ
export updateZ!, updateZ

## Offline algorithms
export KmeansAlg
export RandomSubset
export StdDPP, kDPP
export Greedy

## Online algorithms
export OIPS
export SeqDPP
export StreamKmeans
export UniGrid
export Webscale

## Custom output object
export UniformGrid

const jitt = 1e-5

abstract type AbstractInducingPointsSelectionAlg end

const AIPSA = AbstractInducingPointsSelectionAlg

abstract type OfflineInducingPointsSelectionAlg <: AIPSA end

const OffIPSA = OfflineInducingPointsSelectionAlg

abstract type OnlineInducingPointsSelectionAlg <: AIPSA end

const OnIPSA = OnlineInducingPointsSelectionAlg

## Wrapper for matrices
"""
     inducingpoints([rng::AbstractRNG], alg::AIPSA, X::AbstractVector; [kwargs...])
     inducingpoints([rng::AbstractRNG], alg::AIPSA, X::AbstractMatrix; obsdim=1, [kwargs...])

Select inducing points according to the algorithm `alg`. For some algorithms, additional keyword arguments are required. 
"""
inducingpoints

function inducingpoints(
    rng::AbstractRNG, alg::AIPSA, X::AbstractMatrix; obsdim=1, kwargs...
)
    return inducingpoints(rng, alg, vec_of_vecs(X; obsdim=obsdim); kwargs...)
end

function inducingpoints(alg::AIPSA, X::AbstractMatrix; obsdim=1, kwargs...)
    return inducingpoints(GLOBAL_RNG, alg, X; obsdim=obsdim, kwargs...)
end

## Wrapper for the RNG
function inducingpoints(alg::AIPSA, X::AbstractVector; kwargs...)
    return inducingpoints(GLOBAL_RNG, alg, X; kwargs...)
end

## Online IP selection functions 
@doc raw"""
    updateZ!([rng::AbstractRNG], Z::AbstractVector, alg::OnIPSA, X::AbstractVector; [kwargs...])

Update inducing points `Z` with data `X` and algorithm `alg`. Requires additional keyword arguments
for some algorithms. Also see `InducingPoints`.
"""
updateZ!

function updateZ!(Z::AbstractVector, alg::OnIPSA, X::AbstractVector; kwargs...)
    return updateZ!(GLOBAL_RNG, Z, alg, X; kwargs...)
end
# Default behavior is to simply add points
function updateZ!(
    rng::AbstractRNG, Z::AbstractVector, alg::OnIPSA, X::AbstractVector; kwargs...
)
    return add_point!(rng, Z, alg, X; kwargs...)
end

@doc raw"""
    updateZ([rng::AbstractRNG], Z::AbstractVector, alg::OnIPSA, X::AbstractVector; kwargs...)

Return new vector of inducing points `Z` with data `X` and algorithm `alg` without changing the original one
"""
updateZ

function updateZ(Z::AbstractVector, alg::OnIPSA, X::AbstractVector; kwargs...)
    return updateZ(GLOBAL_RNG, Z, alg, X; kwargs...)
end
# Default behavior is to simply add points
function updateZ(
    rng::AbstractRNG, Z::AbstractVector, alg::OnIPSA, X::AbstractVector; kwargs...
)
    return add_point(rng, Z, alg, X; kwargs...)
end

## Offline algorithms
include("offline/kmeans.jl")
include("offline/randomsubset.jl")
include("offline/stddpp.jl")
include("offline/kdpp.jl")
include("offline/greedyip.jl")

## Online algorithms
include("online/seqdpp.jl")
include("online/streamkmeans.jl")
include("online/webscale.jl")
include("online/oips.jl")
include("online/unigrid.jl")

## Utilities
include("utils.jl")

@deprecate initZ inducingpoints

end
