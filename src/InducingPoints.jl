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
using StatsBase: Weights, sample

export AbstractInducingPointsSelectionAlg

## Generic functions

export inducingpoints
export initZ
export updateZ!

## Offline algorithms
export KmeansAlg
export RandomSubset
export StdDPP, kDPP
export Greedy

## Online algorithms
export OIPS
export SeqDPP
export StreamKmeans
export UniformGrid
export Webscale

const jitt = 1e-5

abstract type AbstractInducingPointsSelectionAlg end

const AIPSA = AbstractInducingPointsSelectionAlg

abstract type OfflineInducingPointsSelectionAlg <: AIPSA end

const OffIPSA = OfflineInducingPointsSelectionAlg

abstract type OnlineInducingPointsSelectionAlg <: AIPSA end

const OnIPSA = OnlineInducingPointsSelectionAlg

## Wrapper for matrices
"""
     inducingpoints([rng::AbstractRNG], alg::OffIPSA, X::AbstractVector; kwargs...)
     inducingpoints([rng::AbstractRNG], alg::OffIPSA, X::AbstractMatrix; obsdim=1, kwargs...)

Select inducing points according to the algorithm `alg`.
"""
inducingpoints

function inducingpoints(
    rng::AbstractRNG, alg::AIPSA, X::AbstractMatrix; obsdim=1, kwargs...
)
    return inducingpoints(rng, alg, vec_of_vecs(X; obsdim=obsdim), kwargs...)
end

function inducingpoints(alg::AIPSA, X::AbstractMatrix; obsdim=1, kwargs...)
    return inducingpoints(GLOBAL_RNG, alg, X; obsdim=obsdim, kwargs...)
end

## Wrapper for rng generator
function inducingpoints(alg::AIPSA, X::AbstractVector; kwargs...)
    return inducingpoints(GLOBAL_RNG, alg, X; kwargs...)
end

## Online IP selection functions 
"""
     initZ([rng::AbstractRNG], alg::OnIPSA, X::AbstractVector; kwargs...)
     initZ([rng::AbstractRNG], alg::OnIPSA, X::AbstractMatrix; obsdim=1, kwargs...)

Select inducing points according to the algorithm `alg` and return a Vector of Vector.
"""
initZ

initZ(Z::OnIPSA, X::AbstractVector; kwargs...) = initZ(GLOBAL_RNG, Z, X; kwargs...)

function initZ(alg::OnIPSA, X::AbstractMatrix; obsdim=1, kwargs...)
    return initZ(GLOBAL_RNG, alg, X; obsdim=obsdim, kwargs...)
end

function initZ(rng::AbstractRNG, alg::OnIPSA, X::AbstractMatrix; obsdim=1, kwargs...)
    return initZ(rng, vec_of_vecs(X; obsdim=obsdim), X; kwargs...)
end

"""
    updateZ!([rng::AbstractRNG], Z::AbstractVector, alg::OnIPSA, X::AbstractVector; kwargs...)

Update inducing points `Z` with data `X` and algorithm `alg`
"""
updateZ!

function updateZ!(Z::AbstractVector, alg::OnIPSA, X::AbstractVector; kwargs...)
    return updateZ!(GLOBAL_RNG, Z, alg, X; kwargs...)
end
function updateZ!(
    rng::AbstractRNG, Z::AbstractVector, alg::OnIPSA, X::AbstractVector; kwargs...
)
    return add_point!(rng, Z, alg, X; kwargs...)
end

## Offline algorithms
include(joinpath("offline", "kmeans.jl"))
include(joinpath("offline", "randomsubset.jl"))
include(joinpath("offline", "stddpp.jl"))
include(joinpath("offline", "kdpp.jl"))
include(joinpath("offline", "greedyip.jl"))

## Online algorithms
include(joinpath("online", "seqdpp.jl"))
include(joinpath("online", "streamkmeans.jl"))
include(joinpath("online", "webscale.jl"))
include(joinpath("online", "oips.jl"))
include(joinpath("online", "unigrid.jl"))

## Utilities
include("utils.jl")

end
