module InducingPoints

import Base: rand, show

using AbstractGPs
using Clustering: kmeans!
using DataStructures
using DeterminantalPointProcesses: DPP
using Distances
using KernelFunctions
using KernelFunctions: ColVecs, RowVecs, vec_of_vecs
using LinearAlgebra#: Symmetric, Eigen, eigen, eigvals, I, logdet, diag, norm
using Random: rand, bitrand, AbstractRNG, MersenneTwister
using StatsBase: Weights, sample


export AbstractInducingPointsSelectionAlg

## Generic function

export inducingpoints

## Offline algorithms
export KmeansAlg
export RandomSubset
export StdDPP, kDPP
export Greedy

## Online algorithms
export Webscale, OIPS, SeqDPP, UniformGrid, StreamKmeans
export init, update!

const jitt = 1e-5

"""
     inducingpoints([rng::AbstractRNG}, alg::AbstractInducingPointsSelectionAlg, X::AbstractVector; kwargs...)
     inducingpoints([rng::AbstractRNG], alg::AbstractInducingPointsSelectionAlg, X::AbstractMatrix; obsdim=1, kwargs...)

Select inducing points according to the algorithm `alg`.
"""
inducingpoints

abstract type AbstractInducingPoints end

const AIPSA = AbstractInducingPointsSelectionAlg

abstract type OfflineInducingPointsSelectionAlg <: AIPSA end

const OffIPSA = OfflineInducingPointsSelectionAlg

abstract type OnlineInducingPointsSelectionAlg <: AIPSA end

const OnIPSA = OnlineInducingPointsSelectionAlg

## Wrapper for matrices
function inducingpoints(rng::AbstractRNG, alg::AIPSA, X::AbstractMatrix; obsdim=1, kwargs...)
     return inducingpoints(alg, vec_of_vecs(X; obsdim=obsdim), kwargs...)
end

function inducingpoints(alg::AIPSA, X::AbstractMatrix; obsdim=1, kwargs...)
     return inducingpoints(GLOBAL_RNG, alg, X; obsdim=obsdim, kwargs...)
end

init(Z::OnIPSA, X::AbstractVector) = init(Z, X)

update!(Z::OnIP, X::AbstractMatrix; obsdim = 1, kwargs...) = init(Z, vec_of_vecs(X, obsdim = obsdim); kwargs...)
update!(Z::OnIP, X::AbstractVector; kwargs...) = add_point!(Z, X; kwargs...) # Standard use is update!

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



end
