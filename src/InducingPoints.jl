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

update!(Z::OnIP, X::AbstractMatrix, args...; obsdim = 1) = init(Z, vec_of_vecs(X, obsdim = obsdim), args...)

update!(Z::OnIP, X::AbstractVector, args...) = add_point!(Z, X, args...)

add_point!(Z::OnIP, X::AbstractVector, ::Kernel) = add_point!(Z, X)

remove_point!(::OnIP, args...) = nothing

## Offline algorithms
include(joinpath("offline", "kmeans.jl"))
include(joinpath("offline", "randomsubset.jl"))
include(joinpath("offline", "stddpp.jl"))
include("kdpp.jl")
include("greedyip.jl")

## Online algorithms
include("seqdpp.jl")
include("streamkmeans.jl")
include("webscale.jl")
include("oips.jl")
include("unigrid.jl")



end
