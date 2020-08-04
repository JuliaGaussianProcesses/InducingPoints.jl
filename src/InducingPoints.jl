module InducingPoints

export AbstractInducingPoints
export KmeansIP
export OptimIP
export Webscale, OIPS, Kmeans, kDPP, StdDPP, SeqDPP, Greedy, UniformSampling, UniGrid
export init!, add_point!, remove_point!

using StatsBase: Weights, sample
using DeterminantalPointProcesses
using AbstractGPs
using LinearAlgebra#: Symmetric, Eigen, eigen, eigvals, I, logdet, diag, norm
using Clustering: kmeans!
using Distances
using DataStructures
using KernelFunctions
using KernelFunctions: ColVecs
using Random: rand, bitrand, AbstractRNG, MersenneTwister
using Requires
import Base: rand, show

const jitt = 1e-5

abstract type AbstractInducingPoints{S, TZ<:AbstractVector{S}} <: AbstractVector{S} end

const AIP = AbstractInducingPoints

abstract type OfflineInducingPoints{S, TZ<:AbstractVector{S}} <: AIP{S, TZ} end

const OffIP = OfflineInducingPoints

abstract type OnlineInducingPoints{S, TZ<:AbstractVector{S}} <: AIP{S, TZ} end

const OnIP = OnlineInducingPoints

Base.size(Z::AIP) = size(Z.Z)
Base.length(Z::AIP) = length(Z.Z)
Base.getindex(Z::AIP, i::Int) = getindex(Z.Z, i)
Base.getindex(Z::AIP, i::Int, j::Int) = getindex(getindex(Z.Z, i), j)
Base.vec(Z::AIP) = Z.Z

update!(Z::OnIP, X::AbstractVector) = error("`update!` is not implemented for type $(typeof(Z))")

struct CustomInducingPoints{S,TZ<:AbstractVector{S}} <: OffIP{S,TZ}
     Z::TZ
end

function Base.convert(::Type{<:AbstractInducingPoints}, X::AbstractVector)
    CustomInducingPoints(X)
end

function __init__()
    @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" include("optimIP.jl")
end

include("seqdpp.jl")
include("kdpp.jl")
include("stddpp.jl")
include("streamingkmeans.jl")
include("webscale.jl")
include("oips.jl")
include("kmeans.jl")
include("greedy.jl")
include("uniform.jl")
include("unigrid.jl")



end
