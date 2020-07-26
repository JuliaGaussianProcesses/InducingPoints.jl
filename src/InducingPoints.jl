module InducingPoints

export InducingPoints, AbstractInducingPoints
export Webscale, OIPS, Kmeans, kDPP, StdDPP, SeqDPP, Greedy, UniformSampling, UniGrid
export init!, add_point!, remove_point!

using StatsBase: Weights, sample
using DeterminantalPointProcesses
using LinearAlgebra#: Symmetric, Eigen, eigen, eigvals, I, logdet, diag, norm
using Clustering: kmeans!
using Distances
using DataStructures
using KernelFunctions
using KernelFunctions: ColVecs
using Random: rand, bitrand, AbstractRNG, MersenneTwister
using Flux.Optimise
import Base: rand, show

const jitt = 1e-5

abstract type AbstractInducingPoints{S, TZ<:AbstractVector{S}} <: AbstractVector{S} end

const AIP = AbstractInducingPoints

abstract type OfflineInducingPoints{S, TZ<:AbstractVector{S}} <: AIP{S, TZ} end

const OffIP = OfflineInducingPoints

abstract type OnlineInducingPoints{S, TZ<:AbstractVector{S}} <: AIP{S, TZ} end

const OnIP = OnlineInducingPoints


struct InducingPoints{S,TZ<:AbstractVector{S}} <: InducingPoints{S,TZ}
    Z::TZ
end

init!(ip::AIP, args...) = nothing
add_point!(ip::AIP, args...) = nothing
remove_point!(ip::AIP, args...) = nothing

Base.size(Z::AIP) = size(Z.Z)
Base.getindex(Z::AIP, i::Int) = getindex(Z.Z, i)
Base.getindex(Z::AIP, i::Int, j::Int) = getindex(Z.Z, i, j)

function __init__()
    @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" include("opt_IP.jl")
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
