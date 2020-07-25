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
using Random: rand, bitrand, AbstractRNG, MersenneTwister
using Flux.Optimise
import Base: rand, show

const jitt = 1e-5

abstract type AbstractInducingPoints{T, M<:AbstractMatrix{T}} <: AbstractMatrix{T} end

const AIP = AbstractInducingPoints

struct InducingPoints{T,M<:AbstractMatrix{T}} <: InducingPoints{T,M}
    Z::M
end

init!(ip::AIP, args...) = nothing
add_point!(ip::AIP, args...) = nothing
remove_point!(ip::AIP, args...) = nothing

Base.size(Z::AIP) = size(Z.Z)
Base.size(Z::AIP, i::Int) = size(Z.Z, i)
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
@requi


end
