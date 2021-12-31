"""
    UniGrid(m::Int)

where `m` is the number of points on each dimension.
Adaptive uniform grid based on [1].
The resulting inducing points are stored in the memory-efficient custom type 
`UniformGrid`. 

[1] Moreno-Muñoz, P., Artés-Rodríguez, A. & Álvarez, M. A. Continual Multi-task Gaussian Processes. (2019).
"""
struct UniGrid <: OnIPSA
    m::Int # Number of points per dimension
end

Base.show(io::IO, Z::UniGrid) = print(io, "Uniform grid with side length $(Z.m).")

function inducingpoints(
    ::AbstractRNG,
    alg::UniGrid,
    X::Union{AbstractVector{<:Real},AbstractVector{<:AbstractVector{<:Real}}};
    kwargs...,
)
    ndim = length(first(X)) # Take the dimensionality
    bounds = [extrema(x -> getindex(x, i), X) for i in 1:ndim]
    Z = map(bounds) do lims
        LinRange(lims..., alg.m)
    end
    proditer = Iterators.product(Z...)
    return UniformGrid(proditer)
end

function updateZ!(
    ::AbstractRNG, Z::AbstractVector, alg::UniGrid, X::AbstractVector; kwargs...
)
    ndim = length(X[1])
    new_bounds = [extrema(x -> getindex(x, i), X) for i in 1:ndim]
    Zn = map(Z.proditer.iterators, new_bounds) do old_Z, new_b
        x_start = min(old_Z.start, new_b[1])
        x_stop = max(old_Z.stop, new_b[2])
        return LinRange(x_start, x_stop, alg.m)
    end
    Z.proditer = Iterators.product(Zn...)
    return Z
end

function updateZ(
    rng::AbstractRNG, Z::AbstractVector, alg::UniGrid, X::AbstractVector; kwargs...
)
    Zn = deepcopy(Z)
    return updateZ!(rng, Zn, alg, X; kwargs...)
end

"""
    UniformGrid{T,Titer} <: AbstractVector{T}

A memory-efficient custom object representing a wrapper around a `Iterators.ProductIterator`.
Supports broadcasting and other relevant array methods, and avoids explicitly computing all points on the grid.  
"""
mutable struct UniformGrid{T,Titer} <: AbstractVector{T}
    proditer::Titer
end

function UniformGrid(proditer::Iterators.ProductIterator{NTuple{N,S}}) where {N,S}
    T = Vector{eltype(first(proditer.iterators))}
    return UniformGrid{T,typeof(proditer)}(proditer)
end

import Base: getindex, broadcastable, eachindex, length, size, iterate, eltype, IndexStyle

_getelement(x::NTuple{1,<:Real}) = only(x)
_getelement(x::NTuple) = collect(x)

function Base.getindex(ug::UniformGrid, i::Integer)
    return _getelement(first(Iterators.drop(ug.proditer, i - 1)))
end

function Base.getindex(ug::UniformGrid, r::UnitRange)
    return [
        _getelement(x) for
        x in Iterators.take(Iterators.drop(ug.proditer, first(r) - 1), length(r))
    ]
end

Base.getindex(ug::UniformGrid, ::Colon) = Base.getindex(ug, 1:length(ug))

function Base.broadcastable(ug::UniformGrid)
    r = similar(1:1, eltype(ug.proditer), length(ug))
    copyto!(r, ug.proditer)
end

Base.length(ug::UniformGrid) = prod(length, ug.proditer.iterators)
Base.size(ug::UniformGrid) = (prod(length, ug.proditer.iterators),)

# slows iteration, but makes it consistent with iterating over an AbstractVector
function Base.iterate(ug::UniformGrid, state...)
    r = Base.iterate(ug.proditer, state...)
    r === nothing && return nothing
    return (_getelement(r[1]), r[2])
end

import KernelFunctions: pairwise, pairwise!
function pairwise(d::PreMetric, x::UniformGrid)
    return KernelFunctions.Distances_pairwise(d, x.proditer)
end

function pairwise!(out::AbstractMatrix, d::PreMetric, x::UniformGrid)
    return KernelFunctions.Distances.pairwise!(out, d, x.proditer)
end

Base.eltype(ug::UniformGrid) = typeof(ug[1])

Base.IndexStyle(::UniformGrid) = IndexLinear()

## show needs more improvement, maybe
function Base.show(io::IO, ug::UniformGrid)
    return print(io, "$(length.(ug.proditer.iterators)) uniform grid")
end

function Base.show(io::IO, ::MIME"text/plain", ug::UniformGrid)
    println(io, "$(length.(ug.proditer.iterators)) uniform grid with edges")
    pr(iter) = println(io, [iter[1], iter[end]])
    return pr.(ug.proditer.iterators)
end
