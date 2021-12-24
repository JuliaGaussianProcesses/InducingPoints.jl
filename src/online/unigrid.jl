"""
    UniGrid(m::Int)

where `m` is the number of points on each dimension
Adaptive uniform grid based on [1]

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

export UniformGrid
mutable struct UniformGrid{T,Titer} <: AbstractVector{T}
    proditer::Titer
end

function UniformGrid(proditer::Iterators.ProductIterator{NTuple{N,S}}) where {N,S}
    T = Vector{eltype(first(proditer.iterators))}
    return UniformGrid{T,typeof(proditer)}(proditer)
end

import Base: getindex, broadcastable, eachindex, length, size, enumerate, eltype, IndexStyle
Base.getindex(ug::UniformGrid, i) = _getelement(first(Iterators.drop(ug.proditer, i - 1)))
_getelement(x::NTuple{1,<:Real}) = only(x)
_getelement(x::NTuple) = collect(x)

function Base.getindex(ug::UniformGrid, r::UnitRange)
    return [
        collect(x) for
        x in Iterators.take(Iterators.drop(ug.proditer, first(r) - 1), length(r))
    ]
end

Base.broadcastable(ug::UniformGrid) = Base.broadcastable(ug.proditer)[:]

Base.eachindex(ug::UniformGrid) = Base.OneTo(length(ug))

Base.length(ug::UniformGrid) = prod(length, ug.proditer.iterators)
Base.size(ug::UniformGrid) = (prod(length, ug.proditer.iterators),)

Base.enumerate(ug::UniformGrid) = Base.enumerate(ug.proditer)

Base.eltype(ug::UniformGrid) = typeof(ug[1])

Base.IndexStyle(::UniformGrid) = IndexLinear()

## show needs more improvement, maybe
function Base.show(io::IO, ug::UniformGrid)
    return print(io, "Lazy $(length.(ug.proditer.iterators)) uniform grid")
end

function Base.show(io::IO, ::MIME"text/plain", ug::UniformGrid)
    println("$(length.(ug.proditer.iterators)) uniform grid with edges")
    pr(iter) = println([iter[1], iter[end]])
    return pr.(ug.proditer.iterators)
end
