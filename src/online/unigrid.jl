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
    return Z
end

function updateZ!(
    ::AbstractRNG, Z::AbstractVector, alg::UniGrid, X::AbstractVector; kwargs...
)
    ndim = length(Z)
    new_bounds = [extrema(x -> getindex(x, i), X) for i in 1:ndim]
    map!(Z, Z, new_bounds) do Z_d, new_b
        x_start = min(Z_d.start, new_b[1]) # Find the new limits
        x_stop = max(Z_d.stop, new_b[2])
        return LinRange(x_start, x_stop, alg.m) # readapt bounds
    end
    return Z
end

function updateZ(rng::AbstractRNG, Z::AbstractVector, alg::UniGrid, X::AbstractVector)
    Zn = deepcopy(Z)
    return updateZ!(rng, Zn, alg, X)
end

export UniformGrid
struct UniformGrid{N, T} <: AbstractVector{T} 
    proditer::Iterators.ProductIterator{NTuple{N, LinRange{T, Int64}}}

    function UniformGrid(proditer::Iterators.ProductIterator{NTuple{N, LinRange{T, Int64}}}) where {N, T}
        new{N,T}(proditer)
    end
end



import Base: getindex
Base.getindex(ug::UniformGrid, i) = collect(first(Iterators.drop(ug.proditer, i)))


### show still this need more improvement
Base.show(io::IO, ug::UniformGrid) = print(io, "Lazy $(length.(ug.proditer.iterators)) uniform grid")

Base.show(io::IO, ::MIME"text/plain", ug::UniformGrid) = Base.show(io, ug)

import Base: length, size
Base.length(ug::UniformGrid) = prod(length.(ug.proditer.iterators))

# alternative: (typeof(t).parameters[1], prod(length.(ug.proditer.iterators)))
Base.size(ug::UniformGrid) = length.(ug.proditer.iterators)