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
    ::AbstractRNG, alg::UniGrid, X::AbstractVector{T}; kwargs...
) where {T}
    ndim = length(first(X)) # Take the dimensionality
    bounds = [extrema(x -> getindex(x, i), X) for i in 1:ndim]
    Z = map(bounds) do lims
        LinRange(lims..., alg.m)
    end
    tmp = Iterators.product(Z...)
    Zm = reshape(collect(Iterators.flatten(tmp)), ndim, :)

    if T <: Real
        return Zm[:]
    else
        return ColVecs(Zm)
    end
end

function updateZ!(
    ::AbstractRNG, Z::AbstractVector, alg::UniGrid, X::AbstractVector; kwargs...
)
    ndim = length(first(Z))
    old_bounds = collect(zip(Z[1], Z[end]))
    new_bounds = [extrema(x -> getindex(x, i), X) for i in 1:ndim]
    newZ = map(old_bounds, new_bounds) do old_b, new_b
        x_start = min(old_b[1], new_b[1]) # Find the new limits
        x_stop = max(old_b[2], new_b[2])
        return LinRange(x_start, x_stop, alg.m) # readapt bounds
    end
    tmp = Iterators.product(newZ...)

    if Z isa AbstractVector{<:Real}
        Z[:] = collect(Iterators.flatten(tmp))
    elseif Z isa ColVecs
        Z.X[:] = collect(Iterators.flatten(tmp))
    end
    return Z
end

function updateZ(
    rng::AbstractRNG, Z::AbstractVector, alg::UniGrid, X::AbstractVector; kwargs...
)
    Zn = deepcopy(Z)
    return updateZ!(rng, Zn, alg, X; kwargs...)
end
