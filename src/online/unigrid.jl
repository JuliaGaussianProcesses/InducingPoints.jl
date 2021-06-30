"""
    UniGrid(K::Int)

where `m` is the number of points on each dimension
Adaptive uniform grid based on [1]

[1] Moreno-Muñoz, P., Artés-Rodríguez, A. & Álvarez, M. A. Continual Multi-task Gaussian Processes. (2019).
"""
struct UniGrid{} <: OnIPSA
    m::Int # Number of points per dimension
end

Base.show(io::IO, Z::UniGrid) = print(io, "Uniform grid with side length $(Z.K).")

function init(
    ::AbstractRNG,
    alg::UniGrid,
    X::Union{AbstractVector{<:Real},AbstractVector{<:AbstractVector{<:Real}}},
)
    ndim = length(first(X)) # Take the dimensionality
    bounds = [extrema(x -> getindex(x, i), X) for i in 1:ndim]
    Z = map(bounds) do lims
        LinRange(lims..., alg.m)
    end
    return Z
end

function add_point!(::AbstractRNG, Z::AbstractVector, alg::UniGrid, X::AbstractVector)
    ndim = length(Z)
    new_bounds = [extrema(x -> getindex(x, i), X) for i in 1:ndim]
    map!(Z, Z, new_bounds) do Z_d, new_b
        x_start = min(Z_d.start, new_b[1]) # Find the new limits
        x_stop = max(Z_d.stop, new_b[2])
        return LinRange(x_start, x_stop, alg.m) # readapt bounds
    end
    return Z
end
