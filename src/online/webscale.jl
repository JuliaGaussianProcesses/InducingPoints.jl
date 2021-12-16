"""
    Webscale(m::Int)

Online k-means algorithm based on [1].

[1] Sculley, D. Web-scale k-means clustering. in Proceedings of the 19th international conference on World wide web - WWW ’10 1177 (ACM Press, 2010). doi:10.1145/1772690.1772862.
"""
struct Webscale <: OnIPSA
    m::Int
    v::Vector{Int}
end

function Webscale(m::Int)
    return Webscale(m, zeros(Int, m))
end

Base.show(io::IO, alg::Webscale) = print(io, "Webscale (m = ", alg.m, ")")

function inducingpoints(
    rng::AbstractRNG, alg::Webscale, X::AbstractVector; arraytype=Vector{Float64}, kwargs...
)
    length(X) >= alg.m || error(
        "Input data not big enough given desired number of inducing points : $(alg.m)"
    )
    Z = to_vec_of_vecs(X[sample(rng, 1:length(X), alg.m; replace=false)], arraytype)
    return Z
end

function add_point!(
    ::AbstractRNG, Z::AbstractVector{T}, alg::Webscale, X::AbstractVector; kwargs...
) where {T}
    d = zeros(Int, length(X))
    for i in 1:length(X)
        d[i] = find_nearest_center(X[i], Z)[1] # Save the closest IP index for each point
    end
    for i in 1:length(X)
        alg.v[d[i]] += 1
        η = 1 / alg.v[d[i]]
        Z[d[i]] = (1 - η) * Z[d[i]] + η * X[i] # Update the IP position
    end
    return Z
end

function add_point(
    ::AbstractRNG, Z::AbstractVector{T}, alg::Webscale, X::AbstractVector; kwargs...
) where {T}
    d = zeros(Int, length(X))
    Z = copy(Z)
    for i in 1:length(X)
        d[i] = find_nearest_center(X[i], Z)[1] # Save the closest IP index for each point
    end
    for i in 1:length(X)
        alg.v[d[i]] += 1
        η = 1 / alg.v[d[i]]
        Z[d[i]] = (1 - η) * Z[d[i]] + η * X[i] # Update the IP position
    end
    return Z
end
