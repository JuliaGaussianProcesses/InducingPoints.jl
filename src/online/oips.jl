"""
    OIPS(ρ_accept=0.8; η=0.95, kmax=Inf, kmin=10, ρ_remove=Inf)
    OIPS(kmax, η=0.98, kmin=10)

Online Inducing Points Selection.
Method from the paper include reference here.
"""
struct OnlineIPSelection{T,Tv<:AbstractVector{T},Tk} <: OnIPSA
    ρs::Tv # Vector of two elements corresponding to ρ_accept and ρ_remove
    kmax::Tk
    kmin::Int
    η::T
end

const OIPS = OnlineIPSelection

function Base.show(io::IO, alg::OIPS)
    return print(
        io,
        "Online Inducing Point Selection (ρ_accept : $(alg.ρs[1]), ρ_remove : $(alg.ρs[2]), kmax : $(alg.kmax))",
    )
end

function OIPS(
    ρ_accept::Real=0.8; η::Real=0.95, kmax::Real=Inf, ρ_remove::Real=Inf, kmin::Int=10
)
    0.0 <= ρ_accept <= 1.0 || throw(ArgumentError("ρ_accept should be between 0 and 1"))
    0.0 <= η <= 1.0 || throw(ArgumentError("η should be between 0 and 1"))
    ρ_remove = isinf(ρ_remove) ? sqrt(ρ_accept) : ρ_remove
    0.0 <= ρ_remove <= 1.0 || throw(ArgumentError("ρ_remove should be between 0 and 1"))
    T = promote_type(typeof(ρ_accept), typeof(ρ_remove), typeof(η))
    ρs = T[ρ_accept, ρ_remove]
    return OIPS(ρs, kmax, kmin, T(η))
end

function OIPS(kmax::Int, η::T=0.98, kmin::Real=10) where {T<:Real}
    kmax > 0 || throw(ArgumentError("kmax should be bigger than 0"))
    0.0 <= η <= 1.0 || throw(ArgumentError("η should be between 0 and 1"))
    return OIPS(T[0.95, sqrt(0.95)], kmax, kmin, η)
end

function initZ(
    rng::AbstractRNG,
    alg::OIPS,
    X::AbstractVector;
    kernel::Kernel,
    arraytype=Vector{Float64},
    kwargs...,
)
    N = length(X) # Number of samples
    N >= alg.kmin ||
        throw(ArgumentError("First batch should have at least $(alg.kmin) samples"))
    samples = sample(rng, 1:N, floor(Int, alg.kmin); replace=false)
    Z = to_vec_of_vecs(X[samples], arraytype)
    # Z = collect.(X[samples])
    Z = updateZ!(rng, Z, alg, X; kernel=kernel)
    return Z
end

function updateZ!(
    rng::AbstractRNG,
    Z::AbstractVector,
    alg::OIPS,
    X::AbstractVector;
    kernel::Kernel,
    kwargs...,
)
    return add_point!(rng, Z, alg, X, kernel)
end

function updateZ(
    rng::AbstractRNG,
    Z::AbstractVector,
    alg::OIPS,
    X::AbstractVector;
    kernel::Kernel,
    kwargs...,
)
    return add_point(rng, Z, alg, X, kernel)
end

function add_point!(
    rng::AbstractRNG, Z::AbstractVector{T}, alg::OIPS, X::AbstractVector, kernel::Kernel
) where {T}
    b = length(X)
    for i in 1:b # Parse all points from X
        kx = kernelmatrix(kernel, X[i:i], copy(Z))
        # d = find_nearest_center(X[i,:],Z.centers,kernel)[2]
        if maximum(kx) < alg.ρs[1] # If the biggest correlation is smaller than threshold add point
            push!(Z, X[i])
        end
        while length(Z) > alg.kmax ## If maximum number of points is reached, readapt the threshold
            K = kernelmatrix(kernel, copy(Z))
            m = maximum(K - Diagonal(K))
            alg.ρs[2] = alg.η * m
            remove_point!(rng, Z, alg, K)
            if alg.ρs[2] < alg.ρs[1] # Readapt the thresholds
                alg.ρs[1] = alg.η * alg.ρs[2]
            end
            @info "ρ_accept reset to : $(alg.ρs[1])"
        end
    end
    return Z
end

function add_point(
    rng::AbstractRNG, Z::AbstractVector{T}, alg::OIPS, X::AbstractVector, kernel::Kernel
) where {T}
    b = length(X)
    for i in 1:b # Parse all points from X
        kx = kernelmatrix(kernel, X[i:i], copy(Z))
        # d = find_nearest_center(X[i,:],Z.centers,kernel)[2]
        if maximum(kx) < alg.ρs[1] # If the biggest correlation is smaller than threshold add point
            Z = vcat(Z, X[i])
        end
        while length(Z) > alg.kmax ## If maximum number of points is reached, readapt the threshold
            K = kernelmatrix(kernel, copy(Z))
            m = maximum(K - Diagonal(K))
            alg.ρs[2] = alg.η * m
            Z = remove_point(rng, Z, alg, K)
            if alg.ρs[2] < alg.ρs[1] # Readapt the thresholds
                alg.ρs[1] = alg.η * alg.ρs[2]
            end
            @info "ρ_accept reset to : $(alg.ρs[1])"
        end
    end
    return Z
end

function remove_point!(rng::AbstractRNG, Z::AbstractVector, alg::OIPS, K::AbstractMatrix)
    if length(Z) > alg.kmin # Only remove points if the minimum is not reached
        overlapcount = (x -> count(x .> alg.ρs[2])).(eachrow(K))
        removable = SortedSet(findall(x -> x > 1, overlapcount))
        toremove = []
        while !isempty(removable) && length(Z) > alg.kmin
            i = sample(rng, collect(removable), Weights(overlapcount[collect(removable)]))
            connected = findall(x -> x > alg.ρs[2], K[i, :])
            overlapcount[connected] .-= 1
            outofloop = filter(x -> overlapcount[x] <= 1, connected)
            for j in outofloop
                if issubset(j, removable)
                    delete!(removable, j)
                end
            end
            push!(toremove, i)
            if issubset(i, removable)
                delete!(removable, i)
            end
        end
        deleteat!(Z, sort(toremove))
    end
    return Z
end

function remove_point(rng::AbstractRNG, Z::AbstractVector, alg::OIPS, K::AbstractMatrix)
    if length(Z) > alg.kmin # Only remove points if the minimum is not reached
        overlapcount = (x -> count(x .> alg.ρs[2])).(eachrow(K))
        removable = SortedSet(findall(x -> x > 1, overlapcount))
        toremove = []
        c = 0
        while !isempty(removable) && length(Z) > alg.kmin
            i = sample(rng, collect(removable), Weights(overlapcount[collect(removable)]))
            connected = findall(x -> x > alg.ρs[2], K[i, :])
            overlapcount[connected] .-= 1
            outofloop = filter(x -> overlapcount[x] <= 1, connected)
            for j in outofloop
                if issubset(j, removable)
                    delete!(removable, j)
                end
            end
            push!(toremove, i)
            if issubset(i, removable)
                delete!(removable, i)
            end
        end
        Z = Z[setdiff(1:end, toremove)]
    end
    return Z
end
