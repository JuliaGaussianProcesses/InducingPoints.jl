"""
    OIPS(ρ_accept=0.8, opt= ADAM(0.001); η = 0.95, kmax = Inf, kmin = 10, ρ_remove = Inf )
    OIPS(kmax, η = 0.98, kmin = 10)

Online Inducing Points Selection.
Method from the paper include reference here.
"""
mutable struct OIPS{S,TZ<:AbstractVector{S}} <: OnIP{S,TZ}
    ρ_accept::Float64
    ρ_remove::Float64
    kmax::Float64
    kmin::Float64
    η::Float64
    k::Int
    Z::TZ
end

Base.show(io::IO, alg::OIPS) = print(
    io,
    "Online Inducing Point Selection (ρ_in : $(alg.ρ_accept), ρ_out : $(alg.ρ_remove), kmax : $(alg.kmax))",
)

function OIPS(
    ρ_accept::Real = 0.8;
    η::Real = 0.95,
    kmax::Real = Inf,
    ρ_remove::Real = Inf,
    kmin::Real = 10,
)
    @assert 0.0 <= ρ_accept <= 1.0 "ρ_accept should be between 0 and 1"
    @assert 0.0 <= η <= 1.0 "η should be between 0 and 1"
    ρ_remove = isinf(ρ_remove) ? sqrt(ρ_accept) : ρ_remove
    @assert 0.0 <= ρ_remove <= 1.0 "ρ_remove should be between 0 and 1"
    return OIPS(
        ρ_accept,
        ρ_remove,
        kmax,
        kmin,
        η,
        0,
        [],
    )
end

function OIPS(kmax::Int, η::Real = 0.98, kmin::Real = 10)
    @assert kmax > 0 "kmax should be bigger than 0"
    @assert 0.0 <= η <= 1.0 "η should be between 0 and 1"
    return OIPS(
        0.95,
        sqrt(0.95),
        kmax,
        kmin,
        η,
        0,
        [],
    )
end

function OIPS(Z::OIPS, X::AbstractVector, k::Kernel)
    N = size(X, 1)
    N >= Z.kmin || "First batch should have at least $(Z.kmin) samples"
    samples = sample(1:N, 10, replace = false)
    return OIPS(Z.ρ_accept, Z.ρ_remove, Z.kmax, Z.kmin, Z.η, 10, deepcopy(X[samples]))
end

function init(alg::OIPS, X::AbstractVector, k::Kernel)
    alg = OIPS(alg, X)
    update!(alg, X, gp)
    return alg
end

function update!(Z::OIPS, X::AbstractVector, k::Kernel)
    add_point!(Z, X, k)
end

function add_point!(Z::OIPS, X::AbstractVector, k::Kernel)
    b = size(X, 1)
    for i = 1:b # Parse all points from X
        k = kernelmatrix(k, X[i], Z)
        # d = find_nearest_center(X[i,:],alg.centers,kernel)[2]
        if maximum(k) < Z.ρ_accept #If biggest correlation is smaller than threshold add point
            alg.Z = push!(Z.Z, deepcopy(X[i]))
            alg.k += 1
        end
        while alg.k > alg.kmax ## If maximum number of points is reached, readapt the threshold
            K = kernelmatrix(k, Z)
            m = maximum(K - Diagonal(K))
            Z.ρ_remove = Z.η * m
            remove_point!(Z, K, k)
            if Z.ρ_remove < Z.ρ_accept
                Z.ρ_accept = Z.η * Z.ρ_remove
            end
            @info "ρ_accept reset to : $(Z.ρ_accept)"
        end
    end
end

function remove_point!(Z::OIPS, K::AbstractMatrix, kernel::Kernel)
    if Z.k > Z.kmin
        overlapcount = (x -> count(x .> Z.ρ_remove)).(eachrow(K))
        removable = SortedSet(findall(x -> x > 1, overlapcount))
        toremove = []
        c = 0
        while !isempty(removable) && Z.k > 10
            i = sample(
                collect(removable),
                Weights(overlapcount[collect(removable)]),
            )
            connected = findall(x -> x > Z.ρ_remove, K[i, :])
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
            Z.k -= 1
        end
        Z.Z = Z.Z[setdiff(1:Z.k, toremove)]
        Z.k = size(Z.Z, 1)
    end
end
