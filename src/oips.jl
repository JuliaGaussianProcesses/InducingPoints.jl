"""
    OIPS(ρ_accept=0.8, opt= ADAM(0.001); η = 0.95, kmax = Inf, ρ_remove = Inf )
    OIPS(kmax, η, opt= ADAM(0.001))

Online Inducing Points Selection.
Method from the paper include reference here.
"""
mutable struct OIPS{S,TZ<:AbstractVector{S}} <: OnIP{S,TZ}
    ρ_accept::Float64
    ρ_remove::Float64
    kmax::Float64
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
    kmax = Inf,
    ρ_remove::Real = Inf,
)
    @assert 0.0 <= ρ_accept <= 1.0 "ρ_accept should be between 0 and 1"
    @assert 0.0 <= η <= 1.0 "η should be between 0 and 1"
    ρ_remove = isinf(ρ_remove) ? sqrt(ρ_accept) : ρ_remove
    @assert 0.0 <= ρ_remove <= 1.0 "ρ_remove should be between 0 and 1"
    return OIPS(
        ρ_accept,
        ρ_remove,
        kmax,
        η,
        0,
        [],
    )
end

function OIPS(kmax::Int, η::Real = 0.98)
    @assert kmax > 0 "kmax should be bigger than 0"
    @assert 0.0 <= η <= 1.0 "η should be between 0 and 1"
    return OIPS(
        0.95,
        sqrt(0.95),
        kmax,
        η,
        0,
        [],
    )
end

function OIPS(alg::OIPS, X::AbstractVector)
    N = size(X, 1)
    N >= 10 || "First batch should have at least 10 samples"
    samples = sample(1:N, 10, replace = false)
    return OIPS(alg.ρ_accept, alg.ρ_remove, alg.kmax, alg.η, 10, Z)
end

function init(alg::OIPS, X::AbstractVector, gp)
    alg = OIPS(alg, X)
    update!(alg, X, gp)
    return alg
end

function update!(alg::OIPS, X::AbstractVector, gp)
    add_point!(alg, X, gp)

end

function add_point!(alg::OIPS, X::AbstractVector, gp)
    b = size(X, 1)
    for i = 1:b # Parse all points from X
        k = kernelmatrix(kernel(gp), X[i], alg.Z)
        # d = find_nearest_center(X[i,:],alg.centers,kernel)[2]
        if maximum(k) < alg.ρ_accept #If biggest correlation is smaller than threshold add point
            alg.Z = push!(alg.Z, X[i])
            alg.k += 1
        end
        while alg.k > alg.kmax ## If maximum number of points is reached, readapt the threshold
            K = kernelmatrix(kernel(gp), alg.Z)
            m = maximum(K - Diagonal(K))
            alg.ρ_remove = alg.η * m
            remove_point!(alg, K, kernel(gp))
            if alg.ρ_remove < alg.ρ_accept
                alg.ρ_accept = alg.η * alg.ρ_remove
            end
            @info "ρ_accept reset to : $(alg.ρ_accept)"
        end
    end
end

function remove_point!(alg::OIPS, K, kernel)
    if alg.k > 10
        overlapcount = (x -> count(x .> alg.ρ_remove)).(eachrow(K))
        removable = SortedSet(findall(x -> x > 1, overlapcount))
        toremove = []
        c = 0
        while !isempty(removable) && alg.k > 10
            i = sample(
                collect(removable),
                Weights(overlapcount[collect(removable)]),
            )
            connected = findall(x -> x > alg.ρ_remove, K[i, :])
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
            alg.k -= 1
        end
        alg.Z = alg.Z[setdiff(1:alg.k, toremove)]
        alg.k = size(alg.Z, 1)
    end
end
