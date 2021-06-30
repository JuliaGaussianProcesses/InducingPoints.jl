"""
    OIPS(ρ_accept=0.8, opt= ADAM(0.001); η = 0.95, kmax = Inf, kmin = 10, ρ_remove = Inf )
    OIPS(kmax, η = 0.98, kmin = 10)

Online Inducing Points Selection.
Method from the paper include reference here.
"""
struct OnlineIPSelection{T} <: OnIPSA{S,TZ}
    ρ_accept::T
    ρ_remove::T
    kmax::Int
    kmin::Int
    η::T
end

const OIPS = OnlineIPSelection

function Base.show(io::IO, Z::OIPS)
    return print(
        io,
        "Online Inducing Point Selection (ρ_in : $(Z.ρ_accept), ρ_out : $(Z.ρ_remove), kmax : $(Z.kmax))",
    )
end

function OIPS(
    ρ_accept::Real=0.8; η::Real=0.95, kmax::Real=Inf, ρ_remove::Real=Inf, kmin::Real=10.0
)
    0.0 <= ρ_accept <= 1.0 || throw(ArgumentError("ρ_accept should be between 0 and 1"))
    0.0 <= η <= 1.0 || throw(ArugmentError("η should be between 0 and 1"))
    ρ_remove = isinf(ρ_remove) ? sqrt(ρ_accept) : ρ_remove
    0.0 <= ρ_remove <= 1.0 || throw(ArgumentError("ρ_remove should be between 0 and 1"))
    return OIPS(ρ_accept, ρ_remove, kmax, kmin, η)
end

function OIPS(kmax::Int, η::Real=0.98, kmin::Real=10)
    kmax > 0 || throw(ArgumentError("kmax should be bigger than 0"))
    0.0 <= η <= 1.0 || throw(ArugmentError("η should be between 0 and 1"))
    return OIPS(0.95, sqrt(0.95), kmax, kmin, η)
end
function OIPS(Z::OIPS, X::AbstractVector)
    N = size(X, 1)
    N >= Z.kmin || error("First batch should have at least $(Z.kmin) samples")
    samples = sample(1:N, floor(Int, Z.kmin); replace=false)
    return OIPS(Z.ρ_accept, Z.ρ_remove, Z.kmax, Z.kmin, Z.η, 10, Vector.(X[samples]))
end

function init(rng::AbstractRNG, alg::OIPS, X::AbstractVector; kernel::Kernel, kwargs...)
    N = length(X) # Number of samples
    N >= Z.kmin ||
        throw(ArgumentError("First batch should have at least $(Z.kmin) samples"))
    samples = sample(1:N, floor(Int, Z.kmin); replace=false)
    Z = update!(rng, X[samples], alg, X; kernel=kernel)
    return Z
end

function update!(
    rng::AbstractRNG, Z::AbstractVector, alg::OIPS, X::AbstractVector; kernel::Kernel
)
    return add_point!(rng, Z, alg, X, kernel)
end

function add_point!(
    rng::AbstractRNG, Z::AbstractVector, alg::OIPS, X::AbstractVector, kernel::Kernel
)
    b = length(X)
    for i in 1:b # Parse all points from X
        kx = kernelmatrix(kernel, [X[i]], Z)
        # d = find_nearest_center(X[i,:],Z.centers,kernel)[2]
        if maximum(kx) < alg.ρ_accept # If the biggest correlation is smaller than threshold add point
            push!(Z, X[i])
        end
        while length(Z) > alg.kmax ## If maximum number of points is reached, readapt the threshold
            K = kernelmatrix(kernel, Z)
            m = maximum(K - Diagonal(K))
            alg.ρ_remove = alg.η * m
            remove_point!(rng, Z, alg, K)
            if alg.ρ_remove < alg.ρ_accept # Readapt the thresholds
                alg.ρ_accept = alg.η * alg.ρ_remove
            end
            @info "ρ_accept reset to : $(alg.ρ_accept)"
        end
    end
    return Z
end

function remove_point!(rng::AbstractRNG, Z::AbstractVector, alg::OIPS, K::AbstractMatrix)
    if length(Z) > alg.kmin # Only remove points if the minimum is not reached
        overlapcount = (x -> count(x .> alg.ρ_remove)).(eachrow(K))
        removable = SortedSet(findall(x -> x > 1, overlapcount))
        toremove = []
        c = 0
        while !isempty(removable) && length(Z) > alg.kmin
            i = sample(rng, collect(removable), Weights(overlapcount[collect(removable)]))
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
        end
        deleteat!(Z, toremove)
    end
    return Z
end
