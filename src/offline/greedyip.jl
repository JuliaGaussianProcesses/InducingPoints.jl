"""
    GreedyIP(m::Int, s::Int)

 - `m` is the desired number of inducing points
 - `s` is the minibatch size on which to select a new inducing point

Greedy approach first proposed by Titsias[1].
Algorithm loops over minibatches of data and select the best ELBO improvement.

[1] Titsias, M. Variational Learning of Inducing Variables in Sparse Gaussian Processes. Aistats 5, 567–574 (2009).
"""
struct Greedy <: OffIPSA
    m::Int # Number of inducing points
    s::Int # Minibatch size for selection
    function Greedy(m, s)
        m > 0 || throw(ArgumentError("Number of inducing points should be positive"))
        s > 0 || throw(ArgumentError("Size of the minibatch should be positive"))
        return new(m, s)
    end
end

function inducingpoints(
    rng::AbstractRNG,
    alg::Greedy,
    X::AbstractVector;
    y,
    kernel::Kernel,
    noise::Real,
    kwargs...,
)
    noise > 0 || throw(ArgumentError("Noise should be positive"))
    length(X) == length(y) || throw(ArgumentError("y and X have different lengths"))
    if alg.m >= length(X)
        return edge_case(alg.m, length(X), X)
    end
    return greedy_ip(rng, X, y, kernel, alg.m, alg.s, noise)
end

Base.show(io::IO, ::Greedy) = print(io, "Greedy Selection of Inducing Points")

function greedy_ip(
    rng::AbstractRNG,
    X::AbstractVector,
    y::AbstractVector,
    kernel::Kernel,
    m::Int,
    s::Int,
    noise::Real,
)
    N = length(X) # Number of samples
    i = rand(rng, 1:N) # Take a random initial point
    Z = collect.(X[i:i]) # Initialize empty array of IPs
    IP_set = Set{Int}(i) # Keep track of selected points
    f = AbstractGPs.GP(kernel) # GP object to compute the elbo
    for _ in 2:m
        # Evaluate on a subset of the points of a maximum size of 1000
        X_test = Set(sample(rng, 1:N, min(1000, N); replace=false))
        best_i = 0
        best_L = -Inf
        # Parse over random points of this subset
        new_candidates = collect(setdiff(X_test, IP_set))
        # Sample a minibatch of candidates
        d = sample(
            rng,
            collect(setdiff(X_test, IP_set)),
            min(s, length(new_candidates));
            replace=false,
        )
        for j in d # Loop over every sample and evaluate the elbo addition with each new sample
            new_Z = vcat(Z, X[j:j])
            L = AbstractGPs.elbo(f(X[collect(X_test)], noise), y[collect(X_test)], f(new_Z))
            if L > best_L
                best_i = j
                best_L = L
            end
        end
        push!(Z, X[best_i])
        push!(IP_set, best_i)
    end
    return Z
end

# function elbo(Z::AbstractVector, X::AbstractVector, y::AbstractVector, kernel::Kernel, σ²::Real)
#     Knm = kernelmatrix(kernel, X, Z)
#     Kmm = kernelmatrix(kernel, Z) + T(jitt) * I
#     Qff = Knm * (Kmm \ Knm')
#     Kt = kerneldiagmatrix(kernel, X) .+ T(jitt) - diag(Qff)
#     Σ = inv(Kmm) + Knm' * Knm / σ²
#     invQnn = 1/σ² * I - 1/ (σ²)^2 * Knm * inv(Σ) * Knm'
#     logdetQnn = logdet(Σ) + logdet(Kmm)
#     return -0.5 * dot(y, invQnn * y) - 0.5 * logdetQnn -
#            1.0 / (2 * σ²) * sum(Kt)
# end
