function partial_randomly_pivoted_cholesky(k::Kernel, x::AbstractVector, M::Int)

    # Initialise working memory.
    d = kernelmatrix_diag(k, x)
    C_max = maximum(d)
    u = kernelmatrix(k, x, x[1:1])

    # Initialise output data structures.
    N = length(x)
    V = zeros(eltype(d), N, M)
    p = collect(1:N)

    # Compute partial pivoted Cholesky.
    for j in 1:M
        # d_max, j_max = findmax(view(d, j:N))
        j_max = rand(AbstractGPs.Categorical(d[j:N] ./ sum(d[j:N])))
        j_max += j - 1
        d_max = d[j_max]

        switch!(p, j, j_max)
        switch!(d, j, j_max)
        switch_rows!(V, j, j_max, 1:M)
        u .= kernelmatrix(k, x[p], x[p[j]:p[j]])

        V[j, j] = sqrt(d_max)

        for i in (j + 1):N
            a = u[i]
            b = view(V, i, 1:(j - 1))' * view(V, j, 1:(j - 1))
            V[i, j] = (a - b) / V[j, j]
            d[i] -= V[i, j]^2
        end
    end
    return (V, p, M)
end

struct RandomVarSelection <: OffIPSA
    M::Int
    function RandomVarSelection(M::Int)
        M > 0 || throw(ArgumentError("Number of inducing points should be positive"))
        return new(M)
    end
end

function inducingpoints(
    rng::AbstractRNG, alg::RandomVarSelection, x::AbstractVector; kernel::Kernel
)
    # Don't try to handle this case.
    alg.M > length(x) && throw(ArgumentError("M > length(x). Requires M <= length(x)."))

    # Perform the partial Cholesky, and return the elements of `x` residing in the first
    # M_used elements of the permutation vector returned.
    V, p, M_used = partial_randomly_pivoted_cholesky(kernel, x, alg.M)
    return x[p[1:M_used]]
end
