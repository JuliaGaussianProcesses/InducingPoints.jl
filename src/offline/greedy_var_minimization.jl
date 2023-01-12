"""
    partial_pivoted_cholesky(C::AbstractMatrix{<:Real}, M::Int, tol::Real)

Computes the partial pivoted Cholesky factorisation of `C`. Runs for at most `M` steps, but
will stop if the threshold provided by `tol` is met.

This implementation is directly translated from Algorithm 1 in [1].

[1] - L. Foster, A. Waagen, N. Aijaz, M. Hurley, A. Luis, J. Rinsky, C. Satyavolu, M. J.
    Way, P. Gazis, and A. Srivastava. Stable and Efficient Gaussian Process Calculations.
    Journal of Machine Learning Research, 2009.
"""
function partial_pivoted_cholesky(k::Kernel, x::AbstractVector, M::Int, tol::Real)

    # Initialise working memory.
    d = kernelmatrix_diag(k, x)
    C_max = maximum(d)
    u = kernelmatrix(k, x, x[1:1])

    # Initialise output data structures.
    N = length(x)
    V = Matrix{eltype(d)}(undef, N, M)
    V .= 0
    p = collect(1:N)

    for j in 1:M
        d_max, j_max = findmax(view(d, j:N))
        j_max += j - 1

        if d_max < tol * C_max
            return (V, p, j - 1)
        end

        if j_max ≢ j
            switch!(p, j, j_max)
            switch!(d, j, j_max)

            u .= kernelmatrix(k, x, x[j_max:j_max])
            switch_rows!(V, j, j_max)
        end

        V[j, j] = sqrt(d_max)

        for i in (j + 1):N
            V[i, j] = (u[i] - dot(view(V, i, 1:(j - 1)), view(V, j, 1:(j - 1)))) / V[j, j]
            d[i] -= V[i, j]^2
        end
    end
    return (V, p, M)
end

@inline function switch!(x::Array, i::Int, j::Int)
    tmp = x[i]
    x[i] = x[j]
    x[j] = tmp
end

@inline function switch_rows!(x::Array, i::Int, j::Int)
    tmp = x[i, :]
    x[i, :] .= x[j, :]
    x[j, :] .= tmp
end

"""
    GreedyVarMinimization(m::Int)

Greedy variance minimization approach to inducing point selection. Originally proposed by
[1], and revisited more recently by [2]. `m` is the desired number of inducing points.

[1] - L. Foster, A. Waagen, N. Aijaz, M. Hurley, A. Luis, J. Rinsky, C. Satyavolu, M. J.
    Way, P. Gazis, and A. Srivastava. Stable and Efficient Gaussian Process Calculations.
    Journal of Machine Learning Research, 2009.
[2] - D. R. Burt, C. E. Rasmussen, and M. van der Wilk. Convergence of Sparse Variational
    Inference in Gaussian Processes Regression. Journal of Machine Learning Research, 2020.
"""
struct GreedyVarMinimization{Ttol<:Real} <: OffIPSA
    M::Int
    tol::Ttol
    function GreedyVarMinimization(M::Int, tol::Ttol) where {Ttol<:Real}
        M > 0 || throw(ArgumentError("Number of inducing points should be positive"))
        tol >= 0 || throw(ArgumentError("tol should be non-negative"))
        return new{Ttol}(M, tol)
    end
end

"""
    inducingpoints(
        rng::AbstractRNG, alg::GreedyVarMinimization, x::AbstractVector; kernel::Kernel
    )

See `GreedyVarMinimization` for more info. `rng` isn't actually used here.
"""
function inducingpoints(
    rng::AbstractRNG, alg::GreedyVarMinimization, x::AbstractVector; kernel::Kernel
)
    # Don't try to handle this case.
    alg.M > length(x) && throw(ArgumentError("M > length(x). Requires M <= length(x)."))

    # Perform the partial Cholesky, and return the elements of `x` residing in the first
    # M_used elements of the permutation vector returned.
    V, p, M_used = partial_pivoted_cholesky(kernel, x, alg.M, alg.tol)
    return x[p[1:M_used]]
end
