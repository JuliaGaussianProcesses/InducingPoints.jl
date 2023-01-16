@testset "greedy_var_minimization" begin
    @testset "partial_pivoted_cholesky" begin
        @testset "N = $N, M = $M, tol = $tol" for (N, M) in [
                (5, 2), (5, 5), (10, 2), (10, 10), (50, 2), (50, 25), (50, 50)
            ],
            tol in [1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1e0]

            x = range(0, 1; length=N)
            V, p, M_used, d = InducingPoints.partial_pivoted_cholesky(SEKernel(), x, M, tol)
            V_M = V[:, 1:M_used]
            p_M = p[1:M_used]

            C = kernelmatrix(SEKernel(), range(0, 1; length=N))
            @test C[p_M, p_M][1:M_used, 1:M_used] ≈ (V_M * V_M')[1:M_used, 1:M_used]
            @test M_used <= M
            @test M_used == M || maximum(diag(C - V_M * V_M')) < maximum(diag(C)) * tol
            @test all(d .>= -10 * eps(Float64))
        end
    end

    @testset "GreedyVarSelection($M, $tol), $N" for (M, N) in [
            (2, 5), (5, 5), (2, 10), (5, 10), (10, 10)
        ],
        tol in [1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1e0]

        # Verify interface is satisfied.
        alg = GreedyVarSelection(M, tol)
        test_Zalg(alg, N, M; kernel=SEKernel())

        # Check that at most M pseudo-inputs are produced.
        x = randn(MersenneTwister(123456), N)
        z = inducingpoints(alg, x; kernel=SEKernel())
        @test length(z) <= M
    end

    # The purpose of this example of to try and guarantee that the algorithm is actually
    # O(NM^2), rather than accidentally being O(N^2) or anything like that. It should choose
    # a very small value for M here, and hence be very tractable.
    @testset "large example" begin
        M = 100
        N = 1_000_000
        tol = 1e-12
        x = randn(MersenneTwister(123456), N)
        alg = GreedyVarSelection(M, tol)
        z = inducingpoints(alg, x; kernel=SEKernel())
        @test length(z) < M
    end
end
