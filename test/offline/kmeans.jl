@testset "k-Means" begin
    seed!(42)
    N = 20
    D = 3
    M = 10
    X = rand(D, N)
    x = ColVecs(X)
    alg = KmeansAlg(M)

    @test alg.metric == SqEuclidean()
    Z = inducingpoints(alg, x)
    @test repr(alg) == "k-Means Selection of Inducing Points (m : $(M))"
    @test length(Z) == M
    Z = inducingpoints(alg, X; obsdim=2, weights=rand(N))
    @test length(Z) == M
    @test length(first(Z)) == D

    test_Zalg(alg)
end
