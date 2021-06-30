@testset "k-Means" begin
    seed!(42)
    N = 20
    nDim = 3
    M = 10

    X = rand(nDim, N)
    x = ColVecs(X)
    alg = KmeansAlg(M)
    @test alg.metric == SqEuclidean()
    Z = inducingpoingpoints(alg, x)
    @test repr(Z) == "k-Means Selection of Inducing Points (k : $(M))"
    @test length(Z) == M
    Z = inducingpoints(alg, X; obsdim=1, weights=rand(N))
    @test length(Z) == M
    @test length(first(Z)) == nDim
end
