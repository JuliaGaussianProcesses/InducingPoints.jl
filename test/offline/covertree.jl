@testset "CoverTree" begin
    seed!(42)
    N = 20
    D = 3
    ϵ = 0.1
    X = rand(D, N)
    x = ColVecs(X)
    y = [rand(D) for _ in 1:N]
    alg = CoverTree()
    @test alg.metric isa Euclidean
    test_Zalg(alg)
    for metric in [Euclidean(), SqEuclidean()], ϵ in [0.1, 0.3], lloyds in [true, false], voronoi in [true, false]
        alg = CoverTree(ϵ, lloyds, voronoi, metric)
        @test alg.metric == metric
        Z = inducingpoints(alg, x)
        @test length(first(Z)) == D
        @test repr(alg) == "Cover Tree Selection of Inducing Points (ϵ : $(ϵ), metric: $(metric))"
        Z = inducingpoints(alg, X; obsdim=2)
        @test length(first(Z)) == D
    end
end