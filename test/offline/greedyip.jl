@testset "Greedy" begin
    seed!(42)
    N = 20
    D = 3
    noise = 0.01
    nInd = 10
    kernel = SqExponentialKernel()
    X = ColVecs(rand(D, N))
    y = rand(N)
    alg = Greedy(nInd, N)
    @test repr(alg) == "Greedy Selection of Inducing Points"
    Z = inducingpoints(alg, X; y=y, kernel=kernel, noise=noise)
    @test length(Z) == nInd
    @test_throws ArgumentError Greedy(-1, 2)
    @test_throws ArgumentError Greedy(10, -3)
    @test_throws ArgumentError inducingpoints(
        alg, X; y=rand(N + 1), kernel=kernel, noise=noise
    )
    @test_throws ArgumentError inducingpoints(alg, X; y=y, kernel=kernel, noise=-1)
    @test_throws ErrorException inducingpoints(
        Greedy(N + 1, N), X; y=y, kernel=kernel, noise=noise
    )
    test_Zalg(Greedy(nInd); kernel=kernel, noise=noise, y=y)
end
