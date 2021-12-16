@testset "kDPP" begin
    seed!(42)
    N = 20
    D = 3
    nInd = 10
    kernel = SqExponentialKernel()
    X = ColVecs(rand(D, N))
    alg = kDPP(nInd, kernel)
    @test repr(alg) == "k-DPP selection of inducing points"
    Z = inducingpoints(alg, X)
    @test length(Z) == nInd
    @test_throws ArgumentError kDPP(-1, kernel)
    @test_throws ErrorException inducingpoints(kDPP(100, kernel), X)
    test_Zalg(kDPP(nInd, kernel), N)
end
