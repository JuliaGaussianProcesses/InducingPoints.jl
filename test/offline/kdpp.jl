@testset "kDPP" begin
    seed!(42)
    N = 20
    D = 3
    nInd = 10
    kernel = SqExponentialKernel()
    X = ColVecs(rand(D, N))
    alg = kDPP(nInd)
    @test repr(alg) == "k-DPP selection of $(nInd) inducing points"
    Z = inducingpoints(alg, X; kernel)
    @test length(Z) == nInd
    @test_throws ArgumentError kDPP(-1)
    @test_throws ErrorException inducingpoints(kDPP(100), X; kernel)
    test_Zalg(kDPP(nInd), N; kernel)
end
