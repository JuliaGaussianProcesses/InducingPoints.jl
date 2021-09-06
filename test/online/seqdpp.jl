@testset "SeqDPP" begin
    seed!(42)
    N = 30
    D = 3
    nInd = 20
    kernel = SqExponentialKernel()
    X = ColVecs(rand(D, N) * 10)
    alg = SeqDPP()
    @test repr(alg) == "Sequential DPP"
    Z = initZ(alg, X; kernel=kernel)
    updateZ!(Z, alg, X; kernel=kernel)

    test_Zalg(alg; kernel=kernel)
end
