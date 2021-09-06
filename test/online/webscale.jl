@testset "Webscale" begin
    seed!(42)
    nInd = 20
    kernel = SqExponentialKernel()
    alg = Webscale(nInd)
    @test repr(alg) == "Webscale (m = $nInd)"
    test_Zalg(alg; kernel=kernel)
end
