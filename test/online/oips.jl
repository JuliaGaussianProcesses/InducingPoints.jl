

@testset "OIPS" begin
    seed!(42)
    N = 30
    D = 3
    nInd = 20
    kernel = SqExponentialKernel()
    X = ColVecs(rand(D, N) * 10)
    ρ_accept = 0.8
    ρ_remove = 0.9
    alg = OIPS(ρ_accept, ρ_remove)
    @test repr(alg) ==
          "Online Inducing Point Selection (ρ_in : $(alg.ρ_accept), ρ_out : $(alg.ρ_remove), kmax : Inf)"
    Z = InducingPoints.init(alg, X; kernel=kernel)
    InducingPoints.update!(Z, alg, X; kernel=kernel)
    alg = OIPS(nInd)
    Z = InducingPoints.init(alg, X; kernel=kernel)
    @test length(alg) <= nInd
end
