@testset "UniGrid" begin
    seed!(42)
    N = 30
    D = 3
    nInd = 20
    X = ColVecs(rand(D, N) * 10)
    alg = UniGrid(nInd)
    @test repr(alg) == "Uniform grid with side length $nInd."
    test_Zalg(alg)
end