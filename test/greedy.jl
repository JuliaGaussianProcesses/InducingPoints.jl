seed!(42)
N = 20
D = 3
nInd = 10
k = SqExponentialKernel()
X = rand(N, D)
y = rand(N)

@testset "Greedy" begin
    alg = Greedy(nInd, N)
    @test repr(alg) == "Greedy Selection of Inducing Points"
    AGP.IPModule.init!(alg, X, y, k)
    @test_throws AssertionError AGP.IPModule.init!(Greedy(N+1, N), X, y, k)
    @test_throws AssertionError AGP.IPModule.init!(Greedy(nInd, N+1), X, y, k)
    @test size(alg) == (nInd, D)
end
