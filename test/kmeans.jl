seed!(42)
N = 20
D = 3
nInd = 10
k = transform(SqExponentialKernel(), 10.0)
X = rand(N, D)
y = rand(N)

@testset "k-Means" begin
    alg = Kmeans(nInd)
    @test repr(alg) == "k-Means Selection of Inducing Points (k : $(nInd))"
    AGP.IPModule.init!(alg, X, y, k)
    @test size(alg) == (nInd, D)
end
