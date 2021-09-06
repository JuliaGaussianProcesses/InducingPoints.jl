@testset "StreamKmeans" begin
    seed!(42)
    N = 30
    D = 3
    nInd = 20
    kernel = SqExponentialKernel()
    X = ColVecs(rand(D, N) * 10)
    alg = StreamKmeans(nInd)
    @test repr(alg) == "Streaming Kmeans (m_target=$nInd)"
    test_Zalg(alg; kernel=kernel)
end
