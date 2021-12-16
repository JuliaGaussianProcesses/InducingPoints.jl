@testset "randomsubset.jl" begin
    N = 30
    test_Zalg(RandomSubset(10))
    test_Zalg(RandomSubset(10), N; weights=rand(N))
end
