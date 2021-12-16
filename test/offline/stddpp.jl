@testset "stddpp.jl" begin
    @testset "randomsubset.jl" begin
    test_Zalg(StdDPP(SqExponentialKernel()))
end
