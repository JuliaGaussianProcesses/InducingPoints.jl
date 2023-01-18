@testset "stddpp.jl" begin
    test_Zalg(StdDPP(); kernel=SqExponentialKernel())
end
