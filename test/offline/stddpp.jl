@testset "stddpp.jl" begin
    test_Zalg(StdDPP(SqExponentialKernel()); kernel=SqExponentialKernel())
end
