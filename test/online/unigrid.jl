@testset "UniGrid" begin
    seed!(42)
    N = 30
    D = 3
    nInd = 20
    X = ColVecs(rand(D, N) * 10)
    alg = UniGrid(nInd)
    @test repr(alg) == "Uniform grid with side length $nInd."
    test_Zalg(alg)

    function testUG(proditer, gridp, ug)
        # getindex
        @test ug[2] == gridp[2]
        @test ug[2:4] == gridp[2:4]
        @test ug[:] == gridp[:]

        @test IndexStyle(ug) isa IndexLinear

        # broadcast
        @test sum.(ug) == sum.(gridp)

        # size
        @test length(ug) == length(proditer)
        @test size(ug) == (length(proditer),)

        @test [(i, a) for (i, a) in enumerate(ug)] == [(i, a) for (i, a) in enumerate(gridp)]

        # kernel matrix
        ker = SqExponentialKernel()
        @test kernelmatrix(ker, ug) == kernelmatrix(ker, gridp)
    end

    ### 1-dim grid
    @testset "1-dim UniformGrid" begin
        proditer = Iterators.product(1.0:5.0)
        gridp = [only(collect(x)) for x in collect(proditer)]
        ug = UniformGrid(proditer)

        testUG(proditer, gridp, ug)

        @test repr("text/plain", ug) == "(5,) uniform grid with edges\n[1.0, 5.0]\n"
        @test repr(ug) == "(5,) uniform grid"
    end

    ### >1-dim grid
    @testset " >1-dim grid" begin
        proditer = Iterators.product(1.0:5.0, 6.0:10.0)
        gridp = collect.(collect(proditer))[:]
        ug = UniformGrid(proditer)

        testUG(proditer, gridp, ug)

        @test repr("text/plain", ug) ==
            "(5, 5) uniform grid with edges\n[1.0, 5.0]\n[6.0, 10.0]\n"
        @test repr(ug) == "(5, 5) uniform grid"
    end
end
