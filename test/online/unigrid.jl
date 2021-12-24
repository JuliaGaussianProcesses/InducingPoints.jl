@testset "UniGrid" begin
    seed!(42)
    N = 30
    D = 3
    nInd = 20
    X = ColVecs(rand(D, N) * 10)
    alg = UniGrid(nInd)
    @test repr(alg) == "Uniform grid with side length $nInd."
    test_Zalg(alg)

    ### 1-dim grid
    @testset "1-dim UniformGrid" begin
        proditer = Iterators.product(1.:5.)
        gridp = [only(collect(x)) for x in collect(proditer)]
        ug = UniformGrid(proditer)

        # getindex
        @test ug[2] == gridp[2]
        @test ug[2:4] == gridp[2:4]
        @test ug[:] == gridp[:]

        @test IndexStyle(ug) isa IndexLinear

        # broadcast
        @test norm.(ug) == norm.(gridp)
    end
    
    ### >1-dim grid
    @testset ">1-dim grid" begin
        proditer = Iterators.product(1.:5., 6.:10.)
        gridp = collect.(collect(proditer))[:]
        ug = UniformGrid(proditer)

        # getindex
        @test ug[2] == gridp[2]
        @test ug[2:4] == gridp[2:4]
        @test ug[:] == gridp[:]

        @test IndexStyle(ug) isa IndexLinear

        # broadcast
        @test norm.(ug) == norm.(gridp)
    end

end
