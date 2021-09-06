function test_Zalg(alg::InducingPoints.OnIPSA, N::Int=30, D::Int=3; kwargs...)
    @testset "Test Vector{Real}" begin
        for T in [Float64, Float32]
            X = rand(T, N) * 10
            Z = initZ(alg, X; arraytype=Vector{T}, kwargs...)
            @test eltype(Z) == T
            X2 = rand(T, 2 * N) * 10
            Z = updateZ!(Z, alg, X2; kwargs...)
            @test eltype(Z) == T
            Z = updateZ(Z, alg, X2; kwargs...)
            @test eltype(Z) == T
        end
    end
    @testset "Test Vector{Vector}" begin
        for T in [Float64, Float32]
            X = [rand(T, D) * 10 for _ in 1:N]
            Z = initZ(alg, X; arraytype=Vector{T}, kwargs...)
            @test eltype(Z) == Vector{T}
            X2 = [rand(T, D) * 10 for _ in 1:N]
            Z = updateZ!(Z, alg, X2; kwargs...)
            @test eltype(Z) == Vector{T}
            Z = updateZ(Z, alg, X2; kwargs...)
            @test eltype(Z) == Vector{T}
        end
    end
    @testset "Test ColVecs" begin
        for T in [Float64, Float32]
            X = ColVecs(rand(T, D, N) * 10)
            Z = initZ(alg, X; arraytype=Vector{T}, kwargs...)
            @test eltype(Z) == Vector{T}
            X2 = ColVecs(rand(T, D, 2 * N) * 10)
            X3 = [rand(T, D) * 10 for _ in 1:N]
            Z = updateZ!(Z, alg, X2; kwargs...)
            @test eltype(Z) == Vector{T}
            Z = updateZ!(Z, alg, X3; kwargs...)
            @test eltype(Z) == Vector{T}
            Z = updateZ(Z, alg, X2; kwargs...)
            @test eltype(Z) == Vector{T}
        end
    end
end
