function test_Zalg(alg::InducingPoints.OnIPSA, N::Int=30, D::Int=3; kwargs...)
    @testset "Test Vector{Real}" begin
        for T in [Float64, Float32]
            X = rand(T, N) * 10
            Z = inducingpoints(alg, X; arraytype=Vector{T}, kwargs...)
            @test eltype(Z) == T
            @test inducingpoints(Xoshiro(0), alg, X) == inducingpoints(Xoshiro(0), alg, X)
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
            Z = inducingpoints(alg, X; arraytype=Vector{T}, kwargs...)
            @test eltype(Z) <: AbstractVector{T}
            @test inducingpoints(Xoshiro(0), alg, X; kwargs...) ==
                inducingpoints(Xoshiro(0), alg, X; kwargs...)
            X2 = [rand(T, D) * 10 for _ in 1:N]
            Z = updateZ!(Z, alg, X2; kwargs...)
            @test eltype(Z) <: AbstractVector{T}
            Z = updateZ(Z, alg, X2; kwargs...)
            @test eltype(Z) <: AbstractVector{T}
        end
    end
    @testset "Test ColVecs" begin
        for T in [Float64, Float32]
            X = ColVecs(rand(T, D, N) * 10)
            Z = inducingpoints(alg, X; arraytype=Vector{T}, kwargs...)
            @test eltype(Z) <: AbstractVector{T}
            @test inducingpoints(Xoshiro(0), alg, X; kwargs...) ==
                inducingpoints(Xoshiro(0), alg, X; kwargs...)
            X2 = ColVecs(rand(T, D, 2 * N) * 10)
            X3 = [rand(T, D) * 10 for _ in 1:N]
            Z = updateZ!(Z, alg, X2; kwargs...)
            @test eltype(Z) <: AbstractVector{T}
            Z = updateZ!(Z, alg, X3; kwargs...)
            @test eltype(Z) <: AbstractVector{T}
            Z = updateZ(Z, alg, X2; kwargs...)
            @test eltype(Z) <: AbstractVector{T}
        end
    end
end

function test_Zalg(alg::InducingPoints.OffIPSA, N::Int=30, D::Int=3; kwargs...)
    @testset "Test Vector{Real}" begin
        for T in [Float64, Float32]
            X = rand(T, N) * 10
            Z = inducingpoints(alg, X; kwargs...)
            @test Z isa AbstractVector
            @test eltype(Z) == T
            @test inducingpoints(Xoshiro(0), alg, X; kwargs...) ==
                inducingpoints(Xoshiro(0), alg, X; kwargs...)
        end
    end
    @testset "Test Vector{Vector}" begin
        for T in [Float64, Float32]
            X = [rand(T, D) * 10 for _ in 1:N]
            Z = inducingpoints(alg, X; kwargs...)
            @test Z isa AbstractVector{<:AbstractVector}
            @test first(Z) isa AbstractVector{<:T}
            @test inducingpoints(Xoshiro(0), alg, X; kwargs...) ==
                inducingpoints(Xoshiro(0), alg, X; kwargs...)
        end
    end
    @testset "Test ColVecs" begin
        for T in [Float64, Float32]
            X = ColVecs(rand(T, D, N) * 10)
            Z = inducingpoints(alg, X; kwargs...)
            @test Z isa ColVecs{T}
            @test inducingpoints(Xoshiro(0), alg, X; kwargs...) ==
                inducingpoints(Xoshiro(0), alg, X; kwargs...)
        end
    end
end
