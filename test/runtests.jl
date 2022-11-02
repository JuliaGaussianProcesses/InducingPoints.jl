using InducingPoints
using Distances
using KernelFunctions
using KernelFunctions: ColVecs
using Test
using Random: seed!, MersenneTwister
include("test_utils.jl")

@testset "InducingPoints.jl" begin
    @testset "Offline" begin
        for file in readdir(joinpath(@__DIR__, "offline"))
            include(joinpath(@__DIR__, "offline", file))
        end
    end
    @testset "Online" begin
        for file in readdir(joinpath(@__DIR__, "online"))
            include(joinpath(@__DIR__, "online", file))
        end
    end
end
