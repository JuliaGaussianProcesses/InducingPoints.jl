using InducingPoints
using Distances
using KernelFunctions
using KernelFunctions: ColVecs
using Test
using Random: seed!
@testset "InducingPoints.jl" begin
    @testset "Offline" begin
        for file in readdir("offline")
            include(joinpath("offline", file))
        end
    end
    @testset "Online" begin
        for file in readdir("online")
            include(joinpath("online", file))
        end
    end
end
