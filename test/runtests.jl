using InducingPoints
using KernelFunctions
using KernelFunctions: ColVecs
using Test
using Random: seed!
@testset "InducingPoints.jl" begin
    @testset "Offline" begin
        for file in readdir("offline"; join=true)
            include(file)
        end
    end
    @testset "Online" begin
        for file in readdir("online"; join=true)
            include(file)
        end
    end
end
