using InducingPoints
using Test

@testset "InducingPoints.jl" begin
    include("seqdpp.jl")
    include("kdpp.jl")
    include("stddpp.jl")
    include("streamingkmeans.jl")
    include("webscale.jl")
    include("oips.jl")
    include("kmeans.jl")
    include("greedy.jl")
    include("uniform.jl")
    include("unigrid.jl")
end
