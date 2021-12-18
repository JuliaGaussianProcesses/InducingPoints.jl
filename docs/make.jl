using InducingPoints
using Documenter

ENV["GKSwstype"] = "100"

makedocs(;
    modules=[InducingPoints],
    authors="Theo Galy-Fajou <theo.galyfajou@gmail.com> and contributors",
    repo="https://github.com/JuliaGaussianProcesses/InducingPoints.jl/blob/{commit}{path}#L{line}",
    sitename="InducingPoints.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JuliaGaussianProcesses.github.io/InducingPoints.jl",
        assets=String[],
    ),
    pages=["Home" => "index.md",
    "Algorithms" => "algorithms.md",
    "API" => "api.md"],
)

deploydocs(; repo="github.com/JuliaGaussianProcesses/InducingPoints.jl")
