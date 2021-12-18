```@setup base
using Random: seed!
seed!(42)
using KernelFunctions
using Plots
using InducingPoints
D = 2
N = 50
M = 10
x = [rand(D) for _ in 1:N]

function plot_inducing_points(x,Z)
    p = scatter(getindex.(x, 1), getindex.(x, 2), 
        label = "Original Data",
        color = :black, 
        markersize = 6,
        markerstrokewidth = 0,
        xlims = [0, 1.], ylims = [0., 1.])
    scatter!(p, getindex.(Z,1), getindex.(Z, 2), 
        marker = :star5, 
        markersize = 6, 
        color = :orangered3,
        label = "Inducing Points")
        return p
end
```

# Available Algorithms {.unlisted .unnumbered}

The algorithms available through InducingPoints.jl can be split into offline and online use. 
While all algorithms can be used to create one-off sets of inducing points, the online algorithms are designed in a way that allows for cheap updating. 

```@contents
    Pages = ["algorithms.md"]
    Depth = 3
```

## Offline Algorithms

### [`KmeansAlg`](@ref) 
Uses the k-means algorithm to select centroids minimizing the square distance with the dataset. The seeding is done via `k-means++`. Note that the inducing points are not going to be a subset of the data. 

```@example base
alg = KmeansAlg(M)
Z = inducingpoints(alg, x)
plot_inducing_points(x,Z) #hide
savefig("kmeans.svg"); nothing # hide
```
![](kmeans.svg)



### [`kDPP`](@ref)
Sample from a k-Determinantal Point Process to select `k` points. `Z` will be a subset of `X`. Requires a kernel from [KernelFunctions.jl](https://juliagaussianprocesses.github.io/KernelFunctions.jl/stable/kernels/)

```@example base
kernel = SqExponentialKernel()
alg = kDPP(M, kernel)
Z = inducingpoints(alg, x)
plot_inducing_points(x,Z) #hide
savefig("kdpp.svg"); nothing # hide
```
![](kdpp.svg)

### [`StdDPP`](@ref)
Samples from a standard Determinantal Point Process. The number of inducing points is not fixed here. `Z` will be a subset of `X`. Requires a kernel from [KernelFunctions.jl](https://juliagaussianprocesses.github.io/KernelFunctions.jl/stable/kernels/)

```@example base
kernel = SqExponentialKernel()
alg = StdDPP(kernel)
Z = inducingpoints(alg, x)
plot_inducing_points(x,Z) #hide
savefig("StdDPP.svg"); nothing # hide
```
![](StdDPP.svg)

### [`RandomSubset`](@ref)
Sample randomly `k` points from the data set uniformly.

```@example base
alg = RandomSubset(M)
Z = inducingpoints(alg, x)
plot_inducing_points(x,Z) #hide
savefig("RandomSubset.svg"); nothing # hide
```
![](RandomSubset.svg)


### [`Greedy`](@ref) 
This algorithm will select a subset of `X` which maximizes the `ELBO` (in a stochastic way)