```@setup base
using Random: seed!
seed!(42)
using KernelFunctions
using Plots
plotlyjs()
using InducingPoints
D = 2
N = 50
M = 10
x = [rand(D) .* [0.8, 1.0] for _ in 1:N]
N₂ = 25
x₂ = [rand(D) .* [0.2, 1.0] + [0.8, 0.0] for _ in 1:N₂]

function plot_inducing_points(x,Z, x₂ = nothing, Z₂=nothing)
    p = scatter(getindex.(x, 1), getindex.(x, 2), 
        label = "Original Data",
        color = :black, 
        markersize = 7,
        markerstrokewidth = 0,
        xlims = [0, 1.], ylims = [0., 1.])

    scatter!(p, getindex.(Z,1), getindex.(Z, 2), 
        marker = :xcross, 
        markersize = 4, 
        color = :orangered3,
        label = "Inducing Points Z")
        
    if !isnothing(Z₂)
        scatter!(p, getindex.(x₂,1), getindex.(x₂, 2), 
            markersize = 7, 
            color = :grey42,
            label = "Additional Data")
        scatter!(p, getindex.(Z₂,1), getindex.(Z₂, 2), 
            marker = :cross, 
            markersize = 4, 
            color = :seagreen,
            label = "Updated Z")
    end
    return p
end
```

# Available Algorithms


The algorithms available through InducingPoints.jl can be split into offline and online use. 
While all algorithms can be used to create one-off sets of inducing points, the online algorithms are designed in a way that allows for cheap updating. 

```@contents
    Pages = ["algorithms.md"]
    Depth = 3
```

We start with a set of `N` data points of dimension `D`, which we would like to reduce to only `M < N ` points.

```@example
D = 2
N = 50
M = 10
x = [rand(D) .* [0.8, 1.0] for _ in 1:N]
nothing # hide
```

## Offline Algorithms

### [`KmeansAlg`](@ref) 
Uses the k-means algorithm to select centroids minimizing the square distance with the dataset. The seeding is done via `k-means++`. Note that the inducing points are not going to be a subset of the data. 

```@example base
alg = KmeansAlg(M)
Z = inducingpoints(alg, x)
plot_inducing_points(x, Z) #hide
savefig("kmeans.svg"); nothing # hide
```
![](kmeans.svg)



### [`kDPP`](@ref)
Sample from a k-Determinantal Point Process to select `k` points. `Z` will be a subset of `X`. Requires a kernel from [KernelFunctions.jl](https://juliagaussianprocesses.github.io/KernelFunctions.jl/stable/kernels/)

```@example base
kernel = SqExponentialKernel()
alg = kDPP(M, kernel)
Z = inducingpoints(alg, x)
plot_inducing_points(x, Z) #hide
savefig("kdpp.svg"); nothing # hide
```
![](kdpp.svg)

### [`StdDPP`](@ref)
Samples from a standard Determinantal Point Process. The number of inducing points is not fixed here. `Z` will be a subset of `X`. Requires a kernel from [KernelFunctions.jl](https://juliagaussianprocesses.github.io/KernelFunctions.jl/stable/kernels/)

```@example base
kernel = SqExponentialKernel()
alg = StdDPP(kernel)
Z = inducingpoints(alg, x)
plot_inducing_points(x, Z) #hide
savefig("StdDPP.svg"); nothing # hide
```
![](StdDPP.svg)

### [`RandomSubset`](@ref)
Sample randomly `k` points from the data set uniformly.

```@example base
alg = RandomSubset(M)
Z = inducingpoints(alg, x)
plot_inducing_points(x, Z) #hide
savefig("RandomSubset.svg"); nothing # hide
```
![](RandomSubset.svg)


### [`Greedy`](@ref) 
This algorithm will select a subset of `X` which maximizes the `ELBO` (Evidence Lower BOund), which is done in a stochastic way via minibatches of size `s`. This also requires passing the output data, the kernel and the noise level as additional arguments to `inducingpoints`.

```@example base
y = rand(N)
s = 5
kernel = SqExponentialKernel()
noise = 0.1
alg = Greedy(M, s)
Z = inducingpoints(alg, x; y = y, kernel = kernel, noise = noise)
plot_inducing_points(x, Z) #hide
savefig("Greedy.svg"); nothing # hide
```
![](Greedy.svg)



## Online Algorithms

These algorithms are useful if we assume that we will have another set of data points that we would like to incorporate into an existing inducing point set. 

```@example
D = 2 # hide
N₂ = 25
x₂ = [rand(D) .* [0.2, 1.0] + [0.8, 0.0] for _ in 1:N₂]
nothing # hide
```
We can then update the inital set of inducing points `Z` via 
[`updateZ`](@ref) (or inplace via [`updateZ!`](@ref)).

### [`OnlineIPSelection`](@ref InducingPoints.OnlineIPSelection)
A method based on distance between inducing points and data. This algorithm has several parameters to tune the result. It also requires the kernel to be passed as a keyword argument to `inducingpoints` and `updateZ`.

```@example base
kernel = SqExponentialKernel() ∘ ScaleTransform(4.0)
alg = OIPS()
Z = inducingpoints(alg, x; kernel = kernel)
Z₂ = updateZ(Z, alg, x₂; kernel = kernel)
plot_inducing_points(x, Z, x₂, Z₂) #hide
savefig("OIPS.svg"); nothing # hide
```
![](OIPS.svg)

### [`UniGrid`](@ref) 
A regularly-spaced grid whose edges are adapted given the data.

```@example base
alg = UniGrid(5)
Z = inducingpoints(alg, x)
Z₂ = updateZ(Z, alg, x₂)
plot_inducing_points(x, Z, x₂, Z₂) #hide
savefig("UniGrid.svg"); nothing # hide
```
![](UniGrid.svg)


### [`SeqDPP`](@ref) 
Sequential Determinantal Point Processes, subsets are regularly sampled from the new data batches conditioned on the existing inducing points.

```@example base
kernel = SqExponentialKernel()
alg = SeqDPP()
Z = inducingpoints(alg, x; kernel = kernel)
Z₂ = updateZ(Z, alg, x₂; kernel = kernel)
plot_inducing_points(x, Z, x₂, Z₂) #hide
savefig("SeqDPP.svg"); nothing # hide
```
![](SeqDPP.svg)


### [`StreamKmeans`](@ref) 
An online version of k-means.

```@example base
alg = StreamKmeans(M)
Z = inducingpoints(alg, x)
Z₂ = updateZ(Z, alg, x₂)
plot_inducing_points(x, Z, x₂, Z₂) #hide
savefig("StreamKmeans.svg"); nothing # hide
```
![](StreamKmeans.svg)


### [`Webscale`](@ref) 
Another online version of k-means

```@example base
alg = Webscale(M)
Z = inducingpoints(alg, x)
Z₂ = updateZ(Z, alg, x₂)
plot_inducing_points(x, Z, x₂, Z₂) #hide
savefig("Webscale.svg"); nothing # hide
```
![](Webscale.svg)