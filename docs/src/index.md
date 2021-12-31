```@meta
CurrentModule = InducingPoints
```

# Intro

InducingPoints.jl aims at providing an easy way to select inducing points locations for Sparse Gaussian Processes both in an online and offline setting. These most used most prominently in sparse GP regression (see e.g. [`ApproximateGPs.jl](https://github.com/JuliaGaussianProcesses/ApproximateGPs.jl))

All algorithms inherit from `AbstractInducingPointsSelection` or `AIPSA` which can be passed to the different APIs.

## Quickstart
InducingPoints.jl provides the following list of algorithms. For details on the specific usage see the [algorithms section](@ref available_algorithms).


### Offline Inducing Points Selection
These algorithms are designed to compute inducing points for a data set that is likely to remain unchanged. 
If the data set changes, the algorithms have to be rerun from scratch. 
```julia
alg = KMeansAlg(10)
Z = inducingpoints(alg, X; kwargs...)
```

The Offline options are:
- [`KmeansAlg`](@ref) : Use the k-means algorithm to select centroids minimizing the square distance with the dataset. The seeding is done via `k-means++`. Note that the inducing points are not going to be a subset of the data.
- [`kDPP`](@ref) : Sample from a k-Determinantal Point Process to select `k` points. `Z` will be a subset of `X`.
- [`StdDPP`](@ref) : Sample from a standard Determinantal Point Process. The number of inducing points is not fixed here. `Z` will be a subset of `X`.
- [`RandomSubset`](@ref) : Sample randomly `k` points from the data set uniformly.
- [`Greedy`](@ref) : Will select a subset of `X` which maximizes the `ELBO` (in a stochastic way).


### Online Inducing Points Selection

Online selection algorithms compute an initial set similarly to the offline methods via `inducingpoints`. For successive changes of the data sets, InducingPoints.jl allows for efficient updating via `updateZ!`.
```julia
alg = OIPS()
Z = inducingpoints(alg, x_1; kwargs...)
for x in eachbatch(X)
    updateZ!(Z, alg, x; kwargs...)
end
```

The Online options are:
- [`OnlineIPSelection`](@ref) : A method based on distance between inducing points and data
- [`UniGrid`](@ref) : A regularly-spaced grid whom edges are adapted given the data
- [`SeqDPP`](@ref) : Sequential Determinantal Point Processes, subsets are regularly sampled from the new data batches conditioned on the existing inducing points.
- [`StreamKmeans`](@ref) : An online version of k-means.
- [`Webscale`](@ref) : Another online version of k-means

## Index 
```@index
```

