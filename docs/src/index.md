```@meta
CurrentModule = InducingPoints
```

# InducingPoints

InducingPoints.jl aims at providing an easy way to select inducing points locations for Sparse Gaussian Processes both in an online and offline setting.

The point selection is splitted in the online (`OnIPSA`) and offline settings.

All algorithms inherit from `AbstractInducingPointsSelection` or `AIPSA` which can be passed to the different APIs

## Offline Inducing Points Selection

```julia
alg = KMeansAlg(10)
Z = inducingpoints(alg, X; kwargs...)
```

The Offline options are:
- [`KmeansAlg`](@ref) : use the k-means algorithm to select centroids minimizing the square distance with the dataset. The seeding is done via `k-means++`. Note that the inducing points are not going to be a subset of the data
- [`kDPP`](@ref) : sample from a k-Determinantal Point Process to select `k` points. `Z` will be a subset of `X`
- [`StdDPP`](@ref) : sample from a standard Determinantal Point Process. The number of inducing points is not fixed here. `Z` will be a subset of `X`
- [`RandomSubset`](@ref) : sample randomly `k` points from the data set uniformly.
- [`Greedy`](@ref) : Will select a subset of `X` which maximizes the `ELBO` (in a stochastic way)
## Online Inducing Points Selection

Online selection is a bit more involved.
```julia
alg = OIPS()
Z = inducingpoints(alg, x_1; kwargs...)
for x in eachbatch(X)
    updateZ!(Z, alg, x; kwargs...)
end
```

With `inducingpoints`, similarly to the offline setting, a first instance of `Z` is created.
`updateZ!` will then update the vectors in place.

The Online options are:
- [`OnlineIPSelection`](@ref) : A method based on distance between inducing points and data
- [`UniGrid`](@ref) : A regularly-spaced grid whom edges are adapted given the data
- [`SeqDPP`](@ref) : Sequential Determinantal Point Processes, subsets are regularly sampled from the new data batches conditionned on the existing inducing points.
- [`StreamKmeans`](@ref) : An online version of k-means.
- [`Webscale`](@ref) : Another online version of k-means

## Index 
```@index
```

## API
```@autodocs
Modules = [InducingPoints]
```
