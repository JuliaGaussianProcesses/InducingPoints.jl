```@meta
CurrentModule = InducingPoints
```

# InducingPoints

InducingPoints.jl aims at providing an easy way to select inducing points locations for Sparse Gaussian Processes both in an online and offline setting.

The point selection is splitted in the online and offline settings.

All methods inherit from `AbstractInducingPoints` which acts as a `Vector` of `Vector`s, making it especially compatible with `KernelFunctions.jl`

## Offline Inducing Points Selection

Given a set of features `X` you can get a point selection by calling

```julia
    Z = KmeansIP(X, 10, obsdim=1)
```

The Offline options are:
- [`KmeansIP`](@ref) : use the k-means algorithm to select centroids minimizing the square distance with the dataset.
The seeding is done via `k-means++`.
Note that the inducing points are not going to be a subset of the data
- [`kDPP`](@ref) : sample from a k-Determinantal Point Process to select `k` points. `Z` will be a subset of `X`
- [`StdDPP`](@ref) : sample from a standard Determinantal Point Process. The number of inducing points is not fixed here. `Z` will be a subset of `X`
- [`Uniform`](@ref) : sample randomly `k` points from the data set uniformly.
- [`Greedy`](@ref) : Will select a subset of `X` which maximizes the `ELBO` (in a stochastic way)
## Online Inducing Points Selection

Online selection is a bit more involved.
```julia
Z = OIPS()
Z = init(x_1, args)
for x in eachbatch(X)
    update!(Z, x)
end
```

After the first instance is created `init` will return a new instance when seeing the first batch of data with the right parametrization.
After one can simply call `update!` to update the vectors in place.

The Online options are:
- [OIPS](@ref) : A method based on distance between inducing points and data
- [UniGrid](@ref) : A regularly-spaced grid whom edges are adapted given the data
- [SeqDPP](@ref) : Sequential Determinantal Point Processes, subsets are regularly sampled from the new data batches conditionned on the existing inducing points.
- [StreamOnline](@ref) : An online version of k-means.

## Index 
```@index
```

## API
```@autodocs
Modules = [InducingPoints]
```
