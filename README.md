# InducingPoints

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaGaussianProcesses.github.io/InducingPoints.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaGaussianProcesses.github.io/InducingPoints.jl/dev)
![BuildStatus](https://github.com/JuliaGaussianProcesses/InducingPoints.jl/workflows/CI/badge.svg)
[![Coverage](https://coveralls.io/repos/github/JuliaGaussianProcesses/InducingPoints.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaGaussianProcesses/InducingPoints.jl?branch=master)

A package for selecting inducing points for sparse GPs

This package provide a collection of inducing point location selection algorithms, both offline and online.

## Offline algorithms


Offline algorithms are meant to be run once over the data before training begins.
Here is an example where we use the k-means algorithm
```julia
using InducingPoints
X = [rand(5) for _ in 1:100]
alg = KMeansAlg(10) # Create the kmeans algorithm
Z = inducingpoints(alg, X) # Returns a vector of vector of size 10 
```
will return 10 inducing points selected as clusters by the k-means algorithm

Note that it is possible to pass data as a matrix as well following the convention of [KernelFunctions.jl](https://juliagaussianprocesses.github.io/KernelFunctions.jl/dev/userguide/#Creating-a-Kernel-Matrix)
```julia
X = rand(5 , 1000)
alg = KMeansAlg(10, Euclidean()) # We can also use different metrics
Z = inducingpoints(alg, X) # This still returns a vector of vector of size 10 
```

## Online algorithms

Online algorithms needs two API, a first one to create the initial vector of inducing points and another one to update it with new data.
For example following [this work](https://drive.google.com/file/d/1IPTUBfY_b2WElTWBIVU4lrbHcXnbTWdB/view)

```julia
alg = OIPS(200) # We expect 200 inducing points
kernel = SqExponential()
X = [rand(5) for _ in 1:100] # We have some initial data
Z = init(alg, X; kernel=kernel) # We create an initial vector
X_new = [rand(5) for _ in 1:50] # We get some new data
update!(Z, alg, X_new; kernel=kernel) # Points will be acordingly added (or removed!)
```

Note that `Z` is directly changed in place.

## Notes

Make sure to check each algorithm docs independently, they will give you more details on what arguments they need and what they do!