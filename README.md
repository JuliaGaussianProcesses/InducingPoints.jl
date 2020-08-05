# InducingPoints

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaGaussianProcesses.github.io/InducingPoints.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaGaussianProcesses.github.io/InducingPoints.jl/dev)
[![Build Status](https://travis-ci.com/JuliaGaussianProcesses/InducingPoints.jl.svg?branch=master)](https://travis-ci.com/JuliaGaussianProcesses/InducingPoints.jl)
[![Coverage](https://coveralls.io/repos/github/JuliaGaussianProcesses/InducingPoints.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaGaussianProcesses/InducingPoints.jl?branch=master)

A package for selecting inducing points for sparse GPs

This package provide a collection of inducing point location selection algorithms, both offline and online.

For example

```julia
using InducingPoints
Z = KmeansIP(X, 10, obsdim = 1)
```
will return 10 inducing points selected as clusters by the k-means algorithm

For online algorithms, one need to first create an empty object. For example

```julia
Z = OIPS(200) # Method from unpublished paper (for now!)
```

The first time data is met `init(Z, X, args...)` should be called (args depend of the algorithm). `init` will return a new object correctly parametrized.

In the subsequent calls `update!(Z, X, args...)` to add/remove/move points.
