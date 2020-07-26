mutable struct KmeansIP{S,TZ<:AbstractVector{S}} <: AIP{T,TZ}
  k::Int64
  Z::TZ
end

function Kmeans(
  X::AbstractMatrix,
  m::Integer;
  obsdim = 2,
  nMarkov = 10,
  weights = nothing,
  tol::Real = 1e-3,
)
  @assert size(X, obsdim) >= alg.k "Input data not big enough given $(alg.k)"
  return KMeans(
    m,
    kmeans_ip(
      X,
      m,
      obsdim = obsdim,
      nMarkov = nMarkov,
      kweights = weights,
      tol = tol,
    ),
  )
end


Base.show(io::IO, alg::KmeansIP) =
  print(io, "k-Means Selection of Inducing Points (k : $(alg.k))")

function init!(alg::Kmeans, X, y, kernel; tol = 1e-3)
  @assert size(X, 1) >= alg.k "Input data not big enough given $(alg.k)"
  alg.Z = kmeans_ip(X, alg.k, nMarkov = alg.nMarkov, tol = tol)
end

#Return K inducing points from X, m being the number of Markov iterations for the seeding
function kmeans_ip(
  X::AbstractArray{T,N},
  nC::Integer;
  obsdim::Int = 2,
  nMarkov::Int = 10,
  weights = nothing,
  tol = 1e-3,
) where {T,N}
  if obsdim == 2
    C = kmeans_seeding(X', nC, nMarkov)
    if !isnothing(weights)
      kmeans!(X', C, weights = weights, tol = tol)
    else
      kmeans!(X', C, tot = tol)
    end
    return ColVecs(C)
  elseif obsdim == 1
    C = kmeans_seeding(X, nC, nMarkov)
    if !isnothing(weights)
      kmeans!(X, C, weights = weights, tol = tol)
    else
      kmeans!(X, C, tol = tol)
    end
    return ColVecs(C)
  end
end

"""Fast and efficient seeding for KMeans based on [`Fast and Provably Good Seeding for k-Means](https://las.inf.ethz.ch/files/bachem16fast.pdf)"""
function kmeans_seeding(
  X::AbstractMatrix{T},
  nC::Integer,
  nMarkov::Integer,
) where {T} #X is the data, nC the number of centers wanted, m the number of Markov iterations
  nDim, nSamples = size(X)
  #Preprocessing, sample first random center
  init = rand(1:nSamples, 1)
  C = zeros(T, nDim, nC)
  C[:, 1] .= X[:, init]
  q = vec(pairwise(SqEuclidean(), X, C[:, 1:1], dims = 2))
  sumq = sum(q)
  q = Weights(q / sumq .+ 1.0 / (2 * nSamples), 1)
  for i = 2:nC
    x = X[:, sample(q)] # weighted sampling,
    mindist = mindistance(x, C)
    for j = 2:nMarkov
      y = X[:, sample(q)] #weighted sampling
      dist = mindistance(y, C)
      if (dist / mindist > rand())
        x = y
        mindist = dist
      end
    end
    C[:, i] .= x
  end
  return C
end

#Compute the minimum distance between a vector and a collection of vectors
function mindistance(
  x::AbstractVector,
  C::AbstractMatrix,
)#Point to look for, collection of centers, number of centers computed
  return minimum(pairwise(SqEuclidean(), C, permutedims(x), dims = 2))
end
