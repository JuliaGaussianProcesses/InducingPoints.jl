mutable struct Kmeans{T,M<:AbstractMatrix{T}} <: AIP{T,M}
  k::Int64
  nMarkov::Int64
  Z::M
  function Kmeans(nInducingPoints::Integer; nMarkov = 10)
    return new{Float64,Matrix{Float64}}(
      nInducingPoints,
      opt,
      nMarkov,
    )
  end
end

Base.show(io::IO, alg::Kmeans) =
  print(io, "k-Means Selection of Inducing Points (k : $(alg.k))")

function init!(alg::Kmeans, X, y, kernel; tol = 1e-3)
  @assert size(X, 1) >= alg.k "Input data not big enough given $(alg.k)"
  alg.Z = kmeans_ip(X, alg.k, nMarkov = alg.nMarkov, tol = tol)
end

"""Fast and efficient seeding for KMeans based on [`Fast and Provably Good Seeding for k-Means](https://las.inf.ethz.ch/files/bachem16fast.pdf)"""
function kmeans_seeding(
  X::AbstractArray{T,N},
  nC::Integer,
  nMarkov::Integer,
) where {T,N} #X is the data, nC the number of centers wanted, m the number of Markov iterations
  NSamples = size(X, 1)
  #Preprocessing, sample first random center
  init = sample(1:NSamples, 1)
  C = zeros(nC, size(X, 2))
  C[1, :] = X[init, :]
  q = zeros(NSamples)
  for i = 1:NSamples
    q[i] = 0.5 * norm(X[i, :] .- C[1])^2
  end
  sumq = sum(q)
  q = Weights(q / sumq .+ 1.0 / (2 * NSamples), 1)
  for i = 2:nC
    x = X[sample(1:NSamples, q, 1), :] #weighted sampling,
    mindist = mindistance(x, C, i - 1)
    for j = 2:nMarkov
      y = X[sample(q), :] #weighted sampling
      dist = mindistance(y, C, i - 1)
      if (dist / mindist > rand())
        x = y
        mindist = dist
      end
    end
    C[i, :] = x
  end
  return C
end

#Return K inducing points from X, m being the number of Markov iterations for the seeding
function kmeans_ip(
  X::AbstractArray{T,N},
  nC::Integer;
  nMarkov::Integer = 10,
  kweights::Vector{T} = [0.0],
  tol = 1e-3,
) where {T,N}
  C = copy(transpose(kmeans_seeding(X, nC, nMarkov)))
  if kweights != [0.0]
    kmeans!(copy(transpose(X)), C, weights = kweights, tol = tol)
  else
    kmeans!(copy(transpose(X)), C)
  end
  return copy(transpose(C))
end


#Compute the minimum distance
function mindistance(
  x::AbstractArray{T,N1},
  C::AbstractArray{T,N2},
  nC::Integer,
) where {T,N1,N2}#Point to look for, collection of centers, number of centers computed
  mindist = Inf
  for i = 1:nC
    mindist = min.(norm(x .- C[i])^2, mindist)
  end
  return mindist
end
