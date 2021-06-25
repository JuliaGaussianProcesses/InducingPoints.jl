"""
    kDPP(m::Int, kernel::Kernel)

k-DPP (Determinantal Point Process) will return a subset of `X` of size `m`,
according to DPP probability
"""
struct kDPP{K<:Kernel} <: OffIPSA
    m::Int
    kernel::K
    function kDPP(m::Int, kernel::K) where {K<:Kernel}
		m > 0 || throw(ArgumentError("The number of inducing points m should be positive"))
        new{K}(m, kernel)
    end
end


function incudingpoints(rng::AbstractRNG, alg::kDPP, X::AbstractVector; kwargs...)
    return kddp_ip(rng, X, alg.m, alg.kernel)
    return kDPP(m, kernel, Z)
end

Base.show(io::IO, alg::kDPP) = print(io, "k-DPP selection of inducing points")

function kdpp_ip(rng::AbstractRNG, X::AbstractVector, m::Int, kernel::Kernel)
    nSamples = length(X)
    Z = Vector{eltype(X)}()
    i = rand(1:N)
    push!(Z, Vector(X[i]))
    IP_set = Set(i)
    k = 1
    kᵢᵢ = kerneldiagmatrix(kernel, X) .+ jitt
    while k < m
        X_set = setdiff(1:N, IP_set)
        kᵢZ = kernelmatrix(kernel, X[collect(X_set)], Z)
        KZ = kernelmatrix(kernel, Z) + jitt * I
        Vᵢ = kᵢᵢ[collect(X_set)] - diag(kᵢZ * inv(KZ) * kᵢZ')
        pᵢ = Vᵢ / sum(Vᵢ)
        j = sample(collect(X_set), Weights(pᵢ))
        push!(Z, Vector(X[j])); push!(IP_set, j)
        k += 1
    end
end
