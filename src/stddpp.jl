mutable struct StdDPP{T,M<:AbstractMatrix{T},K<:Kernel} <: AIP{T,M}
    kernel::K
    k::Int64
    Z::M
    function StdDPP(kernel::K) where {K<:Kernel}
        return new{Float64,Matrix{Float64},K}(kernel)
    end
end


function init!(alg::StdDPP{T},X,y,kernel) where {T}
    jitt = T(Jittering())
    K = Symmetric(kernelmatrix(alg.kernel,X,obsdim=1)+jitt*I)
    dpp = DPP(K)
    samp = rand(dpp,1)[1]
    alg.Z = X[samp,:]
    alg.k = length(samp)
end
