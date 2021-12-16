"""
    SeqDPP()

Sequential sampling via Determinantal Point Processes
"""
struct SeqDPP <: OnIPSA end

Base.show(io::IO, ::SeqDPP) = print(io, "Sequential DPP")

function inducingpoints(
    rng::AbstractRNG,
    ::SeqDPP,
    X::AbstractVector;
    kernel::Kernel,
    arraytype=Vector{Float64},
    kwargs...,
)
    length(X) > 2 || throw(ArgumentError("First batch should contain at least 3 elements"))
    K = kernelmatrix(kernel, X) + jitt * I
    dpp = DPP(K)
    samp = []
    while length(samp) < 3 # Sample from a normal DPP until at least 3 elements are sampled
        samp = rand(rng, dpp)
    end
    Z = to_vec_of_vecs(X[samp], arraytype)
    return Z
end

function add_point!(
    rng::AbstractRNG,
    Z::AbstractVector{T},
    ::SeqDPP,
    X::AbstractVector;
    kernel::Kernel,
    kwargs...,
) where {T}
    L = kernelmatrix(kernel, vcat(Z, X)) + jitt * I
    Iₐ = diagm(vcat(zeros(length(Z)), ones(length(X))))
    Lₐ = Symmetric(
        inv(inv(L + Iₐ)[(length(Z) + 1):size(L, 1), (length(Z) + 1):size(L, 1)]) - I
    )
    new_dpp = DPP(Lₐ)
    new_samp = rand(rng, new_dpp)
    return push!(Z, X[new_samp]...)
end

function add_point(
    rng::AbstractRNG,
    Z::AbstractVector{T},
    ::SeqDPP,
    X::AbstractVector;
    kernel::Kernel,
    kwargs...,
) where {T}
    L = kernelmatrix(kernel, vcat(Z, X)) + jitt * I
    Iₐ = diagm(vcat(zeros(length(Z)), ones(length(X))))
    Lₐ = Symmetric(
        inv(inv(L + Iₐ)[(length(Z) + 1):size(L, 1), (length(Z) + 1):size(L, 1)]) - I
    )
    new_dpp = DPP(Lₐ)
    new_samp = rand(rng, new_dpp)
    return vcat(Z, T.(X[new_samp]))
end
