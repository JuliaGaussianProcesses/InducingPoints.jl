"""
    SeqDPP()

Sequential sampling via DeterminantalPointProcesses
"""
struct SeqDPP <: OnIPSA end

Base.show(io::IO, Z::SeqDPP) = print(io, "Sequential DPP")

SeqDPP() = SeqDPP(0, Symmetric(Matrix{Float64}(I(0))), [])

function SeqDPP(X::AbstractVector, k::Kernel)
    return SeqDPP(m, K, Z)
end

function init(rng::AbstractRNG, ::SeqDPP, X::AbstractVector; kernel::Kernel)
    length(X) > 2 || throw(ArgumentError("First batch should contain at least 3 elements"))
    K = kernelmatrix(kernel, X) + jitt * I
    dpp = DPP(K)
    samp = []
    while length(samp) < 3 # Sample from a normal DPP until at least 3 elements are sampled
        samp = rand(rng, dpp)
    end
    Z = X[samp]
    return Z
end

function add_point!(
    rng::AbstractRNG, Z::AbstractVector, ::SeqDPP, X::AbstractVector; kernel::Kernel
)
    L = kernelmatrix(kernel, vcat(Z, X)) + jitt * I
    Iₐ = diagm(vcat(zeros(length(Z)), ones(length(X))))
    Lₐ = inv(inv(L + Iₐ)[(length(Z) + 1):size(L, 1), (length(Z) + 1):size(L, 1)]) - I
    new_dpp = DPP(Lₐ)
    new_samp = rand(rng, new_dpp)
    return push!(Z, X[new_samp]...)
end

# function add_point_old!(alg::SeqDPP, X, y, kernel)
#     alg.K = Symmetric(kernelmatrix(alg.Z, kernel) + 1e-7I)
#     for i = 1:size(X, 1)
#         k = kernelmatrix(reshape(X[i, :], 1, size(X, 2)), alg.Z, kernel)
#         kk = kerneldiagmatrix(reshape(X[i, :], 1, size(X, 2)), kernel)[1]
#         #using (A B; C D) = (A - C invD B, invD B; 0, I)*(I, 0; C, D)
#         # p = logdet(alg.K - k'*inv(kk)*k) + logdet(kk) - (logdet(alg.K - k'*inv(kk+1)*k)+logdet(kk+1))
#         p =
#             logdet(alg.K - k' * inv(kk) * k) + logdet(kk) -
#             (logdet(alg.K - k' * inv(kk + 1) * k) + logdet(kk + 1))
#         # if p > log(alg.lim)
#         if p > log(rand())
#             # println(exp(p))
#             alg.Z = vcat(alg.Z, X[i, :]')
#             alg.K = symcat(alg.K, vec(k), kk)
#             alg.k = size(alg.Z, 1)
#         end
#     end
# end
