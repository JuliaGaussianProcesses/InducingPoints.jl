"""
    CoverTree(ϵ::Real, lloyds::Bool=true, voronoi::Bool=true, metric::SemiMetric=Euclidean())

The CoverTree algorithm [1], recursively builds a tree for which the node will optimally cover the given dataset according to the `metric` distance.

## Arguments:
- `ϵ::Real`: Spatial resolution. Higher `ϵ` will result in less points.
- `lloyds::Bool`: Use the centroid of the ball created around the sampled datapoint instead of the datapoint itself if no other inducing point is close
- `voronoi::Bool`: Reattributes samples to each node at the end of the proecssing of each layer
- `metric::SemiMetric`: Distance metric used to determine distance between points. 

[1] Alexander Terenin, David R. Burt, Artem Artemev, Seth Flaxman, Mark van der Wilk, Carl Edward Rasmussen, Hong Ge: Numerically Stable Sparse Gaussian Processes via Minimum Separation using Cover Trees: https://arxiv.org/abs/2210.07893
"""
struct CoverTree{Tϵ,Tm} <: OffIPSA
    ϵ::Tϵ
    lloyd::Bool
    voronoi::Bool
    metric::Tm
    function CoverTree(
        ϵ::Tϵ=5e-2, lloyd::Bool=true, voronoi::Bool=true, metric::Tm=Euclidean()
    ) where {Tϵ<:Real,Tm<:SemiMetric}
        ϵ > 0 || throw(ArgumentError("The number of inducing points m should be positive"))
        return new{Tϵ,Tm}(ϵ, lloyd, voronoi, metric)
    end
end

function Base.show(io::IO, alg::CoverTree)
    return print(
        io, "Cover Tree Selection of Inducing Points (ϵ : $(alg.ϵ), metric: $(alg.metric))"
    )
end

struct CoverTreeNode{Tx} <: AbstractNode{Tx}
    x::Tx # Mean of the collection
    R::Real # Max radius
    A::AbstractVector # Collection of sample indices associated with center x
    parent::Union{CoverTreeNode,Nothing}
    neighbours::Vector{CoverTreeNode}
    children::Vector{CoverTreeNode}

    function CoverTreeNode(
        x::T, R, A, parent=nothing, neighbours=CoverTreeNode[], children=CoverTreeNode[]
    ) where {T}
        return new{T}(x, R, A, parent, neighbours, children)
    end
end

AbstractTrees.children(n::CoverTreeNode) = n.children
neighbours(n::CoverTreeNode) = Iterators.flatten((n, n.neighbours))
Base.show(io::IO, n::CoverTreeNode) = print(io, "CoverTreeNode (x=$(n.x))")
Base.length(::CoverTreeNode) = 1
Base.iterate(n::CoverTreeNode) = n, nothing
Base.iterate(::CoverTreeNode, ::Any) = nothing

function inducingpoints(
    rng::AbstractRNG, alg::CoverTree, X::TX; kwargs...
) where {TX<:AbstractVector}
    x₀ = mean(X)
    dmax = maximum(Base.Fix1(alg.metric, x₀), X)
    L = ceil(Int, log2(dmax / alg.ϵ))
    R = 2^L * alg.ϵ
    root = CoverTreeNode(x₀, R, collect(eachindex(X)))
    parents = [root]
    for l in 2:(L + 1)
        R /= 2
        children = distribute(rng, alg, parents, R, X)
        assign_neighbours!(alg, parents, L, l, R, X)
        alg.voronoi && reassign_data!(alg, children, X)
        parents = children
    end
    return convert_back(TX, getproperty.(parents, :x))
end

function distribute(
    rng::AbstractRNG,
    alg::CoverTree,
    parents::AbstractVector{<:CoverTreeNode},
    R::Real,
    X::AbstractVector,
)
    mapreduce(vcat, parents) do p
        children = CoverTreeNode[]
        while !isempty(p.A)
            ζ = X[rand(rng, p.A)] # Chose an arbitrary point.
            ζ =
                if alg.lloyd && any(neighbours(p)) do neighbour
                    !isempty(neighbour.children) ||
                        any(x -> alg.metric(x.x, ζ) < R, neighbour.children)
                end # Check that min_z∈Z' ||ζ - z|| < R 
                    ζ
                else
                    mean(view(X, neighbours(alg, ζ, p.A, R, X))) # Find the mean of the points around the ball R
                end
            c = CoverTreeNode(ζ, R, [], p) # Create a child.
            # Assign the appropriate data to the newly found point.
            for r in neighbours(p) # Assign the points to the child instead.
                cover_indices = neighbour_indices(alg, c, r.A, R, X)
                append!(c.A, r.A[cover_indices])
                deleteat!(r.A, cover_indices)
            end
            push!(children, c)
        end
        append!(p.children, children)
        children
    end
end

function neighbour_indices(alg::CoverTree, z::CoverTreeNode, A, R::Real, X)
    findall(eachindex(A)) do i
        alg.metric(z.x, X[A[i]]) < R
    end
end

function neighbours(alg::CoverTree, ζ, Z::AbstractVector, R::Real, X::AbstractVector)
    filter(Z) do z
        alg.metric(ζ, X[z]) <= R
    end
end

function assign_neighbours!(
    alg::CoverTree, parents::AbstractVector{<:CoverTreeNode}, R::Real, L, l, X
)
    factor = 4 * (1 - (1 / 2)^(L - l))
    R = R * factor
    for p in parents
        potential_neighbours = mapreduce(children, vcat, neighbours(p))
        for child in children(p)
            neighbours = filter(potential_neighbours) do neighbour
                alg.metric(child.x, neighbour.x) <= R
            end
            append!(child.neighbours, neighbours)
        end
    end
end

function reassign_data!(
    alg::CoverTree, nodes::AbstractVector{<:CoverTreeNode}, X::AbstractVector
)
    for node in nodes
        new_A = Int[]
        for i in node.A
            x = X[i]
            if isempty(node.neighbours)
                push!(new_A, i)
                continue
            end
            min_node = argmin(node.neighbours) do neighbour
                alg.metric(neighbour.x, x)
            end
            if alg.metric(node.x, x) <= alg.metric(min_node.x, x)
                push!(new_A, i)
            else
                push!(min_node.A, i)
            end
        end
        append!(empty!(node.A), new_A)
    end
end
