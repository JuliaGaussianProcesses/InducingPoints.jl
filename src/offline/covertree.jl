"""
    CoverTree(m::Int, metric::SemiMetric=SqEuclidean(); nMarkov = 10, tol = 1e-3)


"""
struct CoverTree{Tϵ} <: OffIPSA
    ϵ::Tϵ
    function KmeansAlg(
        ϵ::Real, metric::SemiMetric=SqEuclidean(); nMarkov::Int=10, tol::T=1e-3
    ) where {T<:Real}
        ϵ > 0 || throw(ArgumentError("The number of inducing points m should be positive"))
        tol > 0 || throw(ArgumentError("The tolerance tol should be positive"))
        return new{typeof(metric),T}(m, metric, nMarkov, tol)
    end
end

struct CoverTreeNode
    x::AbstractVector # Mean of the collection
    R::Real # Max radius
    A::AbstractVector # Collection of samples associated with center
    parent::Union{CoverTreeNode,Nothing}
    neighbours::Vector{CoverTreeNode}
    children::Vector{CoverTreeNode} # children
    
    function CoverTreeNode(mean, R, A, parent=nothing, neighbours=CoverTreeNode[], children=CoverTreeNode)
        new(mean, R, A, parent, neighbours, children)
    end
end

function inducingpoints(
    rng::AbstractRNG, alg::CoverTree, X::AbstractVector; weights=nothing, kwargs...
)   
    x₀ = mean(X)
    metric = Euclidean()
    dmax = maximum(Base.Fix1(metric, x₀), X)
    L = ceil(Int, log2(dmax / alg.ϵ))
    R = 2^L * alg.ϵ
    root = CoverTreeNode(x₀, R, x)
    parents = [root]
    for l in 2:L+1
        children = distribute(rng, parents)
        assign_neighbours(children, L, l)
        parents = children
    end
end

function distribute(rng, parents)
    for p in parents
        A = copy(p.A)
        R = parent.R / 2
        children = CoverTreeNode[]
        while !isempty(A)
            i_ζ = rand(rng, 1:length(A)) # Chose an arbitrary point.
            ζ = A[i_ζ] 
            ζ′ = mean(neighbours(ζ, A, R))
            Z′ = 
            if minimum(Base.Fix1(Euclidean(), ζ), ) > R
                ζ = ζ′
            end
            z = CoverTreeNode(ζ, R, [ζ],) # Create a child.
            # Assign the appropriate data to the newly found point.
            for r in p.neighbours
                cover_indices = neighbours_indices(z, r.A, R)
                append!(z.A, r.A[cover_indices])
                deleteat!(r.A, cover_indices)
            end
            push!(children, CoverTreeNode)
        end
    end
end

function neighbour_indices(z::CoverTreeNode, A, R)
    findall(eachindex(A)) do i
        Euclidean()(A[i], z.x) < R
    end
end

function neighbours(ζ::AbstractVector, Z, R)
    filter(Z) do z
        Euclidean()(ζ, z) <= R
    end
end

function assign_neighbours(parents::AbstractVector{<:CoverTreeNode}, R, L, l)
    factor = 4 * (1 - (1 / 2)^(L - l))
    R = R * factor
    for p in parents
        potential_neighbours = vcat(neighbour.children for neighbour in z.parent.neighbours)
        for child in p.children
            neighbours = filter(potential_neighbours) do neighbour
                Euclidean()(child.x, neighbour) <= R
            end
            append!(z.neighbours, neighbours)
        end
    end
end
    self.levels[0].append(root)
    neighbor_factor = 4 * (1 - 1 / 2 ** np.arange(num_levels, -1, -1))

    for level in range(1, num_levels):
        radius = max_radius / (2**level)
        for parent in self.levels[level - 1]:
            while len(parent.data[0]) > 0:
                initial_point = parent.data[0][0]
                if lloyds:
                    initial_r_neighbor_x = parent.data[0]
                    initial_distances = self.distance((initial_point, initial_r_neighbor_x))
                    initial_neighborhood = initial_r_neighbor_x[initial_distances <= radius, :]
                    point = initial_neighborhood.mean(axis=-2)
                    for r_neighbor in parent.r_neighbors:
                        for child in r_neighbor.children:
                            if np.linalg.norm(point - child.point) < radius:
                                point = initial_point
                                break
                        else:
                            continue
                        break
                else:
                    point = initial_point
                neighborhood_x = np.empty((0, parent.data[0].shape[-1]))
                neighborhood_y = np.empty((0, parent.data[1].shape[-1]))
                for r_neighbor in parent.r_neighbors:
                    (r_neighbor_x, r_neighbor_y) = r_neighbor.data
                    distances = self.distance((point, r_neighbor_x))
                    indices = distances <= radius
                    neighborhood_x = np.concatenate(
                        (neighborhood_x, r_neighbor_x[indices, :]), axis=-2
                    )
                    neighborhood_y = np.concatenate(
                        (neighborhood_y, r_neighbor_y[indices, :]), axis=-2
                    )
                    r_neighbor.data = (r_neighbor_x[~indices, :], r_neighbor_y[~indices, :])
                child = CoverTreeNode(point, radius, parent, (neighborhood_x, neighborhood_y))
                self.levels[level].append(child)
                parent.children.append(child)
        for parent in self.levels[level - 1]:
            potential_child_r_neighbors = [
                child for r_neighbor in parent.r_neighbors for child in r_neighbor.children
            ]
            # children = [child.point for child in parent.children]
            # r_neighbors = [child.point for child in potential_child_r_neighbors]
            for child in parent.children:
                child.r_neighbors = [
                    r_neighbor
                    for r_neighbor in potential_child_r_neighbors
                    if self.distance((r_neighbor.point, child.point))
                    <= neighbor_factor[level] * radius
                ]
                if plotting:
                    child.plotting_data = (child.data[0].copy(), child.data[1].copy())
        if voronoi:
            for parent in self.levels[level - 1]:
                (voronoi_x, voronoi_y) = parent.voronoi_data
                if voronoi_x.size > 0:
                    potential_child_r_neighbors = [
                        child
                        for r_neighbor in parent.r_neighbors
                        for child in r_neighbor.children
                    ]
                    potential_points = np.stack(
                        [child.point for child in potential_child_r_neighbors]
                    )
                    potential_distances = self.distance(
                        (potential_points[:, None, ...], voronoi_x[None, :, ...])
                    )
                    nearest_potential_child = np.argmin(potential_distances, axis=0)
                    for (idx, child) in enumerate(potential_child_r_neighbors):
                        if not hasattr(child, "voronoi_data"):
                            child.voronoi_data = (
                                np.empty((0, parent.voronoi_data[0].shape[-1])),
                                np.empty((0, parent.voronoi_data[1].shape[-1])),
                            )
                        child_indices = nearest_potential_child == idx
                        node_neighborhood_x = voronoi_x[child_indices, :]
                        node_neighborhood_y = voronoi_y[child_indices, :]
                        neighborhood_x = np.concatenate(
                            (child.voronoi_data[0], node_neighborhood_x)
                        )
                        neighborhood_y = np.concatenate(
                            (child.voronoi_data[1], node_neighborhood_y)
                        )
                        voronoi_x = voronoi_x[~child_indices, :]
                        voronoi_y = voronoi_y[~child_indices, :]
                        nearest_potential_child = nearest_potential_child[~child_indices]
                        child.voronoi_data = (neighborhood_x, neighborhood_y)
                        child.data = (
                            child.voronoi_data[0].copy(),
                            child.voronoi_data[1].copy(),
                        )

    self.nodes = [node for level in self.levels for node in level]
end

function Base.show(io::IO, alg::KmeansAlg)
    return print(io, "k-Means Selection of Inducing Points (m : $(alg.m))")
end

#Return K inducing points from X, m being the number of Markov iterations for the seeding
function kmeans_ip(
    rng::AbstractRNG,
    X::AbstractVector{T},
    nC::Int,
    metric::SemiMetric;
    nMarkov::Int=10,
    weights=nothing,
    tol=1e-3,
) where {T}
    C = kmeans_seeding(rng, X, nC, metric, nMarkov)
    C = reduce(hcat, C)
    kmeans!(reduce(hcat, X), C; weights=weights, tol=tol, distance=metric)
    if T <: Real
        return vec(C)
    else
        return ColVecs(C)
    end
end

