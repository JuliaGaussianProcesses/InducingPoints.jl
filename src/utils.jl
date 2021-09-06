"Find the closest center to X among Z, return the index and the distance"
function find_nearest_center(X::Union{Real,AbstractVector}, Z::AbstractVector, kernel=nothing)
    best = 1
    best_val = Inf
    for i in 1:length(Z)
        val = distance(X, Z[i], kernel)
        if val < best_val
            best_val = val
            best = i
        end
    end
    return best, best_val
end

#Compute the minimum distance between a vector and a collection of vectors
function mindistance(metric::SemiMetric, x::AbstractVector, C::AbstractVector) 
    #Point to look for, collection of centers, number of centers computed
    return minimum(evaluate(metric, c, x) for c in C)
end


# Compute the distance (kernel if included) between a point and a find_nearest_center"
function distance(X, C, ::Nothing)
    return sum(abs2, X - C)
end

function distance(X, C, k::Kernel)
    return k(X, C)
end

function edge_case(m, N, X)
    m > N && error("Number of inducing points ($m) larger than the number of points ($N)")
    return m == N && return X
end

function to_vec_of_vecs(x::AbstractVector{<:Real}, V)
    return V(x)
end

function to_vec_of_vecs(X::AbstractVector, V)
    return V.(X)
end