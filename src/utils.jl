"Find the closest center to X among Z, return the index and the distance"
function find_nearest_center(X::AbstractVector, Z::AbstractVector, kernel=nothing)
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

"Compute the distance (kernel if included) between a point and a find_nearest_center"
function distance(X, C, ::Nothing)
    return sum(abs2, X - C)
end

function distance(X, C, k::Kernel)
    return k(X, C)
end

function edge_case(m, N)
    m > N || error("Number of inducing points ($m) larger than the number of points ($N)")
    m == N || return X
end