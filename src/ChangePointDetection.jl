module ChangePointDetection
using Random
using LinearAlgebra

const DEFAULT_SIGMA_LIST = [0.25, 0.5, 0.75, 1.0, 1.2, 1.5, 2.0, 2.5, 2.2, 3.0, 5.0]
const DEFAULT_LAMBDA_LIST = [1.00000000e-03, 3.16227766e-03, 1.00000000e-02, 3.16227766e-02,
                             1.00000000e-01, 3.16227766e-01, 1.00000000e+00, 3.16227766e+00,
                             1.00000000e+01]

"""
    squared_distance(X::AbstractVector, C::AbstractVector)

Computes the pairwise squared distance between elements of X and C.
Returns a matrix where `result[i,j] = (X[i] - C[j])^2`.
"""
function squared_distance(X::AbstractVector{T}, C::AbstractVector{T}) where {T<:AbstractFloat}
    return X.^2 .+ (C').^2 .- 2 .* X .* C'
end

# Fallback for mixed types: promote
function squared_distance(X::AbstractVector{<:AbstractFloat}, C::AbstractVector{<:AbstractFloat})
    T = promote_type(eltype(X), eltype(C))
    return squared_distance(T.(X), T.(C))
end

"""
    set_H!(H, dist, sigma)

Compute the RBF kernel matrix in-place: `H[i,j] = sqrt(sigma^2 * pi) * exp(-dist[i,j] / (4*sigma^2))`.
"""
function set_H!(H::AbstractMatrix{T}, dist::AbstractMatrix{T}, sigma::Real) where {T<:AbstractFloat}
    s2 = T(sigma)^2
    coeff = sqrt(s2 * T(pi))
    inv4s2 = T(-1) / (4 * s2)
    @. H = coeff * exp(dist * inv4s2)
    return nothing
end

"""
    set_h!(h_vecs, dists, sigma, count)

Compute kernel mean vectors in-place for each CV fold.
`h_vecs[i] .= (1/count) * sum(exp.(-dists[i] / (2*sigma^2)), dims=1)`
"""
function set_h!(h_vecs::Vector{Vector{T}}, dists::Vector{<:AbstractMatrix{T}}, sigma::Real, count::Integer) where {T<:AbstractFloat}
    inv2s2 = T(-1) / (2 * T(sigma)^2)
    inv_count = T(1) / T(count)
    for (i, dist) in enumerate(dists)
        # Compute column sums of exp.(-dist/(2*sigma^2)) and scale
        h = h_vecs[i]
        fill!(h, zero(T))
        @inbounds for col in 1:size(dist, 2)
            s = zero(T)
            for row in 1:size(dist, 1)
                s += exp(dist[row, col] * inv2s2)
            end
            h[col] = s * inv_count
        end
    end
    return nothing
end

"""
    lsdd(x, y; folds=5, sigma_list=nothing, lambda_list=nothing, rng=Random.default_rng())

Computes the least-squares density-difference (LSDD) between arrays `x` and `y`.
The LSDD value characterizes how different the probability densities that generated `x` and `y` are.
The closer the LSDD is to 0, the more similar the probability densities are.

# Arguments
- `x`, `y`: arrays of data upon which to perform the LSDD computation.
- `folds`: number of cross-validation folds. Higher is more precise but more expensive.
- `sigma_list`, `lambda_list`: grid points for kernel bandwidth and regularization optimization.
- `rng`: random number generator (for thread safety).

# Returns
- `L2`: the LSDD value.
"""
function lsdd(x::AbstractVector{T}, y::AbstractVector{T};
              folds::Integer = 5,
              sigma_list::Union{Nothing, AbstractVector{<:Real}} = nothing,
              lambda_list::Union{Nothing, AbstractVector{<:Real}} = nothing,
              rng::AbstractRNG = Random.default_rng()) where {T<:AbstractFloat}

    lx, ly = length(x), length(y)
    b = min(lx + ly, 300)

    # Select b random centers from the combined data without vcat allocation
    perm = randperm(rng, lx + ly)
    C = Vector{T}(undef, b)
    @inbounds for ci in 1:b
        idx = perm[ci]
        C[ci] = idx <= lx ? x[idx] : y[idx - lx]
    end

    CC_dist2 = squared_distance(C, C)
    xC_dist2 = squared_distance(collect(T, x), C)
    yC_dist2 = squared_distance(collect(T, y), C)

    Tx = lx - div(lx, folds)
    Ty = ly - div(ly, folds)

    # Cross-validation fold indices
    cv_split1 = floor.(Int, collect(1:lx) .* folds ./ lx)
    cv_split2 = floor.(Int, collect(1:ly) .* folds ./ ly)
    cv_index1 = shuffle(rng, cv_split1)
    cv_index2 = shuffle(rng, cv_split2)

    tr_idx1 = [findall(!=(i), cv_index1) for i in 1:folds]
    tr_idx2 = [findall(!=(i), cv_index2) for i in 1:folds]
    te_idx1 = [findall(==(i), cv_index1) for i in 1:folds]
    te_idx2 = [findall(==(i), cv_index2) for i in 1:folds]

    xTr_dist = [xC_dist2[idx, :] for idx in tr_idx1]
    yTr_dist = [yC_dist2[idx, :] for idx in tr_idx2]
    xTe_dist = [xC_dist2[idx, :] for idx in te_idx1]
    yTe_dist = [yC_dist2[idx, :] for idx in te_idx2]

    # Sigma and lambda lists
    sigmas = sigma_list === nothing ? T.(DEFAULT_SIGMA_LIST) : T.(sigma_list)
    lambdas = lambda_list === nothing ? T.(DEFAULT_LAMBDA_LIST) : T.(lambda_list)
    n_sigma = length(sigmas)
    n_lambda = length(lambdas)

    score_cv = zeros(T, n_sigma, n_lambda)
    H = Matrix{T}(undef, b, b)

    # Pre-allocate h vectors as flat Vector{T} for each fold
    hx_tr = [Vector{T}(undef, b) for _ in 1:folds]
    hy_tr = [Vector{T}(undef, b) for _ in 1:folds]
    hx_te = [Vector{T}(undef, b) for _ in 1:folds]
    hy_te = [Vector{T}(undef, b) for _ in 1:folds]

    # Pre-allocate workspace for the CV inner loop
    h_tr_buf = Vector{T}(undef, b)
    h_te_buf = Vector{T}(undef, b)
    theta_buf = Vector{T}(undef, b)
    Htheta_buf = Vector{T}(undef, b)
    alpha_buf = Vector{T}(undef, b)
    scaled_buf = Vector{T}(undef, b)

    for (sigma_idx, sigma) in enumerate(sigmas)
        set_H!(H, CC_dist2, sigma)
        set_h!(hx_tr, xTr_dist, sigma, Tx)
        set_h!(hy_tr, yTr_dist, sigma, Ty)
        set_h!(hx_te, xTe_dist, sigma, lx - Tx)
        set_h!(hy_te, yTe_dist, sigma, ly - Ty)

        # Eigendecompose H once per sigma to avoid repeated Cholesky solves
        F = eigen(Symmetric(H))
        eigvals = F.values
        V = F.vectors
        Vt = V'  # pre-transpose for reuse

        for i in 1:folds
            @. h_tr_buf = hx_tr[i] - hy_tr[i]
            @. h_te_buf = hx_te[i] - hy_te[i]

            # Project h_tr into eigenbasis: alpha = V' * h_tr
            mul!(alpha_buf, Vt, h_tr_buf)

            for (lambda_idx, lam) in enumerate(lambdas)
                # theta = V * diag(1/(eigenvalues + lambda)) * V' * h_tr
                @inbounds for k in 1:b
                    scaled_buf[k] = alpha_buf[k] / (eigvals[k] + lam)
                end
                mul!(theta_buf, V, scaled_buf)

                # H*theta = V * diag(eigenvalues / (eigenvalues + lambda)) * V' * h_tr
                @inbounds for k in 1:b
                    scaled_buf[k] = eigvals[k] * alpha_buf[k] / (eigvals[k] + lam)
                end
                mul!(Htheta_buf, V, scaled_buf)

                score_cv[sigma_idx, lambda_idx] += dot(theta_buf, Htheta_buf) - 2 * dot(theta_buf, h_te_buf)
            end
        end
    end

    # Retrieve optimal parameters (Bug fix: use [1] for sigma row, [2] for lambda column)
    best_idx = findmin(score_cv)[2]
    sigma_chosen = sigmas[best_idx[1]]
    lambda_chosen = lambdas[best_idx[2]]

    # Final computation with optimal parameters
    set_H!(H, CC_dist2, sigma_chosen)
    # Add regularization in-place
    @inbounds for k in 1:b
        H[k, k] += lambda_chosen
    end

    inv2s2 = T(-1) / (2 * sigma_chosen^2)
    inv_lx = T(1) / T(lx)
    inv_ly = T(1) / T(ly)

    # Compute h = (1/lx)*sum(K(x,C)) - (1/ly)*sum(K(y,C))
    h_final = Vector{T}(undef, b)
    @inbounds for j in 1:b
        sx = zero(T)
        for i in 1:lx
            sx += exp(xC_dist2[i, j] * inv2s2)
        end
        sy = zero(T)
        for i in 1:ly
            sy += exp(yC_dist2[i, j] * inv2s2)
        end
        h_final[j] = sx * inv_lx - sy * inv_ly
    end

    # Solve (H + lambda*I) * theta = h  (H already has lambda*I added)
    theta_final = H \ h_final

    # Recompute H without regularization for the L2 score
    set_H!(H, CC_dist2, sigma_chosen)
    mul!(Htheta_buf, H, theta_final)
    L2 = 2 * dot(theta_final, h_final) - dot(theta_final, Htheta_buf)

    return L2
end

# Convenience: promote mixed float types
function lsdd(x::AbstractVector{<:AbstractFloat}, y::AbstractVector{<:AbstractFloat}; kwargs...)
    T = promote_type(eltype(x), eltype(y))
    return lsdd(T.(x), T.(y); kwargs...)
end

"""
    lsdd_profile(ts; window=150)

Returns for each point of the given time-series `ts` the LSDD value.
Allows estimation of changes in the underlying probability density:
a peak in the LSDD value indicates a change point.

Uses multithreading when Julia is started with multiple threads.
"""
function lsdd_profile(ts::AbstractVector{T}; window::Integer = 150) where {T<:AbstractFloat}
    n = length(ts)
    niter = n - 2 * window
    niter <= 0 && return T[]

    result = Vector{T}(undef, niter)

    # Pin BLAS to 1 thread to avoid oversubscription when using Julia threads
    old_blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        Threads.@threads for i in 1:niter
            pd1 = @view ts[(i+1):(i+window)]
            pd2 = @view ts[(i+window+1):(i+2*window)]
            # Use a per-iteration RNG for thread safety and reproducibility
            local_rng = Xoshiro(i)
            result[i] = lsdd(pd1, pd2; rng=local_rng)
        end
    finally
        BLAS.set_num_threads(old_blas_threads)
    end

    return result
end

# Fallback for non-AbstractFloat (e.g., Int arrays): convert to Float64
function lsdd_profile(ts::AbstractVector; window::Integer = 150)
    return lsdd_profile(Float64.(ts); window = window)
end

"""
    changepoints(ts; threshold=0.5, window=150)

Estimates change points in the underlying probability density of a time series via LSDD.
Every time the LSDD value exceeds the given threshold, a change point is detected.
Returns the list of detected change point indices.
"""
function changepoints(ts; threshold = 0.5, window = 150)
    profile = lsdd_profile(ts; window = window)
    return getpoints(profile; threshold = threshold)
end

"""
    getpoints(profile; threshold=0.9)

Given an LSDD profile, returns indices where the profile exceeds the threshold.
Uses hysteresis to avoid detecting multiple points for one change.
"""
function getpoints(profile; threshold = 0.9)
    points = Int[]
    exceeded = false
    for (index, value) in enumerate(profile)
        if value > threshold && !exceeded
            push!(points, index)
            exceeded = true
        elseif value < threshold && exceeded
            exceeded = false
        end
    end
    return points
end

export squared_distance, lsdd, lsdd_profile, changepoints, getpoints
end
