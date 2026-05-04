# Parallel merge of variance / covariance sufficient statistics (Chan; Pebay).
# Narrative and references: docs/MOMENTS_NUMERICS_PIPELINE.md.

"""
    merge_variance_chan(n1, μ1, M2_1, n2, μ2, M2_2) -> (n, μ, M2)

Merge **two disjoint groups** of scalar samples already reduced separately.

For group ``k \\in \\{1,2\\}``: `n_k` is the count, `μ_k` the mean, and `M2_k` the **sum of squared
deviations** about that group’s mean,
``M2_k = \\sum (x_i - \\mu_k)^2`` (not divided by `n_k`; not Bessel-corrected).

Returns pooled `(n, μ, M2)` for the union. This is the Chan et al. parallel variance merge; merging
`M2` avoids the bias you get from naïvely averaging child second moments in floating point.
"""
function merge_variance_chan(
    n1::Integer,
    μ1::FT,
    M2_1::FT,
    n2::Integer,
    μ2::FT,
    M2_2::FT,
) where {FT <: AbstractFloat}
    n = n1 + n2
    δ = μ2 - μ1
    μ = (n1 * μ1 + n2 * μ2) / n
    M2 = M2_1 + M2_2 + δ^2 * (n1 * n2 / n)
    return n, μ, M2
end

"""
    merge_covariance_pebay(n1, μx1, μy1, C1, n2, μx2, μy2, C2) -> (n, μx, μy, C)

Merge **two disjoint groups** of paired samples `(x, y)`, each summarized on its own.

`C_k` is the **sum of cross-deviations** about the group means:
``C_k = \\sum (x_i - \\mu_{x,k})(y_i - \\mu_{y,k})``. Returns pooled statistics for the union (Pebay
et al. style merge).
"""
function merge_covariance_pebay(
    n1::Integer,
    μx1::FT,
    μy1::FT,
    C1::FT,
    n2::Integer,
    μx2::FT,
    μy2::FT,
    C2::FT,
) where {FT <: AbstractFloat}
    n = n1 + n2
    δx = μx2 - μx1
    δy = μy2 - μy1
    μx = (n1 * μx1 + n2 * μx2) / n
    μy = (n1 * μy1 + n2 * μy2) / n
    C = C1 + C2 + (n1 * n2 / n) * δx * δy
    return n, μx, μy, C
end

"""
    merge_variance_children(n_child, μs, M2s) -> (n, μ, M2)

Fold **several** same-sized subgroups (each with count `n_child`, means `μs`, sums of squared
deviations `M2s`) into one pooled summary using [`merge_variance_chan`](@ref).
"""
function merge_variance_children(
    n_child::Integer,
    μs::AbstractVector{FT},
    M2s::AbstractVector{FT},
) where {FT <: AbstractFloat}
    k = length(μs)
    length(M2s) == k || throw(DimensionMismatch("μs and M2s must have the same length (got $k and $(length(M2s)))"))
    k >= 1 || throw(ArgumentError("need at least one subgroup to merge"))
    n_acc = n_child
    μ_acc = μs[1]
    M2_acc = M2s[1]
    for j in 2:k
        n_acc, μ_acc, M2_acc = merge_variance_chan(n_acc, μ_acc, M2_acc, n_child, μs[j], M2s[j])
    end
    return n_acc, μ_acc, M2_acc
end

"""
    merge_covariance_children(n_child, μxs, μys, Cs) -> (n, μx, μy, C)

Fold several subgroups using [`merge_covariance_pebay`](@ref); same layout as
[`merge_variance_children`](@ref) but for paired `(x,y)` summaries.
"""
function merge_covariance_children(
    n_child::Integer,
    μxs::AbstractVector{FT},
    μys::AbstractVector{FT},
    Cs::AbstractVector{FT},
) where {FT <: AbstractFloat}
    k = length(μxs)
    length(μys) == k || throw(DimensionMismatch("μxs and μys"))
    length(Cs) == k || throw(DimensionMismatch("Cs length must match μxs"))
    k >= 1 || throw(ArgumentError("need at least one subgroup to merge"))
    n_acc = n_child
    μx_acc = μxs[1]
    μy_acc = μys[1]
    C_acc = Cs[1]
    for j in 2:k
        n_acc, μx_acc, μy_acc, C_acc = merge_covariance_pebay(
            n_acc, μx_acc, μy_acc, C_acc,
            n_child, μxs[j], μys[j], Cs[j],
        )
    end
    return n_acc, μx_acc, μy_acc, C_acc
end
