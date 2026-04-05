module CairoMakieExt

using CairoMakie: CairoMakie as CM
using Dates: Dates
using DataFrames: DataFrames
using Printf: Printf
using Random: Random
using Statistics: Statistics

using MLCondensateDistributions: MLCondensateDistributions as MLCD

"""Short string for resolution tick labels (avoids 10+ significant figures)."""
function _resolution_tick_string(x::Float64)
    !isfinite(x) && return "NaN"
    ax = abs(x)
    (ax >= 10_000 || (ax > 0 && ax < 0.01)) && return Printf.@sprintf("%.4g", x)
    return string(round(x; sigdigits=5))
end

"""
At most `max_ticks` positions from sorted unique values, for readable axes when there are
hundreds of distinct resolutions (e.g. `resolution_z`).
"""
function _compact_resolution_ticks(vals::AbstractVector{<:Real}; max_ticks::Int=12)
    v = sort(unique(Float64.(vals)))
    n = length(v)
    n == 0 && return Float64[], String[]
    if n <= max_ticks
        return v, [_resolution_tick_string(x) for x in v]
    end
    idx = unique(round.(Int, range(1, n; length=max_ticks)))
    return v[idx], [_resolution_tick_string(x) for x in v[idx]]
end

"""Makie heatmap edges for `n` cells of equal width in index space (avoids Cairo triangulation on irregular physical edges)."""
function _uniform_index_edges(n::Int)
    n <= 0 && return Float64[]
    return collect(range(0.5, n + 0.5; length=n + 1))
end

"""
Tick positions in **1-based index space** (cell centers `1..n`) with labels from sorted unique `xvals`.
"""
function _compact_resolution_index_ticks(xvals::Vector{Float64}; max_ticks::Int=14)
    n = length(xvals)
    n == 0 && return Float64[], String[]
    if n <= max_ticks
        return collect(1.0:n), [_resolution_tick_string(x) for x in xvals]
    end
    idx = unique(round.(Int, range(1, n; length=max_ticks)))
    return Float64.(idx), [_resolution_tick_string(xvals[i]) for i in idx]
end

"""Sequential: white = zero log-count, dark blue = dense (not the moment value)."""
function _log_density_colormap()
    return CM.cgrad([:white, :lightsteelblue, :royalblue, :midnightblue]; categorical=false)
end

"""Diverging for signed quantities: blue (negative) → white (0) → red (positive); avoids viridis-style purple at an end."""
function _covariance_diverging_colormap()
    return CM.cgrad([:dodgerblue, :lightblue, :white, :wheat, :red]; categorical=false)
end

"""Sequential for nonnegative magnitudes: white (small) → amber → brown (no purple)."""
function _sequential_positive_colormap()
    return CM.cgrad([:white, :wheat, :goldenrod, :saddlebrown]; categorical=false)
end

"""Sequential for generic matrices (training diagnostics): light (low) → dark (high), discrete cells only."""
function _matrix_sequential_colormap()
    return CM.cgrad([:white, :gainsboro, :gray60, :gray15]; categorical=false)
end

"""
Group target columns into the `q_`, `var_`, and `cov_` subsets used by the diagnostics plots.
"""
function _target_groups(target_cols::Vector{Symbol})
    q_targets = [t for t in target_cols if startswith(String(t), "q_")]
    var_targets = [t for t in target_cols if startswith(String(t), "var_")]
    cov_targets = [t for t in target_cols if startswith(String(t), "cov_")]
    return q_targets, var_targets, cov_targets
end

"""Plot training/test loss history on a single axis and save it to `out_dir`."""
function _plot_losses(train_loss::Vector{Float64}, test_loss::Vector{Float64}, out_dir::String)
    fig = CM.Figure(size=(1000, 500))
    ax = CM.Axis(fig[1, 1], title="Train/Test Loss", xlabel="Epoch", ylabel="MSE")
    CM.lines!(ax, 1:length(train_loss), train_loss, label="train")
    CM.lines!(ax, 1:length(test_loss), test_loss, label="test")
    CM.axislegend(ax, position=:rt)
    CM.save(joinpath(out_dir, "loss_train_test.png"), fig)
end

"""Plot a single training loss curve (checkpoint path has no held-out test history)."""
function _plot_training_loss(loss_history::Vector{Float64}, out_dir::String)
    fig = CM.Figure(size=(1000, 500))
    ax = CM.Axis(fig[1, 1], title="Training Loss", xlabel="Step", ylabel="MSE")
    CM.lines!(ax, 1:length(loss_history), loss_history, label="train")
    CM.axislegend(ax, position=:rt)
    CM.save(joinpath(out_dir, "training_loss.png"), fig)
end

"""
Render truth-vs-prediction histograms for a group of target columns.

The plot uses shared bins and special-cases constant targets so the result stays
readable even when a target is mostly or entirely zero.
"""
function _plot_group_distributions(group_name::String, cols::Vector{Symbol}, target_cols::Vector{Symbol}, y_true::AbstractMatrix, y_pred::AbstractMatrix, out_dir::String)
    isempty(cols) && return
    n = length(cols)
    fig = CM.Figure(size=(max(500, 350 * n), 400))
    for (i, c) in enumerate(cols)
        idx = findfirst(==(c), target_cols)
        truth = vec(Float64.(y_true[idx, :]))
        pred = vec(Float64.(y_pred[idx, :]))
        ax = CM.Axis(fig[1, i], title=String(c), xlabel="Value", ylabel="Count")

        if MLCD.Common.is_constant(truth)
            CM.hist!(ax, pred; bins=60, color=(:orange, 0.6))
            const_val = first(truth)
            CM.vlines!(ax, [const_val], color=:blue, linewidth=3)
        else
            lo, hi = MLCD.Common.robust_limits(vcat(truth, pred))
            bins = range(lo, hi; length=70)
            CM.hist!(ax, pred; bins=bins, color=(:orange, 0.45))
            ax.limits = (lo, hi, nothing, nothing)
        end
    end
    CM.Label(fig[2, 1:n], "Blue = Truth, Orange = Prediction")
    CM.save(joinpath(out_dir, "$(group_name)_distributions.png"), fig)
end

"""
Render truth-vs-prediction scatter plots for a group of target columns.

The data are subsampled for readability, and constant targets fall back to a
simple prediction histogram rather than a degenerate scatter.
"""
function _plot_group_scatter(group_name::String, cols::Vector{Symbol}, target_cols::Vector{Symbol}, y_true::AbstractMatrix, y_pred::AbstractMatrix, out_dir::String; max_points::Int=5000)
    isempty(cols) && return
    n = length(cols)
    fig = CM.Figure(size=(max(500, 350 * n), 400))
    for (i, c) in enumerate(cols)
        idx = findfirst(==(c), target_cols)
        truth = vec(Float64.(y_true[idx, :]))
        pred = vec(Float64.(y_pred[idx, :]))
        m = min(length(truth), max_points)
        pick = Random.randperm(length(truth))[1:m]

        if MLCD.Common.is_constant(truth)
            ax = CM.Axis(fig[1, i], title=String(c) * " (constant truth)", xlabel="Prediction", ylabel="Count")
            CM.hist!(ax, pred; bins=60, color=(:orange, 0.6))
            continue
        end

        lo, hi = MLCD.Common.robust_limits(vcat(truth, pred))
        ax = CM.Axis(fig[1, i], title=String(c), xlabel="Truth", ylabel="Prediction")
        CM.scatter!(ax, truth[pick], pred[pick]; markersize=2, color=(:dodgerblue, 0.35))
        CM.lines!(ax, [lo, hi], [lo, hi]; color=:black, linestyle=:dash, linewidth=1.5)
        ax.limits = (lo, hi, lo, hi)
    end
    CM.save(joinpath(out_dir, "$(group_name)_truth_vs_pred_scatter.png"), fig)
end

"""
Render compact truth and prediction heatmaps for the full target matrix.

The view is subsampled to keep the diagnostic image tractable on large test sets.
"""
function _plot_moment_heatmaps(target_cols::Vector{Symbol}, y_true::AbstractMatrix, y_pred::AbstractMatrix, out_dir::String)
    n = min(size(y_true, 2), 1000)
    pick = Random.randperm(size(y_true, 2))[1:n]
    yt = Float64.(y_true[:, pick])
    yp = Float64.(y_pred[:, pick])

    lo, hi = MLCD.Common.robust_limits(vcat(vec(yt), vec(yp)); qlo=0.01, qhi=0.99)
    fig = CM.Figure(size=(1200, 650))

    ax1 = CM.Axis(fig[1, 1], title="Truth Matrix (first $(n) samples)", xlabel="Sample", ylabel="Target")
    hm1 = CM.heatmap!(
        ax1,
        1:n,
        1:length(target_cols),
        Float32.(yt);
        colormap=_matrix_sequential_colormap(),
        colorrange=(Float32(lo), Float32(hi)),
        interpolate=false,
    )
    ax1.yticks = (1:length(target_cols), string.(target_cols))
    CM.Colorbar(fig[1, 2], hm1)

    ax2 = CM.Axis(fig[2, 1], title="Prediction Matrix (first $(n) samples)", xlabel="Sample", ylabel="Target")
    hm2 = CM.heatmap!(
        ax2,
        1:n,
        1:length(target_cols),
        Float32.(yp);
        colormap=_matrix_sequential_colormap(),
        colorrange=(Float32(lo), Float32(hi)),
        interpolate=false,
    )
    ax2.yticks = (1:length(target_cols), string.(target_cols))
    CM.Colorbar(fig[2, 2], hm2)

    CM.save(joinpath(out_dir, "target_truth_pred_matrices_full.png"), fig)
end

"""Write a CSV summary of truth and prediction distributions for each target."""
function _write_target_summary_table(target_cols::Vector{Symbol}, y_true::AbstractMatrix, y_pred::AbstractMatrix, out_dir::String)
    rows = String[]
    push!(rows, "target,true_zero_frac,true_min,true_p01,true_p50,true_p99,true_max,pred_zero_frac,pred_min,pred_p01,pred_p50,pred_p99,pred_max")

    for (i, target) in enumerate(target_cols)
        t = vec(Float64.(y_true[i, :]))
        p = vec(Float64.(y_pred[i, :]))
        push!(rows, string(
            target, ",",
            count(iszero, t) / length(t), ",", Statistics.minimum(t), ",", Statistics.quantile(t, 0.01), ",", Statistics.quantile(t, 0.50), ",", Statistics.quantile(t, 0.99), ",", Statistics.maximum(t), ",",
            count(iszero, p) / length(p), ",", Statistics.minimum(p), ",", Statistics.quantile(p, 0.01), ",", Statistics.quantile(p, 0.50), ",", Statistics.quantile(p, 0.99), ",", Statistics.maximum(p)
        ))
    end

    open(joinpath(out_dir, "target_summary.csv"), "w") do io
        write(io, join(rows, "\n"))
    end
end

"""
Render conditional median views for `q_liq` and `q_ice` against `qt` and `theta_li`.
"""
function _plot_q_conditional_views(feature_cols::Vector{Symbol}, target_cols::Vector{Symbol}, x_test_raw::AbstractMatrix, y_true::AbstractMatrix, y_pred::AbstractMatrix, out_dir::String)
    idx_qt = findfirst(==(:qt), feature_cols)
    idx_th = findfirst(==(:theta_li), feature_cols)
    idx_ql = findfirst(==(:q_liq), target_cols)
    idx_qi = findfirst(==(:q_ice), target_cols)

    if any(isnothing, (idx_qt, idx_th, idx_ql, idx_qi))
        return
    end

    qt = vec(Float64.(x_test_raw[idx_qt, :]))
    th = vec(Float64.(x_test_raw[idx_th, :]))
    ql_t = vec(Float64.(y_true[idx_ql, :]))
    ql_p = vec(Float64.(y_pred[idx_ql, :]))
    qi_t = vec(Float64.(y_true[idx_qi, :]))
    qi_p = vec(Float64.(y_pred[idx_qi, :]))

    fig = CM.Figure(size=(1400, 900))

    x1, y1t = MLCD.Common.binned_median(qt, ql_t)
    _, y1p = MLCD.Common.binned_median(qt, ql_p)
    ax11 = CM.Axis(fig[1, 1], title="q_liq vs qt (binned median)", xlabel="qt", ylabel="q_liq")
    CM.lines!(ax11, x1, y1t; color=:blue, linewidth=2, label="truth")
    CM.lines!(ax11, x1, y1p; color=:orange, linewidth=2, label="pred")
    CM.axislegend(ax11, position=:rt)

    x2, y2t = MLCD.Common.binned_median(th, ql_t)
    _, y2p = MLCD.Common.binned_median(th, ql_p)
    ax12 = CM.Axis(fig[1, 2], title="q_liq vs theta_li (binned median)", xlabel="theta_li", ylabel="q_liq")
    CM.lines!(ax12, x2, y2t; color=:blue, linewidth=2, label="truth")
    CM.lines!(ax12, x2, y2p; color=:orange, linewidth=2, label="pred")
    CM.axislegend(ax12, position=:rt)

    x3, y3t = MLCD.Common.binned_median(qt, qi_t)
    _, y3p = MLCD.Common.binned_median(qt, qi_p)
    ax21 = CM.Axis(fig[2, 1], title="q_ice vs qt (binned median)", xlabel="qt", ylabel="q_ice")
    CM.lines!(ax21, x3, y3t; color=:blue, linewidth=2, label="truth")
    CM.lines!(ax21, x3, y3p; color=:orange, linewidth=2, label="pred")
    CM.axislegend(ax21, position=:rt)

    x4, y4t = MLCD.Common.binned_median(th, qi_t)
    _, y4p = MLCD.Common.binned_median(th, qi_p)
    ax22 = CM.Axis(fig[2, 2], title="q_ice vs theta_li (binned median)", xlabel="theta_li", ylabel="q_ice")
    CM.lines!(ax22, x4, y4t; color=:blue, linewidth=2, label="truth")
    CM.lines!(ax22, x4, y4p; color=:orange, linewidth=2, label="pred")
    CM.axislegend(ax22, position=:rt)

    CM.save(joinpath(out_dir, "q_targets_conditional_vs_inputs.png"), fig)
end

"""
Plot grouped medians of each target versus a single resolution axis and save the figure.
"""
function _resolution_density_heatmap(df::DataFrames.DataFrame, res_col::Symbol, target::Symbol; nbins::Int=48)
    xvals = sort(unique(Float64.(df[!, res_col])))
    yvals = Float64.(df[!, target])
    ylo, yhi = MLCD.Common.robust_limits(yvals; qlo=0.01, qhi=0.99)
    if ylo == yhi
        if startswith(String(target), "var_")
            ylo = max(0.0, ylo - 0.5)
            yhi = yhi + 0.5
        else
            ylo -= 0.5
            yhi += 0.5
        end
    end

    yedges = collect(range(ylo, yhi; length=nbins + 1))
    counts = fill(0.0, nbins, length(xvals))
    res_vals = Float64.(df[!, res_col])

    for (j, x) in enumerate(xvals)
        ys = yvals[res_vals .== x]
        ys = ys[isfinite.(ys)]
        isempty(ys) && continue

        for b in 1:nbins
            in_bin = b == nbins ? (ys .>= yedges[b]) .& (ys .<= yedges[b + 1]) : (ys .>= yedges[b]) .& (ys .< yedges[b + 1])
            counts[b, j] = count(in_bin)
        end
    end

    return xvals, yedges, counts
end

function MLCD.Viz.plot_targets_vs_resolution(df::DataFrames.DataFrame, targets::Vector{Symbol}, res_col::Symbol, out_path::String)
    n = length(targets)
    ncol = min(4, max(1, n))
    nrow = cld(n, ncol)
    fig = CM.Figure(size=(560 * ncol, 320 * nrow))

    for (i, target) in enumerate(targets)
        r = cld(i, ncol)
        c = ((i - 1) % ncol) + 1
        plot_col = 2 * c - 1
        cb_col = 2 * c
        ax = CM.Axis(fig[r, plot_col], title=String(target), xlabel=String(res_col), ylabel=String(target))

        xvals, yedges, counts = _resolution_density_heatmap(df, res_col, target)
        density = log10.(counts .+ 1)
        nx = size(counts, 2)
        mat = permutedims(density)
        xe_idx = _uniform_index_edges(nx)
        hm = CM.heatmap!(
            ax,
            xe_idx,
            yedges,
            mat;
            colormap=_log_density_colormap(),
            colorrange=(0.0, max(1e-6, Float64(Statistics.maximum(mat)))),
            interpolate=false,
        )
        CM.Colorbar(fig[r, cb_col], hm)
        tx_pos, tx_lab = _compact_resolution_index_ticks(xvals; max_ticks=14)
        ax.xticks = (tx_pos, tx_lab)

        g = MLCD.Analysis.group_reduce(df, [res_col], target; reducer=Statistics.median, output_col=:median)
        x_raw = Float64.(g[!, res_col])
        y_med = Float64.(g[!, :median])
        col_to_i = Dict(Float64(xvals[j]) => Float64(j) for j in eachindex(xvals))
        x_idx = Float64[col_to_i[xr] for xr in x_raw]
        CM.scatter!(ax, x_idx, y_med; color=:black, markersize=7)

        if startswith(String(target), "cov_")
            CM.hlines!(ax, 0.0; color=(:black, 0.45), linestyle=:dash, linewidth=1.2)
        end
    end

    mkpath(dirname(out_path))
    CM.save(out_path, fig)
    return out_path
end

"""
Plot 2D resolution heatmaps for each target and save the composite figure.
"""
function MLCD.Viz.plot_targets_heatmaps(df::DataFrames.DataFrame, targets::Vector{Symbol}, out_path::String)
    n = length(targets)
    ncol = min(3, max(1, n))
    nrow = cld(n, ncol)
    fig = CM.Figure(size=(560 * ncol, 360 * nrow))

    for (i, target) in enumerate(targets)
        r = cld(i, ncol)
        c = ((i - 1) % ncol) + 1
        plot_col = 2 * c - 1
        cb_col = 2 * c
        ax = CM.Axis(fig[r, plot_col], title=String(target), xlabel="resolution_h", ylabel="resolution_z")
        hs = sort(unique(Float64.(df[!, :resolution_h])))
        zs = sort(unique(Float64.(df[!, :resolution_z])))

        if length(hs) < 4 || length(zs) < 4
            grouped = DataFrames.combine(
                DataFrames.groupby(df, [:resolution_h, :resolution_z]),
                target => Statistics.median => :median,
                target => length => :count,
            )
            x = Float64.(grouped[!, :resolution_h])
            y = Float64.(grouped[!, :resolution_z])
            z = Float64.(grouped[!, :median])
            counts = Float64.(grouped[!, :count])
            sizes = 10 .+ 40 .* sqrt.(counts ./ Statistics.maximum(counts))

            if startswith(String(target), "cov_")
                finite_vals = [v for v in z if isfinite(v)]
                maxabs = isempty(finite_vals) ? 1f0 : Statistics.maximum(abs, finite_vals)
                cr = isfinite(maxabs) && maxabs > 0 ? (-maxabs, maxabs) : (-1f0, 1f0)
                pts = CM.scatter!(
                    ax,
                    x,
                    y;
                    color=z,
                    colormap=_covariance_diverging_colormap(),
                    colorrange=cr,
                    markersize=sizes,
                )
                CM.Colorbar(fig[r, cb_col], pts)
            else
                finite_z = [v for v in z if isfinite(v)]
                cr_seq = if isempty(finite_z)
                    (0.0, 1.0)
                else
                    loz, hiz = Statistics.extrema(finite_z)
                    loz == hiz ? (Float64(loz - 1), Float64(loz + 1)) : (Float64(loz), Float64(hiz))
                end
                pts = CM.scatter!(
                    ax,
                    x,
                    y;
                    color=z,
                    colormap=_sequential_positive_colormap(),
                    colorrange=cr_seq,
                    markersize=sizes,
                )
                CM.Colorbar(fig[r, cb_col], pts)
            end
        else
            hs, zs, mat = MLCD.Analysis.build_heatmap_matrix(df, :resolution_h, :resolution_z, target; reducer=Statistics.median)
            # Uniform index edges: irregular physical dz/dh spacing confuses Cairo heatmap meshing (diagonal shards).
            mat_xy = permutedims(mat)
            nh, nz = size(mat_xy)
            xe_idx = _uniform_index_edges(nh)
            ye_idx = _uniform_index_edges(nz)

            if startswith(String(target), "cov_")
                finite_vals = [x for x in vec(mat) if isfinite(x)]
                maxabs = isempty(finite_vals) ? 1f0 : Statistics.maximum(abs, finite_vals)
                cr = isfinite(maxabs) && maxabs > 0 ? (-Float64(maxabs), Float64(maxabs)) : (-1.0, 1.0)
                hm = CM.heatmap!(
                    ax,
                    xe_idx,
                    ye_idx,
                    mat_xy;
                    colormap=_covariance_diverging_colormap(),
                    colorrange=cr,
                    interpolate=false,
                    nan_color=:lightgray,
                )
                CM.Colorbar(fig[r, cb_col], hm)
            else
                finite_m = [v for v in vec(mat_xy) if isfinite(v)]
                cr_var = if isempty(finite_m)
                    (0.0, 1.0)
                else
                    lo = Float64(Statistics.minimum(finite_m))
                    hi = Float64(Statistics.maximum(finite_m))
                    lo == hi ? (lo - 1.0, hi + 1.0) : (lo, hi)
                end
                hm = CM.heatmap!(
                    ax,
                    xe_idx,
                    ye_idx,
                    mat_xy;
                    colormap=_sequential_positive_colormap(),
                    colorrange=cr_var,
                    interpolate=false,
                    nan_color=:lightgray,
                )
                CM.Colorbar(fig[r, cb_col], hm)
            end

            hx_pos, hx_lab = _compact_resolution_index_ticks(hs; max_ticks=12)
            zx_pos, zx_lab = _compact_resolution_index_ticks(zs; max_ticks=12)
            ax.xticks = (hx_pos, hx_lab)
            ax.yticks = (zx_pos, zx_lab)
        end
    end

    mkpath(dirname(out_path))
    CM.save(out_path, fig)
    return out_path
end

"""
Plot binned median of each `target` column versus continuous `x_col` and save a multi-panel figure.
"""
function MLCD.Viz.plot_targets_binned_vs_x(
    df::DataFrames.DataFrame,
    targets::Vector{Symbol},
    x_col::Symbol,
    out_path::String;
    nbins::Int = 40,
    title_suffix::String = "",
)
    isempty(targets) && error("plot_targets_binned_vs_x: targets is empty")
    n = length(targets)
    ncol = min(4, max(1, n))
    nrow = cld(n, ncol)
    fig = CM.Figure(size=(560 * ncol, 320 * nrow))
    xlab = String(x_col)
    for (i, target) in enumerate(targets)
        r = cld(i, ncol)
        c = ((i - 1) % ncol) + 1
        ttl = String(target) * title_suffix
        ax = CM.Axis(fig[r, c]; title=ttl, xlabel=xlab, ylabel=String(target))
        xv = Float64.(df[!, x_col])
        yv = Float64.(df[!, target])
        xc, ym = MLCD.Common.binned_median(xv, yv; nbins=nbins)
        if !isempty(xc)
            CM.lines!(ax, xc, ym; linewidth=2, color=:steelblue)
        end
    end
    mkpath(dirname(out_path))
    CM.save(out_path, fig)
    return out_path
end

"""
Render the full diagnostics bundle from a trained model artifact.
"""
function MLCD.Viz.plot_training_diagnostics_from_artifact(artifact, out_dir::String)
    feature_cols = Vector{Symbol}(artifact.feature_cols)
    target_cols = Vector{Symbol}(artifact.target_cols)
    x_test_raw = artifact.X_test_raw
    y_true = artifact.Y_test
    y_pred = artifact.Y_test_pred

    _plot_losses(artifact.train_loss_history, artifact.test_loss_history, out_dir)
    _plot_moment_heatmaps(target_cols, y_true, y_pred, out_dir)

    q_targets, var_targets, cov_targets = _target_groups(target_cols)
    _plot_group_distributions("q_targets", q_targets, target_cols, y_true, y_pred, out_dir)
    _plot_group_distributions("variance_targets", var_targets, target_cols, y_true, y_pred, out_dir)
    _plot_group_distributions("covariance_targets", cov_targets, target_cols, y_true, y_pred, out_dir)

    _plot_group_scatter("q_targets", q_targets, target_cols, y_true, y_pred, out_dir)
    _plot_group_scatter("variance_targets", var_targets, target_cols, y_true, y_pred, out_dir)
    _plot_group_scatter("covariance_targets", cov_targets, target_cols, y_true, y_pred, out_dir)

    MLCD.Common.write_metrics_table(artifact.metrics, out_dir)
    _write_target_summary_table(target_cols, y_true, y_pred, out_dir)
    _plot_q_conditional_views(feature_cols, target_cols, x_test_raw, y_true, y_pred, out_dir)

    return out_dir
end

"""
Write the standard training diagnostics bundle for a model checkpoint.
"""
function MLCD.write_training_diagnostics(model, ps, st, X_raw, X_std, Y_true, feature_cols, target_cols, loss_history, model_dir)
    out_dir = MLCD.Viz.diagnostics_output_dir(model_dir)
    y_pred, _ = model(X_std, ps, st)

    _plot_training_loss(loss_history, out_dir)
    _plot_feature_matrix(X_raw, feature_cols, out_dir)
    _plot_truth_pred(Y_true, y_pred, target_cols, out_dir)
    return out_dir
end

"""
Render feature-value and feature-correlation heatmaps from the raw input matrix.
"""
function _plot_feature_matrix(X_raw::AbstractMatrix{<:Real}, feature_cols::Vector{Symbol}, out_dir::String)
    n = min(size(X_raw, 2), 400)
    fig = CM.Figure(size=(1000, 500))
    ax = CM.Axis(fig[1, 1], title="Feature Matrix (first $(n) samples)", xlabel="Sample", ylabel="Feature")
    hm = CM.heatmap!(
        ax,
        1:n,
        1:length(feature_cols),
        Float32.(X_raw[:, 1:n]);
        colormap=_matrix_sequential_colormap(),
        interpolate=false,
    )
    ax.yticks = (1:length(feature_cols), string.(feature_cols))
    CM.Colorbar(fig[1, 2], hm)
    CM.save(joinpath(out_dir, "feature_matrix_examples.png"), fig)

    corr_mat = Statistics.cor(permutedims(Float32.(X_raw[:, 1:n])))
    fig2 = CM.Figure(size=(700, 600))
    ax2 = CM.Axis(fig2[1, 1], title="Feature Correlation", xlabel="Feature", ylabel="Feature")
    hm2 = CM.heatmap!(
        ax2,
        1:length(feature_cols),
        1:length(feature_cols),
        corr_mat;
        colormap=_covariance_diverging_colormap(),
        colorrange=(-1.0, 1.0),
        interpolate=false,
    )
    ax2.xticks = (1:length(feature_cols), string.(feature_cols))
    ax2.yticks = (1:length(feature_cols), string.(feature_cols))
    CM.Colorbar(fig2[1, 2], hm2)
    CM.save(joinpath(out_dir, "feature_correlation.png"), fig2)
end

"""
Render truth-vs-prediction matrices, scatter plots, and marginal distributions.
"""
function _plot_truth_pred(Y_true::AbstractMatrix{<:Real}, Y_pred::AbstractMatrix{<:Real}, target_cols::Vector{Symbol}, out_dir::String)
    n = min(size(Y_true, 2), 800)

    fig = CM.Figure(size=(1000, 500))
    ax1 = CM.Axis(fig[1, 1], title="Truth Matrix (first $(n) samples)", xlabel="Sample", ylabel="Target")
    lo_m, hi_m = MLCD.Common.robust_limits(
        vcat(vec(Float64.(Y_true[:, 1:n])), vec(Float64.(Y_pred[:, 1:n])));
        qlo=0.01,
        qhi=0.99,
    )
    cr_m = (Float32(lo_m), Float32(hi_m))
    hm1 = CM.heatmap!(
        ax1,
        1:n,
        1:length(target_cols),
        Float32.(Y_true[:, 1:n]);
        colormap=_matrix_sequential_colormap(),
        colorrange=cr_m,
        interpolate=false,
    )
    ax1.yticks = (1:length(target_cols), string.(target_cols))
    CM.Colorbar(fig[1, 2], hm1)

    ax2 = CM.Axis(fig[2, 1], title="Prediction Matrix (first $(n) samples)", xlabel="Sample", ylabel="Target")
    hm2 = CM.heatmap!(
        ax2,
        1:n,
        1:length(target_cols),
        Float32.(Y_pred[:, 1:n]);
        colormap=_matrix_sequential_colormap(),
        colorrange=cr_m,
        interpolate=false,
    )
    ax2.yticks = (1:length(target_cols), string.(target_cols))
    CM.Colorbar(fig[2, 2], hm2)
    CM.save(joinpath(out_dir, "truth_pred_matrices.png"), fig)

    fig2 = CM.Figure(size=(1000, 400))
    for i in 1:length(target_cols)
        ax = CM.Axis(fig2[1, i], title="$(target_cols[i]): Truth vs Pred", xlabel="Truth", ylabel="Prediction")
        CM.scatter!(ax, Float32.(Y_true[i, 1:n]), Float32.(Y_pred[i, 1:n]); markersize=3)
    end
    CM.save(joinpath(out_dir, "truth_vs_pred_scatter.png"), fig2)

    fig3 = CM.Figure(size=(1000, 400))
    for i in 1:length(target_cols)
        ax = CM.Axis(fig3[1, i], title="$(target_cols[i]) Distribution", xlabel="Value", ylabel="Count")
        CM.hist!(ax, Float32.(Y_true[i, 1:n]); bins=40, color=(:blue, 0.5))
        CM.hist!(ax, Float32.(Y_pred[i, 1:n]); bins=40, color=(:orange, 0.5))
    end
    CM.Label(fig3[2, 1:2], "Blue = Truth, Orange = Prediction")
    CM.save(joinpath(out_dir, "target_distributions.png"), fig3)
end

end