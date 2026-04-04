using DataFrames: DataFrames
using Arrow: Arrow
using HTTP: HTTP
using DimensionalData: DimensionalData

function _googleles_tabular_one_span!(
    z_range::UnitRange{Int},
    q_c_buf,
    non_qc_buffers,
    clw_buf_full,
    cli_buf_full,
    non_qc_specs,
    fine_fields,
    metadata,
    spatial_info,
    site_id::Int,
    month::Int,
    experiment::String,
    case_tables::Vector{DataFrames.DataFrame},
    t_idx::Int,
)::Nothing
    q_c_span = @view q_c_buf[:, :, z_range]
    ta = @view non_qc_buffers["ta"][:, :, z_range]
    clw_span = @view clw_buf_full[:, :, z_range]
    cli_span = @view cli_buf_full[:, :, z_range]
    @inbounds for i in eachindex(q_c_span)
        clw_span[i], cli_span[i] = GoogleLES.partition_condensate(q_c_span[i], ta[i])
    end
    empty!(fine_fields)
    fine_fields["q_c"] = q_c_span
    for (_, c_var) in non_qc_specs
        fine_fields[c_var] = @view non_qc_buffers[c_var][:, :, z_range]
    end
    fine_fields["clw"] = clw_span
    fine_fields["cli"] = cli_span
    metadata_t = (; metadata..., timestep = t_idx)
    spatial_info_t = (; spatial_info..., dz_native_profile = @view spatial_info.dz_native_profile[z_range])
    df = DatasetBuilder.process_abstract_chunk(fine_fields, metadata_t, spatial_info_t)
    if DataFrames.nrow(df) > 0
        _assert_finite_dataframe(df, "GoogleLES site=$(site_id) month=$(month) experiment=$(experiment) timestep=$(t_idx)")
        push!(case_tables, df)
    end
    return nothing
end

"""
    GoogleLES.build_tabular(site_id::Int, month::Int, experiment::String, output_dir::String; 
                               max_timesteps::Int=0, timestep_batch_size::Int=0, 
                               min_h_resolution::Float32=1000.0f0, verbose::Bool=false)

Build tabular Arrow training data from GoogleLES remote Zarr datasets with vertical z-level filtering.

# Algorithm Overview

Efficient single-pass pipeline for remote data processing:

1. **Materialize q_c**: Load q_c from remote into local buffer (once per timestep)
2. **Check clouds**: `_has_cloud_after_2x2(q_c_buf)` on local buffer; skip empty timesteps with `continue`
3. **Build z-mask**: Identify vertical levels surviving future coarsening
4. **Logical spans**: Decompose `z_keep_mask` into contiguous true runs (`_collect_true_spans`).
   The mask uses the full native vertical coarsening schedule (`z_future_factors`), so it is
   the conservative native-z set for `process_abstract_chunk`.
5. **Non-`q_c` Zarr reads (chunk-merge default)**: Spans whose storage z-chunks overlap are
   grouped; **`_load_googleles_timestep_fields_into_span_list!`** runs **one narrow Zarr slice
   per original span** (e.g. `10:20` then `40:50`) — **never** a widened hull passed to Zarr.
   Grouping is batching/orchestration only. Disable with `MLCD_GOOGLELES_Z_CHUNK_MERGE=0`
   (same narrow slices, one group per span).
6. **Local computation**: Coarse-graining uses `@view` into local buffers only (no lazy Zarr).

Each cloudy timestep: `q_c` is materialized once; non-`q_c` fields use chunk-overlap grouping by default (batched narrow slices per group, never a hull widened at the API).

# Critical Design Decision: Local Materialization

**NEVER compute on lazy remote-backed Zarr views.** Each array operation on a lazy remote view triggers decompression. Computing on lazy views → catastrophic gridlock.

Pattern: `read remote → copyto!(local buffer) → use @view of local buffer → compute`

See [docs/v2_remote_load_reduction_questions.md](../docs/v2_remote_load_reduction_questions.md) (“CRITICAL LESSON”) and the canonical pipeline write-up [docs/googleles_build_tabular.md](../docs/googleles_build_tabular.md).

# Arguments

- `site_id::Int`: GoogleLES site identifier
- `month::Int`: Month to process (1-12)
- `experiment::String`: Experiment name (e.g., "ctl", "p4K")
- `output_dir::String`: Directory where Arrow output file is written
- `max_timesteps::Int=0`: Maximum number of timesteps to process (0 = all)
- `timestep_batch_size::Int=0`: Batch size for memory-constrained processing (0 = auto-decide)
- `min_h_resolution::Float32=1000.0f0`: Minimum horizontal grid resolution to extract
- `verbose::Bool=false`: Enable verbose progress reporting

# Output

Writes an Arrow file `<output_dir>/googleles_case__<site_id>__month__<month>__exp__<experiment>.arrow` containing:
- Coarse-grained fields at multiple vertical resolution levels
- Metadata (site_id, month, experiment, timestep)
- Spatial grid information

Returns nothing; output is file-based.

# Memory Management

**Full-case mode** (when data fits in memory):
- Preallocates q_c and non-q_c 3D buffers (reused across timesteps)
- Loads all `q_c` for the case once; fetches other fields only for each contiguous `z_keep_mask` span

**Batched mode** (when data exceeds threshold):
- Processes in timestep batches; buffers reused within each batch
- Reduces peak memory without changing computational logic

# Performance Notes

- Non-`q_c` loads: **chunk-overlap groups** by default (`MLCD_GOOGLELES_Z_CHUNK_MERGE`): batched **narrow** `z_range` per logical span (no hull widening to Zarr); `merge=0` flattens to one span per group. Opt-in full-column: `MLCD_GOOGLELES_NONQC_STRATEGY=full_timestep`.
- Native `z_keep_mask` uses the same future vertical factors as the builder, so requested z-indices match reduction needs; levels that stay `false` are never fetched for non-`q_c` variables.
- `copyto!` after each span read keeps compute off lazy Zarr; a reused `scratch` dict avoids per-span `Dict` allocation.
- Preallocated 3D buffers are reused across timesteps and spans.

# Error Handling

- Returns early if no clouds found at any timestep
- Writes empty Arrow file with schema if no trainable rows extracted
- Validates all finite values before writing

"""
function GoogleLES.build_tabular(site_id::Int, month::Int, experiment::String, output_dir::String; max_timesteps::Int=0, timestep_batch_size::Int=0, min_h_resolution::Float32=1000.0f0, verbose::Bool=false)
    mkpath(output_dir)
    println("Processing GoogleLES case site=$(site_id) month=$(month) experiment=$(experiment)")

    ds = GoogleLES.load_zarr_simulation(site_id, month, experiment)
    if isnothing(ds)
        @error "Could not load simulation."
        _safe_close_http_pools!()
        return
    end

    nt = length(ds["t"])
    if max_timesteps > 0
        nt = min(nt, max_timesteps)
    end

    estimated_case_bytes = _estimate_googleles_case_bytes(ds)
    fullcase_bytes_limit = parse(Int, get(ENV, "MLCD_GOOGLELES_FULLCASE_BYTES_LIMIT", string(GOOGLELES_FULLCASE_BYTES_LIMIT)))
    force_fullcase = _parse_bool_env("MLCD_GOOGLELES_FORCE_FULLCASE", true)
    effective_batch_size = if timestep_batch_size > 0
        timestep_batch_size
    elseif force_fullcase
        nt
    elseif estimated_case_bytes <= fullcase_bytes_limit
        nt
    else
        min(GOOGLELES_DEFAULT_BATCH_SIZE, nt)
    end
    effective_batch_size = max(1, effective_batch_size)
    is_full_case = effective_batch_size >= nt

    x_coords = collect(ds["x"][:])
    z_coords = collect(ds["z"][:])
    dx_native = (x_coords[end] - x_coords[1]) / (length(x_coords) - 1)
    domain_h = x_coords[end] - x_coords[1]
    dz_native_profile = diff(z_coords)
    push!(dz_native_profile, dz_native_profile[end])

    metadata = (
        data_source = "GoogleLES",
        month = month,
        site_id = site_id,
        cfSite_number = site_id,
        forcing_model = "GoogleLES",
        experiment = experiment,
        verbose = verbose,
    )

    spatial_info = (
        dx_native = Float32(dx_native),
        domain_h = Float32(domain_h),
        min_h_resolution = Float32(min_h_resolution),
        dz_native_profile = Float32.(dz_native_profile),
        seeds_h = (1,),
    )
    z_schemes_native = CoarseGraining.compute_z_coarsening_scheme(spatial_info.dz_native_profile, 400f0)
    z_future_factors = Int[z_schemes_native[i][1] for i in 2:length(z_schemes_native)]

    if verbose
        @info "GoogleLES estimated working set=$(round(estimated_case_bytes / 1_000_000; digits=1)) MB, batch_size=$(effective_batch_size), full_case=$(is_full_case)."
    end

    non_qc_specs = [(g_var, c_var) for (g_var, c_var) in GOOGLELES_FIELD_SPECS if c_var != "q_c"]

    z_storage_cz = _googleles_effective_z_chunk_size(ds, "q_c")
    merge_z_chunks = _parse_bool_env("MLCD_GOOGLELES_Z_CHUNK_MERGE", true)
    if verbose
        @info "GoogleLES Zarr storage z-chunk=$z_storage_cz merge_overlapping_chunks=$merge_z_chunks"
    end

    case_tables = DataFrames.DataFrame[]
    processed_timesteps = 0
    started_at = time()
    processing_seconds = 0.0
    cache_seconds = 0.0

    if is_full_case
        if verbose
            @info "Loading full GoogleLES case into memory for $(site_id)/$(month)/$(experiment)..."
        end
        cache_started_at = time()
        q_cache = _load_googleles_cache(ds, 1:nt; field_specs=(("q_c", "q_c"),))
        cache_elapsed = time() - cache_started_at
        cache_seconds += cache_elapsed
        println("GoogleLES q_c cache load complete for site=$(site_id), month=$(month), nt=$(nt): $(round(cache_elapsed; digits=1))s")

        processing_started_at = time()

        # BUFFER SETUP: Pre-allocate all working buffers (reused across all timesteps).
        # This eliminates allocation overhead in the hottest loop.
        q_c0 = @view q_cache["q_c"][1, :, :, :]
        q_c_buf = similar(q_c0)  # Workspace for materializing q_c per timestep
        clw_buf_full = similar(q_c0)  # Workspace for condensed liquid water partition
        cli_buf_full = similar(q_c0)  # Workspace for condensed ice water partition
        non_qc_buffers = Dict{String, Array{Float32, 3}}()  # Dict of preallocated buffers for non-q_c fields
        for (_, c_var) in non_qc_specs
            non_qc_buffers[c_var] = similar(q_c0)
        end
        fine_fields = Dict{String, AbstractArray{Float32, 3}}()  # Reusable field dict for chunk processing
        non_qc_zarr_scratch = Dict{String, AbstractArray{Float32, 3}}()  # Reused for lazy zarr views; no per-span Dict alloc

        # SINGLE PASS: Detect clouds, build z-mask, and process all spans in same iteration.
        # Materializes q_c once per timestep; avoids vector reallocation from push!.
        for local_t in 1:nt
            step_started_at = time()
            processed_timesteps += 1
            t_idx = local_t - 1
            _progress_print("GoogleLES", processed_timesteps, nt, "site=$(site_id) month=$(month) experiment=$(experiment) timestep=$(t_idx)", processing_started_at)

            # CRITICAL: Materialize q_c into q_c_buf immediately (once per timestep).
            # Accessing a lazy Zarr view repeatedly during _has_cloud_after_2x2 can trigger
            # repeated block decompression. Materialization ensures a single sequential read.
            q_c_buf .= q_cache["q_c"][local_t, :, :, :]
            if !_has_cloud_after_2x2(q_c_buf)
                processing_seconds += (time() - step_started_at)
                continue
            end

            # Build sparse z-level mask: true for levels that survive all future coarsening,
            # false for levels that will be completely coarsened away.
            empty_z_levels = CoarseGraining.identify_empty_z_levels(q_c_buf, DatasetBuilder.CLOUD_PRESENCE_THRESHOLD)
            z_keep_mask = CoarseGraining.build_z_level_keep_mask(empty_z_levels, 1, z_future_factors)
            any(z_keep_mask) || begin
                processing_seconds += (time() - step_started_at)
                continue
            end

            nz_z = length(z_keep_mask)
            n_keep_z = count(z_keep_mask)
            n_spans_z = _count_true_spans(z_keep_mask)
            load_full_nonqc = _googleles_use_full_nonqc_timestep_load(n_spans_z, n_keep_z, nz_z)
            spans = _collect_true_spans(z_keep_mask)
            if load_full_nonqc
                _materialize_googleles_nonqc_timestep_into!(non_qc_buffers, ds, local_t; field_specs=non_qc_specs)
                for z_range in spans
                    _googleles_tabular_one_span!(
                        z_range,
                        q_c_buf,
                        non_qc_buffers,
                        clw_buf_full,
                        cli_buf_full,
                        non_qc_specs,
                        fine_fields,
                        metadata,
                        spatial_info,
                        site_id,
                        month,
                        experiment,
                        case_tables,
                        t_idx,
                    )
                end
            else
                span_groups = if merge_z_chunks
                    _group_mask_spans_by_overlapping_z_chunks(spans, nz_z, z_storage_cz)
                else
                    Vector{UnitRange{Int}}[[r] for r in spans]
                end
                for orig_spans in span_groups
                    _load_googleles_timestep_fields_into_span_list!(
                        non_qc_buffers,
                        ds,
                        local_t;
                        field_specs=non_qc_specs,
                        orig_spans=orig_spans,
                        scratch=non_qc_zarr_scratch,
                    )
                    for z_range in orig_spans
                        _googleles_tabular_one_span!(
                            z_range,
                            q_c_buf,
                            non_qc_buffers,
                            clw_buf_full,
                            cli_buf_full,
                            non_qc_specs,
                            fine_fields,
                            metadata,
                            spatial_info,
                            site_id,
                            month,
                            experiment,
                            case_tables,
                            t_idx,
                        )
                    end
                end
            end

            processing_seconds += (time() - step_started_at)
            if processed_timesteps % 8 == 0 || processed_timesteps == nt
                avg_step = processing_seconds / max(processed_timesteps, 1)
                println("\nGoogleLES processing progress: site=$(site_id), month=$(month), processed=$(processed_timesteps)/$(nt), avg_step_seconds=$(round(avg_step; digits=2))")
            end
        end

        println("GoogleLES timestep processing complete for site=$(site_id), month=$(month): total=$(round(processing_seconds; digits=1))s, per_timestep=$(round(processing_seconds / max(nt, 1); digits=2))s")
    else
        if verbose
            @info "Processing GoogleLES case in batches of $(effective_batch_size); estimated working set exceeds threshold."
        end

        # BATCHED PROCESSING: Same logic as full-case, but for memory-constrained systems.
        # Process in chunks to keep peak memory below threshold.
        for batch_start in 0:effective_batch_size:(nt - 1)
            batch_stop = min(batch_start + effective_batch_size - 1, nt - 1)
            batch_range = batch_start + 1:batch_stop + 1
            cache_started_at = time()
            q_batch_cache = _load_googleles_cache(ds, batch_range; field_specs=(("q_c", "q_c"),))
            cache_elapsed = time() - cache_started_at
            cache_seconds += cache_elapsed
            println("GoogleLES q_c batch cache load: site=$(site_id), month=$(month), timesteps=$(first(batch_range)-1):$(last(batch_range)-1), seconds=$(round(cache_elapsed; digits=1))")

            processing_started_at = time()

            q_c_batch = q_batch_cache["q_c"]
            q_c0 = @view q_c_batch[1, :, :, :]
            # Allocate buffers for this batch (reused within batch)
            q_c_buf = similar(q_c0)
            clw_buf_full = similar(q_c0)
            cli_buf_full = similar(q_c0)
            non_qc_buffers = Dict{String, Array{Float32, 3}}()
            for (_, c_var) in non_qc_specs
                non_qc_buffers[c_var] = similar(q_c0)
            end
            fine_fields = Dict{String, AbstractArray{Float32, 3}}()
            non_qc_zarr_scratch = Dict{String, AbstractArray{Float32, 3}}()

            # SINGLE PASS: Detect clouds, build z-mask, and process all spans in same iteration.
            # Materializes q_c once per batch timestep; avoids vector reallocation from push!.
            for local_t in axes(q_c_batch, 1)
                step_started_at = time()
                processed_timesteps += 1
                t_idx = batch_start + local_t - 1
                _progress_print("GoogleLES", processed_timesteps, nt, "site=$(site_id) month=$(month) experiment=$(experiment) timestep=$(t_idx)", processing_started_at)

                q_c_buf .= q_c_batch[local_t, :, :, :]
                if !_has_cloud_after_2x2(q_c_buf)
                    processing_seconds += (time() - step_started_at)
                    continue
                end

                empty_z_levels = CoarseGraining.identify_empty_z_levels(q_c_buf, DatasetBuilder.CLOUD_PRESENCE_THRESHOLD)
                z_keep_mask = CoarseGraining.build_z_level_keep_mask(empty_z_levels, 1, z_future_factors)
                any(z_keep_mask) || begin
                    processing_seconds += (time() - step_started_at)
                    continue
                end

                nz_z = length(z_keep_mask)
                n_keep_z = count(z_keep_mask)
                n_spans_z = _count_true_spans(z_keep_mask)
                load_full_nonqc = _googleles_use_full_nonqc_timestep_load(n_spans_z, n_keep_z, nz_z)
                spans = _collect_true_spans(z_keep_mask)
                t_load = batch_start + local_t
                if load_full_nonqc
                    _materialize_googleles_nonqc_timestep_into!(non_qc_buffers, ds, t_load; field_specs=non_qc_specs)
                    for z_range in spans
                        _googleles_tabular_one_span!(
                            z_range,
                            q_c_buf,
                            non_qc_buffers,
                            clw_buf_full,
                            cli_buf_full,
                            non_qc_specs,
                            fine_fields,
                            metadata,
                            spatial_info,
                            site_id,
                            month,
                            experiment,
                            case_tables,
                            t_idx,
                        )
                    end
                else
                    span_groups = if merge_z_chunks
                        _group_mask_spans_by_overlapping_z_chunks(spans, nz_z, z_storage_cz)
                    else
                        Vector{UnitRange{Int}}[[r] for r in spans]
                    end
                    for orig_spans in span_groups
                        _load_googleles_timestep_fields_into_span_list!(
                            non_qc_buffers,
                            ds,
                            t_load;
                            field_specs=non_qc_specs,
                            orig_spans=orig_spans,
                            scratch=non_qc_zarr_scratch,
                        )
                        for z_range in orig_spans
                            _googleles_tabular_one_span!(
                                z_range,
                                q_c_buf,
                                non_qc_buffers,
                                clw_buf_full,
                                cli_buf_full,
                                non_qc_specs,
                                fine_fields,
                                metadata,
                                spatial_info,
                                site_id,
                                month,
                                experiment,
                                case_tables,
                                t_idx,
                            )
                        end
                    end
                end

                processing_seconds += (time() - step_started_at)
                if processed_timesteps % 8 == 0 || processed_timesteps == nt
                    avg_step = processing_seconds / max(processed_timesteps, 1)
                    println("\nGoogleLES processing progress: site=$(site_id), month=$(month), processed=$(processed_timesteps)/$(nt), avg_step_seconds=$(round(avg_step; digits=2))")
                end
            end
        end

        println("GoogleLES timestep processing complete for site=$(site_id), month=$(month): total=$(round(processing_seconds; digits=1))s, per_timestep=$(round(processing_seconds / max(nt, 1); digits=2))s")
    end

    final_df = if isempty(case_tables)
        nothing
    elseif length(case_tables) == 1
        case_tables[1]
    else
        reduce(vcat, case_tables)
    end

    _progress_finish()

    total_elapsed = cache_seconds + processing_seconds
    if total_elapsed > 0
        cache_pct = 100 * cache_seconds / total_elapsed
        processing_pct = 100 * processing_seconds / total_elapsed
        println("GoogleLES time breakdown: cache=$(round(cache_seconds; digits=1))s ($(round(cache_pct; digits=1))%), processing=$(round(processing_seconds; digits=1))s ($(round(processing_pct; digits=1))%), total=$(round(total_elapsed; digits=1))s")
    end

    if isnothing(final_df)
        out_file = case_arrow_path(site_id, month, experiment, output_dir)
        Arrow.write(out_file, _empty_googleles_case_df())
        if verbose
            @info "No trainable GoogleLES rows found for site=$site_id month=$month experiment=$experiment; wrote empty Arrow case file."
        end
        _safe_close_http_pools!()
        return
    end

    out_file = case_arrow_path(site_id, month, experiment, output_dir)
    _assert_finite_dataframe(final_df, "GoogleLES final site=$(site_id) month=$(month) experiment=$(experiment)")
    Arrow.write(out_file, final_df)
    println("Wrote GoogleLES case file $(out_file) with $(DataFrames.nrow(final_df)) rows.")
    _safe_close_http_pools!()
end

function cfSites.build_tabular(cfSite_number::Int, month::Int, forcing_model::String, experiment::String, output_dir::String; max_timesteps::Int=0, min_h_resolution::Float32=1000.0f0, verbose::Bool=false)
    """
        cfSites.build_tabular(cfSite_number::Int, month::Int, forcing_model::String, experiment::String, 
                                 output_dir::String; max_timesteps::Int=0, min_h_resolution::Float32=1000.0f0, 
                                 verbose::Bool=false)

    Build tabular Arrow training data from cfSites LES simulations (local netCDF files).

    # Differences from GoogleLES

    - cfSites data is stored locally as netCDF files (no remote Zarr)
    - No need for remote materialization optimization; all data is already local
    - Simple linear timestep processing (no batching needed)

    # Arguments

    - `cfSite_number::Int`: CFSite identifier
    - `month::Int`: Month (1-12)
    - `forcing_model::String`: Forcing model name (e.g., "HadGEM2-A")
    - `experiment::String`: Experiment label
    - `output_dir::String`: Output directory
    - `max_timesteps::Int=0`: Limit number of timesteps (0 = all)
    - `min_h_resolution::Float32=1000.0f0`: Minimum horizontal resolution
    - `verbose::Bool=false`: Enable verbose output

    # Output

    Writes Arrow file with coarse-grained fields and metadata.
    """
    mkpath(output_dir)
    println("Processing cfSites case site=$(cfSite_number) month=$(month) model=$(forcing_model) experiment=$(experiment)")

    vars_to_load = ["temperature", "qt", "ql", "qi", "u", "v", "w", "p", "rho", "thetali"]
    les_dir = cfSites.get_cfSite_les_dir(cfSite_number; forcing_model=forcing_model, month=month, experiment=experiment)
    ds_stack = cfSites.load_4d_fields(les_dir, vars_to_load)

    nt = length(DimensionalData.dims(ds_stack, DimensionalData.Ti))
    if max_timesteps > 0
        nt = min(nt, max_timesteps)
    end

    x_coords = collect(DimensionalData.dims(ds_stack, DimensionalData.X))
    z_coords = collect(DimensionalData.dims(ds_stack, DimensionalData.Z))
    dx_native = (x_coords[end] - x_coords[1]) / (length(x_coords) - 1)
    domain_h = x_coords[end] - x_coords[1]
    dz_native_profile = diff(z_coords)
    push!(dz_native_profile, dz_native_profile[end])

    metadata = (
        data_source = "cfSites",
        month = month,
        cfSite_number = cfSite_number,
        forcing_model = forcing_model,
        experiment = experiment,
        verbose = verbose,
    )

    spatial_info = (
        dx_native = Float32(dx_native),
        domain_h = Float32(domain_h),
        min_h_resolution = Float32(min_h_resolution),
        dz_native_profile = Float32.(dz_native_profile),
        seeds_h = (1,),
    )

    final_df = nothing
    processed_timesteps = 0
    started_at = time()

    for t_idx in 1:nt
        processed_timesteps += 1
        _progress_print("cfSites", processed_timesteps, nt, "site=$(cfSite_number) month=$(month) model=$(forcing_model) experiment=$(experiment) timestep=$(t_idx)", started_at)

        fine_fields = Dict{String, AbstractArray{Float32, 3}}()
        for (site_var, canonical_var) in CF_SITES_TRANSLATION
            fine_fields[canonical_var] = Float32.(ds_stack[Symbol(site_var)][DimensionalData.Ti(t_idx)])
        end

        metadata_t = (; metadata..., timestep = t_idx)

        df = DatasetBuilder.process_abstract_chunk(fine_fields, metadata_t, spatial_info)
        if size(df, 1) > 0
            _assert_finite_dataframe(df, "cfSites site=$(cfSite_number) month=$(month) model=$(forcing_model) experiment=$(experiment) timestep=$(t_idx)")
            if isnothing(final_df)
                final_df = df
            else
                append!(final_df, df)
            end
        end
    end

    _progress_finish()

    if isnothing(final_df)
        if verbose
            @info "No trainable cfSites rows found for site=$cfSite_number month=$month model=$forcing_model experiment=$experiment; no Arrow file written."
        end
        return
    end

    out_file = joinpath(output_dir, "cfsites_$(forcing_model)_$(experiment)_$(month)_$(cfSite_number).arrow")
    _assert_finite_dataframe(final_df, "cfSites final site=$(cfSite_number) month=$(month) model=$(forcing_model) experiment=$(experiment)")
    Arrow.write(out_file, final_df)
    println("Wrote cfSites case file $(out_file) with $(DataFrames.nrow(final_df)) rows.")
end