#=

    Right now this will just help us figure out how to blast through the data quickly

=#


using Random: Random
using Statistics: Statistics
(nz, nx, ny, nt) = (480, 124, 124, 73)
GoogleLES_size = (nz, nx, ny, nt)
a_test = Random.rand(GoogleLES_size)


# ========================================================================== #
# Binary Reductions
# -------------------------------- #
"""
    This does binary reductions to take advantage of pooling for statistics (like split apply combine), to avoid repeated data processing at decreasing resolutions

    Every number an be written as odd_part * 2^k, so basically we just need unique seeds. By default, we'll use all odd numbers less than the domain


"""
function binary_coarsening(
    data::AbstractArray{FT, 4},
    min_h::FT,
    max_z::FT,
    dh::FT, # currently we only support fixed dh == dx == dy
    dz::Union{FT, AbstractVector{FT}},
    ;
    seeds_h::Ntuple{N, Int} = (),
    seeds_z::Ntuple{N, Int} = (),
    fail_on_hz_constraints::Bool = false,
    verbose::Bool = false,
) where {FT, N}

    nz, nx, ny, nt = size(data)

    isequal(nx, ny) || error("Currently only supports square horizontal domains")
    nh = nx



    # Horizontal reductions 
    if isempty(seeds_h)
        max_seed = iseven(nh) ? (nh - 1) : nh
        seeds_h = ntuple(i -> (2i-1), max_seed ÷ 2 + 1)
    else
        # enforce all seeds are unique and odd
        length(seeds_h) == length(unique(seeds_h)) || error("Seeds must be unique, got duplicates in seeds_h: $(seeds_h)")
        all(isodd, seeds_h) || error("All seeds must be odd, got even seed in seeds_h: $(seeds_h)")
    end


    # Vertical reductions
    if isempty(seeds_z)
        max_seed_z = iseven(nz) ? (nz - 1) : nz
        seeds_z = ntuple(i -> (2i-1), max_seed_z ÷ 2 + 1)
    else
        # enforce all seeds are unique and odd
        length(seeds_z) == length(unique(seeds_z)) || error("Seeds must be unique, got duplicates in seeds_z: $(seeds_z)")
        all(isodd, seeds_z) || error("All seeds must be odd, got even seed in seeds_z: $(seeds_z)")
    end




    # Handle grid too coarse for requestd max_z
    if (Statistics.minimum(dz) > max_z)
        if fail_on_hz_constraints
            error("dz must be less than max_z to have any coarsened levels, got minimum(dz) = $(Statistics.minimum(dz)) > max_z=$(max_z)")
        else            
            verbose && @warn "dz < max_z is required to have any coarsened levels in the vertical; got dz=$(dz), max_z=$(max_z). defaulting to native dz with no vertical coarsening"
            seeds_z = (1,)
        end
    end

    # Drop all levels violating max_z constraint right away to save computation
    if any(Base.Fix2(>, max_z), dz) # dz .> max_z without allocating 
        verbose && @info "Dropping vertical levels with dz > max_z=$(max_z) to save computation"
        seeds_z = filter(seed -> dz[seed] <= max_z, seeds_z)
        if length(dz) == nz # Allow for dz on centers or faces
            error("not implemented yet")
        else
            error("not implmented yet")
        end
        nz = error("not implemented yet")
    end


    # Handle domain not big enough for min_h
    if (dh * nh) < min_h
        if fail_on_hz_constraints
            error("domain size (dh * nh) = $(dh * nh) must be greater than min_h=$(min_h)")
        else
            verbose && @warn "domain size (dh * nh) = $(dh * nh) is less than min_h=$(min_h); defaulting to native dh with no horizontal coarsening"
            seeds_h = (nh,)
        end
    end





    # Each seed defines a binary reduction chain
    # We can save computation by skipping sizes below min_h and above max_z

    size_out = error("not implemented yet") # this will be the number of (res_h, res_z) pairs we emit, which determines how many rows we have per vertical level in the output

    stats_out = AbstractArray{FT, size_out}

    # we might be able to optimize this pattern somehow to still further reduce computation, but i havent figured out how yet
    @inbounds for seed_h in seeds_h, seed_z in seeds_z
        # calculate the minimum and maximum levels of coarsening for this seed [so as to adhere to max_z] (note if no coarsening in z, or full coarsening in h, we permit since that's checked above)
        
        res_h_here = dh * seed_h
        # sum in groups of seed_z, we cant do dh .* seed_z since dz might be nonuniform, so we have to check cumulative sum to find where we exceed max_z
        res_zs_here = error("not implemented yet")

        # calculate our initial stats


        # our pattern will be to coarsen horizontally, then go the distance in z 
        while res_h_here >= min_h

            while res_zs_here <= max_z
                # do the coarsening at this (res_h_here, res_zs_here) resolution defined by (seed_h, seed_z)
                # this will be a binary reduction of the original data, not incremental from the last coarsening, to maximize pooling benefits

                binary_pool_stats!(error("not implemented yet")) # this will be the pooling operation. I think it should also be able to reuse memory since it's a reduction idk.
                res_zs_here = error("not implemented yet")

                # write data into stats_out
                error("not implemented yet")
            end
        end


    end

    # filter outputs based on some filter 
    # pass, we are setting this up as a framework right now so we have no filtering yet

    return stats_out

end

# binary stat coarsening on a 2D array (2Nx2N) -> (NxN)
function binary_pool_mean!(μ::AbstractArray{FT, 2}, μ_out::AbstractArray{FT, 2}) where {FT}
    for j in axes(μ_out, 2), i in axes(μ_out, 1)
        @views μ_out[i, j] = Statistics.mean(μ[2i-1:2i, 2j-1:2j])
    end
    return μ_out
end
    
function binary_pool_variance!(σ²::AbstractArray{FT, 2}, σ²_out::AbstractArray{FT, 2}) where {FT}
    for j in axes(σ²_out, 2), i in axes(σ²_out, 1)
        error("not implemented yet, need to recall how to pool variances")
    end
    return σ²_out
end

function binary_pool_covariances!(covs::AbstractArray{FT, 2}, covs_out::AbstractArray{FT, 2}) where {FT}
    for j in axes(covs_out, 2), i in axes(covs_out, 1)
        error("not implemented yet, need to recall how to pool covariances")
    end
    return covs_out
end

# We need weighted versions for possible non-uniform dz

# ========================================================================== #

# Convolutions
function convolutional_coarsening(
    data::AbstractArray{FT, 4},
    min_h::FT,
    max_z::FT,
    dh::FT, # currently we only support fixed dh == dx == dy
    dz::Union{FT, AbstractVector{FT}},
    ;
    fail_on_hz_constraints::Bool = false,
    verbose::Bool = false,
) where {FT}

    # This will be a lot more computationally expensive than binary coarsening, but it will allow us to do coarsening at arbitrary resolutions instead of just powers of 2, so it might be worth it for the flexibility

    error("not implemented yet")

end