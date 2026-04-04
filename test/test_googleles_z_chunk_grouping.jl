using Test: Test
using Statistics: Statistics
using MLCondensateDistributions: MLCondensateDistributions as MLCD

# Chunk-aligned hull over the union of chunks touched by a set of spans (test-only helper).
function _union_chunk_hull_z(spans::Vector{UnitRange{Int}}, nz::Int, cz::Int)::UnitRange{Int}
    isempty(spans) && return 1:0
    cr = Tuple{Int, Int}[MLCD._z_native_span_to_storage_chunk_range(first(s), last(s), cz) for s in spans]
    cmin = Statistics.minimum(x[1] for x in cr)
    cmax = Statistics.maximum(x[2] for x in cr)
    return MLCD._z_storage_chunk_hull_to_native_layer_range(cmin, cmax, cz, nz)
end

Test.@testset "GoogleLES z-chunk axis vs _ARRAY_DIMENSIONS order" begin
    # Canonical write-up: docs/googleles_zarr_layout.md
    # Live stores use _ARRAY_DIMENSIONS = ["t","x","y","z"] while size/chunks follow the
    # permuted Julia order (z,y,x,t) → chunks (60,124,124,1); z must use chunks[1], not [4].
    raw = Any["t", "x", "y", "z"]
    chunks = (60, 124, 124, 1)
    julia_dim_names = (Symbol(raw[4]), Symbol(raw[3]), Symbol(raw[2]), Symbol(raw[1]))
    Test.@test julia_dim_names == (:z, :y, :x, :t)
    z_ax = findfirst(julia_dim_names) do sym
        Symbol(lowercase(String(sym))) === :z
    end
    Test.@test z_ax == 1
    Test.@test chunks[z_ax] == 60
end

Test.@testset "GoogleLES z-chunk span grouping" begin
    cz = 60
    nz = 480
    # Two slabs in the same first chunk → one group
    spans = [1:10, 40:55]
    g = MLCD._group_mask_spans_by_overlapping_z_chunks(spans, nz, cz)
    Test.@test length(g) == 1
    Test.@test g[1] == [1:10, 40:55]
    Test.@test _union_chunk_hull_z(g[1], nz, cz) == 1:60

    # Chunks 1 and 6 only → two groups
    spans2 = [1:10, 305:310]
    g2 = MLCD._group_mask_spans_by_overlapping_z_chunks(spans2, nz, cz)
    Test.@test length(g2) == 2
    Test.@test _union_chunk_hull_z(g2[1], nz, cz) == 1:60
    Test.@test _union_chunk_hull_z(g2[2], nz, cz) == 301:360

    # Touching chunk boundaries: span ending at 60 and span starting at 61 → separate chunks
    spans3 = [1:60, 61:70]
    g3 = MLCD._group_mask_spans_by_overlapping_z_chunks(spans3, nz, cz)
    Test.@test length(g3) == 2

    # Two mask spans in the same storage chunk (3)
    spans5 = [121:130, 150:160]
    g5 = MLCD._group_mask_spans_by_overlapping_z_chunks(spans5, nz, cz)
    Test.@test length(g5) == 1
    Test.@test _union_chunk_hull_z(g5[1], nz, cz) == 121:180
end

Test.@testset "_googleles_nonqc_span_groups single fused load" begin
    cz = 60
    nz = 480
    spans = [10:24, 80:94]
    sg_one = MLCD._googleles_nonqc_span_groups(spans, nz, cz, true, true)
    Test.@test sg_one == [spans]
    sg_part = MLCD._googleles_nonqc_span_groups(spans, nz, cz, true, false)
    Test.@test length(sg_part) == 2
    sg_nomerge = MLCD._googleles_nonqc_span_groups(spans, nz, cz, false, false)
    Test.@test sg_nomerge == [[10:24], [80:94]]
end
