using Test: Test
using MLCondensateDistributions: MLCondensateDistributions as MLCD

# Helpers live in `utils/build_training_common.jl` (loaded by the package); do not
# partially `include` training scripts — that omits dependencies and breaks CI.

Test.@testset "build_training_data dimension helpers" begin
    A = reshape(collect(Float32, 1:2*3*4*5), 2, 3, 4, 5)

    for t_idx in 1:4
        raw = MLCD._slice_t_range_4d(A, 1:2, t_idx)
        Test.@test parent(raw) === A
        Test.@test size(raw, t_idx) == 2
    end

    Test.@test_throws ErrorException MLCD._slice_t_range_4d(A, 1:2, 0)
    Test.@test_throws ErrorException MLCD._slice_t_range_4d(A, 1:2, 5)

    dims_julia = (:z, :y, :x, :t)
    perm = MLCD._perm_to_txyz(dims_julia)
    Test.@test perm == (4, 3, 2, 1)

    raw = MLCD._slice_t_range_4d(A, 2:3, 4) # shape (2, 3, 4, 2)
    canonical = MLCD._reorder_to_txyz_view(raw, dims_julia) # shape (2, 4, 3, 2)
    Test.@test parent(canonical) === raw
    Test.@test size(canonical) == (2, 4, 3, 2)

    # Verify canonical[t, x, y, z] aligns with original[z, y, x, t]
    Test.@test canonical[1, 2, 3, 1] == A[1, 3, 2, 2]
    Test.@test canonical[2, 4, 1, 2] == A[2, 1, 4, 3]

    Test.@test_throws ErrorException MLCD._perm_to_txyz((:z, :y, :x, :q))
end
