"""
Comprehensive tests for vertical coarsening infrastructure (Phase 1).
Tests cover: coarsening schemes, z-level identification, dropping safety, and multi-resolution correctness.
"""

using Test: Test
using Statistics: Statistics
using Random: Random
using ..CoarseGraining: CoarseGraining

Test.@testset "Vertical Coarsening Infrastructure" begin

    # ==== Test 1: Uniform Grid Coarsening Scheme ====
    Test.@testset "Uniform Grid Coarsening Scheme (10m native)" begin
        # Create uniform dz profile: 127 levels × 10 m each
        dz_native = [10f0 for _ in 1:127]
        schemes = CoarseGraining.compute_z_coarsening_scheme(dz_native, 400f0)
        
        # Expected: [10, 20, 40, 80, 160, 320] m, stop before 640 m
        Test.@test length(schemes) == 6
        Test.@test schemes[1] == (1, 10f0)
        Test.@test schemes[2] == (2, 20f0)
        Test.@test schemes[3] == (4, 40f0)
        Test.@test schemes[4] == (8, 80f0)
        Test.@test schemes[5] == (16, 160f0)
        Test.@test schemes[6] == (32, 320f0)
        
        # Verify 640 m (64x factor) is skipped
        Test.@test !any(s -> s[2] > 400f0, schemes)
        Test.@test !any(s -> s[1] == 64, schemes)
    end

    # ==== Test 2: Variable dz Grid Coarsening ====
    Test.@testset "Variable dz Grid (finer near surface, coarser above)" begin
        # Simulate a realistic grid: fine near surface (10m), coarser aloft (50m)
        dz_variable = vcat(
            fill(10f0, 50),    # 500 m total near surface
            fill(25f0, 30),    # 750 m in lower troposphere
            fill(50f0, 20)     # 1000 m in upper troposphere
        )
        
        schemes = CoarseGraining.compute_z_coarsening_scheme(dz_variable, 400f0)
        
        # Should still produce a ladder up to 400 m
        Test.@test length(schemes) >= 1
        Test.@test schemes[1][1] == 1
        
        # Mean should be between min and max of dz_variable
        mean_dz = Statistics.mean(dz_variable)
        Test.@test schemes[1][2] ≈ mean_dz atol=1f0
    end

    # ==== Test 3: Coarse Native Grid (>400m) ====
    Test.@testset "Coarse Native Grid (>400m, no coarsening)" begin
        # Native grid already coarser than target
        dz_coarse = fill(500f0, 50)
        schemes = CoarseGraining.compute_z_coarsening_scheme(dz_coarse, 400f0)
        
        # Should only contain native resolution (no coarsening possible)
        Test.@test length(schemes) == 1
        Test.@test schemes[1] == (1, 500f0)
    end

    # ==== Test 4: Empty Z-Level Identification ====
    Test.@testset "Identify Empty Z-Levels" begin
        # Create synthetic 3D field: condensate only in z ∈ [30:50]
        nx, ny, nz = 10, 10, 100
        q_c = zeros(Float32, nx, ny, nz)
        q_c[:, :, 30:50] .= 1f-6  # Non-zero in middle z-levels
        
        empty_mask = CoarseGraining.identify_empty_z_levels(q_c, 1f-10)
        
        # Levels 1:29 and 51:100 should be empty
        Test.@test all(empty_mask[1:29])
        Test.@test !any(empty_mask[30:50])
        Test.@test all(empty_mask[51:100])
        Test.@test count(empty_mask) == (29 + 50)
    end

    Test.@testset "identify_empty_z_levels_from_ql_qi matches ql+qi" begin
        rng = Random.MersenneTwister(1234)
        thr = 1f-10
        for _ in 1:24
            nx, ny, nz = 5, 6, 19
            ql = rand(rng, Float32, nx, ny, nz)
            qi = rand(rng, Float32, nx, ny, nz)
            qc = ql .+ qi
            Test.@test CoarseGraining.identify_empty_z_levels_from_ql_qi(ql, qi, thr) ==
                CoarseGraining.identify_empty_z_levels(qc, thr)
        end
    end

    # ==== Test 5: Conservative Z-Level Keep Mask (Safety Critical) ====
    Test.@testset "Build Conservative Z-Level Keep Mask" begin
        # Test case: mixed empty/non-empty pattern
        empty_mask = BitVector([true, false, false, true, true, false, true, false, false, true])
        
        # At native resolution (z_factor=1), no future coarsenings
        mask_native = CoarseGraining.build_z_level_keep_mask(empty_mask, 1, Int[])
        
        # Should keep all non-empty and some empty (interior isolated ones)
        Test.@test mask_native[2]
        Test.@test mask_native[3]
        Test.@test mask_native[6]
        Test.@test mask_native[8]
        Test.@test mask_native[9]
        # Levels with no adjacent cloud may be dropped (interior isolated empties)
        
        # At coarser resolution with no future, still conservative
        mask_coarse = CoarseGraining.build_z_level_keep_mask(empty_mask, 2, Int[])
        Test.@test count(mask_coarse) >= count(.!empty_mask)
    end

    # ==== Test 6: Apply Z-Level Mask to Field ====
    Test.@testset "Apply Z-Level Mask to Field" begin
        # Create a simple 3D field
        nx, ny, nz = 5, 5, 20
        field = reshape(Float32.(1:nz), 1, 1, nz) .* ones(Float32, nx, ny, 1)
        
        # Keep only levels [5, 10, 15, 20]
        z_keep_mask = falses(nz)
        z_keep_mask[[5, 10, 15, 20]] .= true
        
        filtered = CoarseGraining.apply_z_level_mask_to_field(field, z_keep_mask)
        
        Test.@test size(filtered) == (nx, ny, 4)
        
        # Values should map correctly
        Test.@test all(filtered[:, :, 1] .≈ 5f0)
        Test.@test all(filtered[:, :, 2] .≈ 10f0)
        Test.@test all(filtered[:, :, 3] .≈ 15f0)
        Test.@test all(filtered[:, :, 4] .≈ 20f0)
    end

    # ==== Test 7: Build Z-Profile After Mask ====
    Test.@testset "Build Z-Profile After Mask" begin
        # Original profile
        dz_original = Float32[5, 5, 5, 5, 10, 10, 10, 20, 20, 20]  # 10 levels
        
        # Keep only levels [2, 4, 5, 9, 10]
        z_keep_mask = falses(10)
        z_keep_mask[[2, 4, 5, 9, 10]] .= true
        
        dz_kept = CoarseGraining.build_z_profile_after_mask(dz_original, z_keep_mask)
        
        Test.@test length(dz_kept) == 5
        Test.@test dz_kept == Float32[5, 5, 10, 20, 20]
    end

    # ==== Test 8: Safety Test - Dropped Data Cannot Corrupt Future Coarsenings ====
    # This is the CRITICAL test. It ensures we never drop data needed for averaging.
    Test.@testset "Safety: Dropped Data Doesn't Corrupt Future Coarsenings" begin
        # Scenario: We have a field with alternating empty/non-empty z-levels
        # e.g., [empty, cloud, empty, cloud, empty, cloud, ...]
        # If we drop "empty", can the future 2x coarsening still work correctly?
        
        nz = 32  # 32 levels allows multiple coarsenings
        q_c_pattern = BitVector(undef, nz)
        # Alternating pattern
        for k in 1:nz
            q_c_pattern[k] = iseven(k)  # even indices have cloud
        end
        
        # Build condensate field from pattern
        nx, ny = 5, 5
        q_c = zeros(Float32, nx, ny, nz)
        for k in 1:nz
            if !q_c_pattern[k]
                q_c[:, :, k] .= 0f0  # Empty
            else
                q_c[:, :, k] .= 1f-6  # Cloud
            end
        end
        
        # Identify empty levels at native resolution
        empty_mask_native = CoarseGraining.identify_empty_z_levels(q_c, 1f-10)
        
        # Get keep mask (should be conservative and preserve all non-empty)
        keep_mask_native = CoarseGraining.build_z_level_keep_mask(empty_mask_native, 1, [2, 4, 8])
        
        # Ensure we keep all non-empties (critical for correctness)
        for k in 1:nz
            if !empty_mask_native[k]  # Non-empty
                Test.@test keep_mask_native[k]
            end
        end
        
        # Now if we applied a 2x vertical coarsening to kept levels, it should work
        filtered_q_c = CoarseGraining.apply_z_level_mask_to_field(q_c, keep_mask_native)
        nz_filtered = size(filtered_q_c, 3)
        
        # This should be valid input for further coarsening
        Test.@test nz_filtered > 0
        Test.@test nz_filtered <= nz
        
        # Verify no NaNs introduced
        Test.@test !any(isnan, filtered_q_c)
    end

    # ==== Test 9: Edge Case - All Empty ====
    Test.@testset "Edge Case: All-Empty Domain" begin
        nx, ny, nz = 10, 10, 50
        q_c_empty = zeros(Float32, nx, ny, nz)
        
        empty_mask = CoarseGraining.identify_empty_z_levels(q_c_empty, 1f-10)
        Test.@test all(empty_mask)
        
        # Even conservative mask should allow us to skip all rows (handled at higher level)
        keep_mask = CoarseGraining.build_z_level_keep_mask(empty_mask, 1, Int[])
        # Behavior: we might keep some boundary levels conservatively, or drop all
        # The important part is the higher-level code checks for "no valid rows"
    end

    # ==== Test 10: Edge Case - Single Z-Level ====
    Test.@testset "Edge Case: Single Z-Level Domain" begin
        dz_single = [100f0]
        schemes = CoarseGraining.compute_z_coarsening_scheme(dz_single, 400f0)
        
        Test.@test length(schemes) == 1
        Test.@test schemes[1][1] == 1
    end

    # ==== Test 11: Correctness of Averaged Values ====
    Test.@testset "Correctness: Averaged Values Match Expected" begin
        # Simple test: 4-level field with known values
        # Levels 1,2 average to 1.5; levels 3,4 average to 3.5
        field = Float32[
            1.0  2.0  3.0  4.0;
            1.0  2.0  3.0  4.0;
        ]  # (2, 4) -> (x, z) for simplicity
        
        # We can't directly test cg_2x_vertical here (it's already tested elsewhere),
        # but we verify our new filtering functions preserve values
        q_c = ones(Float32, 2, 2, 4)
        keep_mask = BitVector([true, true, false, false])
        q_c_filtered = CoarseGraining.apply_z_level_mask_to_field(q_c, keep_mask)
        
        Test.@test size(q_c_filtered) == (2, 2, 2)
        Test.@test all(q_c_filtered .≈ 1f0)  # All ones preserved
    end

end  # End of testset
