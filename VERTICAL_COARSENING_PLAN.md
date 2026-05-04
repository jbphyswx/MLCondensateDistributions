# Vertical Coarsening Implementation Plan

## Current State
- **Horizontal coarsening**: ✅ Implemented. Starts at `min_dh` (1 km), does binary 2x reduction steps upward until tiny or no condensate.
- **Vertical coarsening**: ❌ NOT implemented. Code loads full vertical grid, reads `dz_profile`, stores as metadata, but never coarsens in z.
- **Sparsity dropping**: Partial. Currently drops empty (x,y) cells; does NOT drop empty z-levels.

## Goals
1. Implement vertical binary coarsening from native dz up to 400 m max.
2. For each horizontal resolution, emit multiple vertical resolutions.
3. Drop z-levels where the entire (x,y) plane has no condensate (respecting z-reduction structure, meaning that even if the entire plane is empty, if it would contribute in a future reduction, it should not be dropped)
4. Write comprehensive tests.
5. Ensure `resolution_z` metadata correctly reflects the coarsened grid.

## Design Decisions

### Vertical Resolution Ladder
- **Start**: Finest available grid (native dz or first usable coarsening if native < ~10 m threshold).
- **Stop**: When the next coarsening would exceed 400 m.
- **Step**: Binary (2x) coarsening.
- **Example**: Native dz = 10 m → [10, 20, 40, 80, 160, 320] m. Skip 640 m (exceeds 400 m limit).

### Z-Level Dropping Strategy
When z-levels are dropped:
- A z-level is "empty" if `max(q_c(:,:,z)) == 0` after horizontal coarsening but BEFORE vertical coarsening.
- After identifying empty z-levels, construct a mapping of which fine z-indices map to kept z-levels.
- Apply this mapping to all fields and update `dz_profile` accordingly.
- The dropped z-levels must be consistent across ALL fields (q_t, u, v, w, etc.) and moments.
- We must not drop any data in x y or z that would contribute to a future reduced calcultion. This is a critical rule (we should have extensive tests that enforce this on synthetic data so we know we never violate this. failure will lead to skewed statistics)

### Output Structure
Each (horizontal_resolution, vertical_resolution) pair produces a separate set of rows in Arrow:
- Rows differ in:
  - `resolution_h` (e.g., 1000, 2000, 4000 m)
  - `resolution_z` (e.g., 10, 20, 40 m)
  - Actual column values (different coarsenings)
- DataFrame concatenation is unchanged (vcat at the end).

## Implementation Phases

### Phase 1: Add Vertical Coarsening Infrastructure   
**File**: `utils/coarse_graining.jl`  
**Changes**:
- Add `function coarsen_vertical_grid(z_coords::Vector, target_dz_max::Float32)` → Returns coarsening scheme (indices to average).
- Verify `cg_2x_vertical` is exported and working correctly.
- Add `identify_empty_z_levels(q_c_3d::BitArray{3})` → Returns set of z-indices with all-dry (x,y) plane.

**Tests**:
- Test coarsenings from native grids (10m, 25m, 50m, 100m native).
- Verify 400 m boundary is respected.
- Test that odd z-count grids are handled correctly (topmost level dropped on odd count).

### Phase 2: Modify DatasetBuilder 
**File**: `utils/dataset_builder.jl`  
**Function**: `process_abstract_chunk`

**Changes**:
1. After horizontal coarsening setup, compute vertical coarsening schedule:
   ```julia
   v_scheme = compute_z_coarsening_scheme(spatial_info[:dz_native_profile], max_dz)
   # Returns: [(z_reduction_level, multiplier, effective_dz), ...]
   ```

2. Wrap main processing loop with vertical coarsening loop:
   ```julia
   for (v_level, z_factor, effective_dz) in v_scheme
       # Identify empty z-levels at this coarsening
       z_keep_mask = identify_empty_z_levels(...)
       # Apply z-reduction to all fields
       # Emit rows
   end
   ```

3. For each vertical level:
   - Apply vertical coarsening via `cg_2x_vertical` (chained if z_factor > 2).
   - Drop empty z-levels.
   - Update `dz_profile` to only keep indices from `z_keep_mask`.
   - Gather results using updated profile.

4. Store updated `resolution_z` for each row based on effective_dz.

**Edge Cases**:
- All z-levels empty → skip that vertical resolution (no rows).
- Native dz already > 400 m → single resolution (no coarsening).
- Variable dz grids → sum consecutive intervals correctly during coarsening.

### Phase 3: Add Tests
**File**: `test/test_vertical_coarsening.jl` (new file)

**Test scenarios**:
1. **Uniform grid coarsening** (10 m native):
   - Verify coarsening produces [10, 20, 40, 80, 160, 320] and stops.
   - Check that averaged values are exact (Δ < 1e-6).

2. **Variable dz grid** (e.g., finer near surface, coarser above):
   - Load cfSites or synthetic grid.
   - Verify dz_profile is updated correctly during coarsening.

3. **Z-level dropping**:
   - Create synthetic (x, y, z) field with condensate only in z ∈ [10:50].
   - After dropping, verify:
     - Kept z-levels have valid dz_profile.
     - No rows for dropped levels.
     - Indices map correctly.

4. **Multi-resolution output**:
   - Process one full GoogleLES case.
   - Verify Arrow output has rows for all (resolution_h, resolution_z) pairs.
   - Spot-check that resolution_z in metadata matches actual dz_profile lengths.

5. **Edge cases**:
   - All-dry case (no condensate anywhere) → no rows, empty Arrow.
   - Single z-level → can't coarsen further, single Z-resolution output.
   - Very coarse native grid (dz > 400 m) → only native resolution.

### Phase 4: Integration and Validation
**Deliverables**:
1. Refactor GoogleLES.build_tabular() to call updated DatasetBuilder.
2. Run small batch (5 sites, 1 month) to validate output.
3. Check Arrow schema matches dataset_spec.json (all 33 columns present).
4. Verify no NaNs or infinities in coarsened data.
5. Benchmark: compare runtime (should be slower due to multiple z-coarsenings, but acceptable).

## Questions / Decisions Needed

### Q1: How aggressive to drop z-levels?
**Current plan**: Drop if entire (x,y) plane is 0 at the current horizontal+vertical resolution.
**Alternative**: Drop only at native resolution, then track "mask history" through coarsenings.
**Recommendation**: Current plan is simpler; implement that first, revisit if memory is an issue.

### Q2: Should we emit multiple vertical resolutions in ONE Arrow file or separate files?
**Current plan**: One Arrow file with `resolution_z` as metadata column; rows distinguish via (resolution_h, resolution_z) pairs.
**Benefit**: Single file per (site, month). Cleaner on-disk layout.
**Risk**: Larger Arrow files. Mitigate by filtering at load time by (resolution_h, resolution_z).
**Decision**: One file per (site, month); client filters by resolution pairs as needed.

### Q3: Benchmark target for regeneration?
**Current**: ~195s per case (118s load + 77s processing).
**With vertical coarsening**: ~6 vertical levels × 3 horizontal levels mean ~18 combinations per case.
**Estimate**: 18× more timestep processing (each z-reduction recomputes moments). Cache load same.
**Budget**: Aim for < 10 min per case. If > 15 min, consider skipping some (resolution_h, resolution_z) pairs.

## Implementation Order
1. Test vertical coarsening functions (Phase 1).
2. Modify DatasetBuilder (Phase 2).
3. Write integration tests (Phase 3).
4. Run pilot batch (Phase 4).
5. Full regeneration (500 sites × 4 months).

## Estimated Effort
- **Phase 1 (infrastructure)**: 2 hours.
- **Phase 2 (DatasetBuilder refactor)**: 4 hours (nested loops, bookkeeping).
- **Phase 3 (tests)**: 3 hours (multiple scenarios).
- **Phase 4 (validation)**: 2 hours (running batches, checking outputs).
- **Total**: ~11 hours development + benchmarking time.
