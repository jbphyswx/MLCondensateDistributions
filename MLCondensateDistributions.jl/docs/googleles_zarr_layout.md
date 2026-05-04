# GoogleLES Zarr layout: `size`, `chunks`, and `_ARRAY_DIMENSIONS`

This note is the **single canonical reference** for how GoogleLES 4D Zarr variables are laid out in Julia. Read it before interpreting profiling numbers, chunk sizes, or any code that indexes `size(var)` / `chunks` / `_ARRAY_DIMENSIONS`.

Related code: `utils/build_training_common.jl` (`_load_googleles_cache`, `_load_googleles_timestep_fields!`, `_googleles_storage_z_chunk_size`). Regression test: `test/test_googleles_z_chunk_grouping.jl`.

## What this is *not* (column- vs row-major)

Two different ideas often get mixed up:

1. **Memory layout:** Julia stores dense arrays in **column-major** order (first index varies fastest in memory). Zarr chunks use **`"order": "C"`** in `.zarray` (C / row-major **within** the chunk’s linear layout). Neither of those is what causes the `chunks[i]` vs `_ARRAY_DIMENSIONS[i]` confusion.

2. **Axis order in the API:** For these stores, **`_ARRAY_DIMENSIONS` matches the on-disk Zarr `shape` array** (xarray convention: outer → inner as listed). **`Zarr.jl` then exposes `size(var)` and `chunks` in the reverse order of that `shape`** for 4D variables checked here. So you must not assume `size[i]` lines up with `_ARRAY_DIMENSIONS[i]`.

The recurring bug is using **`_ARRAY_DIMENSIONS` list position** as the index into **`chunks`** or **`size`** in Julia without applying the same permutation the loaders use.

## Verified on live Zarr (`q_c`, site 320, month 1, `amip`)

From `…/data.zarr/q_c/.zarray` (public GCS):

- `"shape": [73, 124, 124, 480]` — file axis order is **`(t, x, y, z)`** lengths.
- `"chunks": [1, 124, 124, 60]` — time chunked by **1**, height by **60** on the **z** axis in file order (last index).
- `q_c/.zattrs`: `"_ARRAY_DIMENSIONS": ["t", "x", "y", "z"]` — aligns with that `shape` order.

In Julia (`Zarr.zopen`), the same array reports **`size ≈ (480, 124, 124, 73)`** and **`chunks ≈ (60, 124, 124, 1)`**: axis order is **reversed** relative to the JSON `shape` / `_ARRAY_DIMENSIONS` list. Hence **`size[1] == 480` is z** and **`size[4] == 73` is t**, and **`chunks[1] == 60` is the z-chunk**, not `chunks[4]`.

## Facts for typical AMIP 4D fields (e.g. `q_c`)

After opening the store in Julia (`Zarr.jl`):

- `size(var) == (480, 124, 124, 73)` (example dimensions; horizontal sizes can differ by case).
- `chunks` is a 4-tuple in the **same index order as `size`**, e.g. `(60, 124, 124, 1)`.
- The Zarr attribute `_ARRAY_DIMENSIONS` is a **list of names** in xarray/Swirl-LM convention. For these runs it is typically:

  `["t", "x", "y", "z"]`

### Map Julia axis index → physics → extent → chunk (for that layout)

| Julia index `i` | Name on axis `i` | Typical extent | Typical `chunks[i]` |
|-----------------|------------------|----------------|---------------------|
| 1 | **z** (vertical) | 480 | 60 |
| 2 | **y** | 124 | 124 |
| 3 | **x** | 124 | 124 |
| 4 | **t** (time) | 73 | 1 |

So: **480 is z**, **73 is t**. The z-chunk size along storage is **`chunks[1]`** (here 60), not the last chunk element.

### Why `_ARRAY_DIMENSIONS` index ≠ `size` index

The metadata list order is **`[t, x, y, z]`** (index 1…4 in the JSON array).

The Julia `size` / `chunks` tuple order is **`[z, y, x, t]`** (index 1…4 in Julia).

So:

- `_ARRAY_DIMENSIONS[1] == "t"` corresponds to **Julia dimension 4** (last in `size`).
- `_ARRAY_DIMENSIONS[4] == "z"` corresponds to **Julia dimension 1** (first in `size`).

**Wrong:** `findfirst(==("z"), _ARRAY_DIMENSIONS)` → 4, then read `chunks[4]` → that is the **time** chunk (often 1), not the z chunk.

**Right:** Build the same `julia_dim_names` tuple the loaders use, find which **Julia** axis is `:z`, then use that index into `chunks`.

### Permutation used in this package (4D only)

For `metadata = var.attrs["_ARRAY_DIMENSIONS"]` (length 4), storage axis `i` is named:

```julia
julia_dim_names = (
    Symbol(metadata[4]),
    Symbol(metadata[3]),
    Symbol(metadata[2]),
    Symbol(metadata[1]),
)
```

So for `["t","x","y","z"]`, `julia_dim_names == (:z, :y, :x, :t)`.

Slice along time **before** permuting to canonical `(t,x,y,z)` using `t_axis_idx = _find_dim_idx(julia_dim_names, :t)` (here 4). See `_slice_t_range_4d` and `_reorder_to_txyz_view` in `build_training_common.jl`.

## Canonical order inside the pipeline

After `PermutedDimsArray`, 4D fields are exposed in **`(t, x, y, z)`** order for the rest of `build_tabular`. Native vertical index `k` in mask logic refers to the **z** dimension of that canonical view (or equivalently the storage **z** axis before permutation).

## When you change this code

- Any new helper that reads `chunks`, `size`, or `_ARRAY_DIMENSIONS` must use **`julia_dim_names`** (or the same permutation), not raw metadata list indices.
- Profiling scripts should state **which axis** they slice (Julia index and physics name), and cite this doc.
