# GPU retest of the Lagrange sweeps after the PR #8 review

The tiled Lagrange sweeps were benchmarked and checked on an H100 at `87ce55e` (before the
review). This file lists what the review changed and what has to be re-run on the GPU. Everything
below has been run on CPU only (KernelAbstractions CPU backend, 4 threads).

## Unchanged: the tiled path

Compared token by token with whitespace ignored (so the JuliaFormatter commit does not count),
`87ce55e` vs `668adb4`: identical

- kernels: `lagrange_tile_contig_kernel!`, `lagrange_tile_strided_kernel!`,
  `lagrange_tile_mirror_kernel!`, `mirror_halo_kernel!`, `lagrange_key_setup_kernel!`
- helpers: `_tile_stage!`, `_tile_stencil`, `_mirror_halo`, `_mirror_shift_bound`,
  `_sample_index(::Mirror, …)`, `_sample_line`, `_shift_key`
- launch parameters (workgroup `_TILE_WG = 256`, `_TILE_LINES = 32`, ndrange per kernel)
- `lagrange_weights`, `shift_split` (only the unused `lagrange_gather*` were removed beside them)

The earlier GPU numbers therefore still hold wherever a sweep is still tiled.

## What changed

| # | Change | Effect | Retest? |
|---|---|---|---|
| 1 | **Tile eligibility.** `W ≤ n ≤ 192` became `_tile_fits(T, nodes, W)`: `32·((nodes + W)·sizeof(T) + sizeof(Int)) ≤ 48 KiB`, with `nodes = n` (periodic) or `n + 2H` (mirror). | Float64 now tiles only up to n = 183 (W = 8), 187 (W = 4), 179 (W = 12), 167 (W = 24); up to 192 now uses the general kernel. Before, those sizes launched tiled with more than 48 KiB of shared memory. Float32 now tiles from n = 192 up to about 360 (W = 8): new, never run. Mirror tiles: Float64 `n + 2H ≤ 191 − W` (was `≤ 192`); Float32 gets more room. | **Yes:** Float32 tiled at n ≈ 200–360 (launch and correctness); Float64 boundary n = 183/184. The benchmark sizes (n = 24, 32, 64) are unaffected. |
| 2 | **Fallback kernel.** `:line` (used for ≥ 10⁵ lines) and `:point` are removed; every non-tiled sweep uses the former `:cached` kernel. It now indexes the same per-key table as the tiles (one entry per velocity index for x-sweeps, per position index for v-sweeps) instead of a per-line table, and the per-line setup kernel is gone. | Different code runs for non-tileable sweeps: Float64 n > 183, and mirror on x-direction 2 or 3. Different memory access in the general kernel. | **Yes:** general-kernel timing against the old `:line`/`:cached` at those sizes; tiled vs general bit-identity on GPU. |
| 3 | **Mirror wall symmetrisation.** The whole-array kernel for non-contiguous axes is removed; one per-line kernel handles every stride (generic `_line_linear_index`). | Mirror on x-direction ≥ 2 runs a different kernel; contiguous mirror (direction 1) keeps the same per-line logic. | **Yes:** mirror along direction 2 (and 3 in 3D), correctness on GPU. |
| 4 | **Buffer ownership.** Shift tables, mirror halos and the work buffer moved from global `Dict`s into `AdvectionPlan`, fetched per sweep through `_buffer!` under the plan's `ReentrantLock`. | Same arrays and shapes, one set per distribution instead of one global set; one host-side lock and dict lookup per sweep. | **Light:** timing at the benchmark sizes (no visible change expected against ~12 ms); GPU memory with several species; `release_advection_buffers!` frees tables and halos. |
| 5 | **Drivers.** `advectX!`/`advectV!` and their single-direction forms share one `_advect!`; same ping-pong between `f.data` and the work buffer. | Single-direction calls make at most one copy back (tiled: none, as before). | Covered by the existing tests. |
| 6 | **Boundary types from PlasmaCore** (≥ 0.1.1). `Periodic` and `Mirror` are PlasmaCore's empty marker types. | Passed to kernels as before; only the defining package changed. | Covered by compiling once on GPU. |

## Minimal GPU retest

1. **Correctness:** run `test/lagrange_sweep_test.jl` after `bslLD.use_cuda!()` (distributions are
   created on the active backend; the test compares tiled against general bit for bit). Add:
   - mirror along x-direction 2,
   - Float32 at n ≈ 256,
   - Float64 at n = 183 (tiled) and n = 184 (general).
2. **Benchmarks:**
   - 3D3V n = 32, W = 8, x and v: confirm ~12–13 ms per direction, unchanged.
   - The same with the general kernel forced, `bslLD._advect!(f, plan, method, bc, contexts, dirs; tiled = false)`,
     to measure what tiling buys over the remaining fallback.
   - One non-tileable case (Float64 n ≈ 256, or mirror along direction 2) against the old `:line`
     timing from `87ce55e`.
3. **Memory:** advect two species of the same size (sequentially and from separate tasks), then
   `release_advection_buffers!`, `GC.gc()`, `CUDA.reclaim()`; check `CUDA.memory_status()`.

## Possible issue, not yet checked

`_SHMEM_BYTES = 48 KiB` is CUDA's static shared-memory limit. Metal's threadgroup memory is
believed to be 32 KiB; if so, Float32 tiles of roughly n = 247–360 would fail to launch on Metal.
AMD (64 KiB LDS) is fine. If Metal matters, take the limit from the backend.
