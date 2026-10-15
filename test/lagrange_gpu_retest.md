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
| 4 | **Buffer ownership.** Shift tables, mirror halos and the work buffer moved from global `Dict`s into `AdvectionPlan`, fetched per sweep through `_buffer!` under the global `_plan_cache_lock`. | Same arrays and shapes, one set per distribution instead of one global set; one host-side lock (global `_plan_cache_lock`) and dict lookup per sweep. | **Light:** timing at the benchmark sizes (no visible change expected against ~12 ms); GPU memory with several species; `release_advection_buffers!` frees tables and halos. |
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

## Follow-up after the GPU retest: halo-based mirror path in the general kernel

The GPU retest (H100, 2026-10-15) found the Mirror branch of the general kernel 9–19× slower
than its Periodic branch (32⁶: x1 683 vs 74 ms, x2 475 vs 30 ms, x3 481 vs 25 ms). That cost
predates the review for x2/x3, but by removing `:line` the review moved a non-tileable
contiguous mirror (2D2V x = 256) from 69 ms to 360 ms.

Change: every mirror sweep, tiled or general, first fills the reflected halo `hb[line, 2H]`
with `mirror_halo_kernel!`, which now takes the line stride, so it works for any x-direction.
The general kernel reads out-of-range taps from `hb[L, …]` (`L` from the existing index
formula) instead of rebuilding the line's multi-index and locating the velocity-flipped
partner per tap; `_sample_index(::Mirror, …)` is removed. The halo holds the same nodes
(`min(1+h, n)` / `max(n−h, 1)` of the partner line) and the summation order is unchanged.

CPU checks:
- tiled vs general mirror still bit-identical;
- the general mirror sweep along x1 and x2 (2D2V, shifts up to ~5 cells, W = 4, 8) equals a
  plain-loop reference of the node fold with velocity flip, bit for bit;
- a deliberate off-by-one in the halo read fails those tests (6 failures);
- CPU timing does not show the GPU effect: general Mirror ≈ Periodic before and after
  (≈ 3.7 ms, 2D2V 32⁴, x2).

Re-time on the GPU (same method as above):
- mirror 32⁶, single direction x1 / x2 / x3 (default path), against 62.0 / 515.7 / 520.6 ms;
- `t29` kernels alone: general Mirror sweep against general Periodic on x1/x2/x3;
- 2D2V x = 256 advectX x1 mirror (non-tileable), against 359.7 ms now and 69.1 ms with old `:line`;
- memory: the halo buffer is now also allocated for non-tiled mirror sweeps (x2/x3), one
  `nl × 2H` array per (H, line count).

### Result on the GPU (H100 PCIe, `8c7f9d2`)

| case (ms, median) | `87ce55e` | `668adb4` | `8c7f9d2` |
|---|---|---|---|
| 2D2V x = 256 mirror, advectX x1 (not tileable) | 69.1 | 359.7 | 44.9 |
| 2D2V x = 256 mirror, advectX x2 | 344.7 | 324.9 | 27.8 |
| 32⁶ mirror advectX x1 (tiled) | 62.4 | 62.0 | 62.5 |
| 32⁶ mirror advectX x2 / x3 | 526 / 525 | 516 / 521 | 75.1 / 69.3 |
| 32⁶ periodic tiled x / v | 12.35 / 13.46 | 12.35 / 13.46 | 12.35 / 13.43 |

- General Mirror sweep against general Periodic, kernels alone: 0.95–1.8×, previously 9–19×.
- The review regression is removed: the non-tileable contiguous mirror is now 1.5× faster than the old `:line`.
- Correctness: the 26 extra cases are unchanged and bit-identical between tiled and general. The plain-loop reference test differs from the host loop by 1 ulp at 24–28 % of points on CUDA, because the GPU fuses multiply-adds (FMA). It now compares with `rtol = 1e-13`, which still fails on a one-node halo shift.
- Memory: unchanged.
- Still open: Metal's threadgroup limit. Mirror on x2/x3 costs about 6× a tiled periodic sweep, because it has no tiled path.
