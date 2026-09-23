# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About

`bslLD` is a Julia package implementing a low-dimensional Backward Semi-Lagrangian (BSL) solver for plasma physics. It is a testing ground for numerical methods feeding into the full 6D solver [BSL6D](https://gitlab.mpcdf.mpg.de/bsl6d/bsl6d). It supports 1D/2D configuration space × 2D velocity space phase-space configurations and multiple field solver regimes.


## File reading policy
- Only read files that are directly referenced in the task, imports/requires 
  of the file(s) you're editing, or files I explicitly mention.
- Do not proactively explore the codebase, read "related" files, or scan 
  directories "just in case."
- If you believe you need another file for context, tell me which file and 
  why before reading it — don't just read it.
- Exception: you may run quick read-only commands (grep, ls, file search) 
  to locate something I've referenced by name, rather than browsing broadly.

## Commands

**Run all tests:**
```julia
julia --project=. -e 'using Pkg; Pkg.test()'
```
or from inside a Julia REPL with the project activated:
```julia
] test
```

**Run a single test file** (e.g., `test/solvers_test.jl`):
```julia
julia --project=. -e 'using bslLD; include("test/solvers_test.jl")'
```

**Check formatting** (CI enforces this):
```julia
julia -e 'using Pkg; Pkg.add("JuliaFormatter"); using JuliaFormatter; format(".", check=true)'
```

**Apply formatting:**
```julia
julia -e 'using JuliaFormatter; format(".")'
```

**Build docs:**
```julia
julia --project=docs docs/make.jl
```

## Architecture

The package is structured around three layers:

### Core (`src/core/`)
- `grid.jl` — `Grid{T,XT,VT,MT,ID}` parametrised by coordinate type `ID` (`Cart` or `Polar`). Constructed via `Grid(etaMin, etaMax, N, nx, b0, Bdir; type=Cart)`. Stores typed tuples of `StepRange` axes for x and v dimensions.
- `fields.jl` — `ScalarField`, `VectorField`, `MatrixField` wrappers around arrays. All field arithmetic (`+`, `-`, `*`, matrix–vector products) is defined here. Fields are always routed through `bslLD.backend_array()` at construction so the same code runs on CPU or GPU.
- `time.jl` — `SimulationTime` and `advance!` / `continue_advection`.
- `indexing.jl` — multi-dimensional index helpers used in kernels.

### Kinetics (`src/kinetics/`)
- `distribution.jl` — `DistributionGrid{DT,NX,NV,NXNV,ID,AT}` holds the phase-space distribution function as a multi-dimensional array. Type aliases `DistributionGrid1d1v`, `DistributionGrid1d2v`, `DistributionGrid2d2v` select specific configurations. `compute_density`, `compute_current`, `compute_momentum_tensor` reduce over velocity dimensions.
- `advectorCart.jl` / `advectorPolar.jl` — `advectX!` and `advectV!` implement BSL back-tracing using `KernelAbstractions` kernels (parallelises over velocity slices on CPU via threads, or over all cells on GPU). Interpolation uses Fourier modes or Dierckx splines.

### Maxwell (`src/maxwell/`)
- `spectral_operators.jl` — FFT-based spectral gradient, curl, divergence; wavenumber arrays.
- `field_solver.jl` — defines `AbstractFieldSolver`, `Moments{rho, J, Pi_diff}`, and `FieldSolution{E, B, Enew}`.
- `solvers_electrostatic.jl` — `PoissonSolver`, `AdiabaticSolver` (both FFT-based, periodic, with 2/3-rule dealiasing).
- `solvers_vacuum.jl` — `EMSolverVacuum` with Crank–Nicolson time-stepping; `electromagnetic_energy` and `maxwell_constraints` diagnostics.
- `solvers_hybrid.jl` — `EMSolverDKPol` (Darwin-Kinetic with polarisation drift, 2×2 per-mode linear solve + Helmholtz) and `EMSolverDKNoPol` (fully spectral 3×3 Cramer solve).
- `cold_plasma.jl` — `ColdIonFluid` for linear benchmarks without a kinetic distribution.

### Backend / GPU (`src/execution.jl`, `ext/bslLDCUDAExt.jl`)
- Default backend is `KernelAbstractions.CPU()`. Call `bslLD.use_cuda!()` after `using CUDA` to switch to GPU.
- `bslLD.backend_array(x)` moves an array to the active backend. `bslLD.backend_copy` deep-copies any field or distribution to the current backend.
- The CUDA extension wires up `CUDA_AVAILABLE_HOOK` and `SET_CUDA_EXECUTION_SPACE_HOOK` via weak-dependency extension loading.

## Key Conventions

- All field types (`ScalarField`, `VectorField`, `MatrixField`) are immutable structs wrapping arrays; mutate via `.data` or in-place kernel calls, not by replacing the struct.
- `solve_fields` (out-of-place) is for electrostatic solvers; `solve_fields!` (in-place, updates `sol.Enew`) is for EM solvers — commit with `sol.E .= sol.Enew` after the call.
- Phase-space configurations (`Cart`/`Polar`) are encoded as type parameters throughout, enabling dispatch without runtime branching.
- Tests live in `test/` (one file per subsystem: `basics_test.jl`, `advection_test.jl`, `index_test.jl`, `differential_operators_test.jl`, `solvers_test.jl`, `hybrid_test.jl`, `maxwell_test.jl`, `gpu_test.jl`).
- Example notebooks are in `examples/` with their own `Project.toml`; run them with a kernel that has `bslLD` on the load path.
