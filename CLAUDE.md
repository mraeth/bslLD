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

Grid, field, distribution, spectral-operator, execution/GPU, and field-solver-interface
primitives have been factored out into the separate `PlasmaCore.jl` package (imported via
`using PlasmaCore: ...` at the top of `src/bslLD.jl`), which `bslLD` depends on via a remote
git URL (see the outer `kinetics/CLAUDE.md`, not something to vendor back in without being
asked). That package provides, among others: `Grid{T,XT,VT,MT,ID}` (`Cart`/`Polar`),
`ScalarField`/`VectorField`/`MatrixField`, `SimulationTime`/`advance!`/`continue_advection`,
`DistributionGrid{DT,NX,NV,NXNV,ID,AT}` (and the `DistributionGrid1d1v`/`1d2v`/`2d2v`
aliases), the spectral operators (`grad`/`div`/`curl`/`differentiate`), the
`AbstractFieldSolver`/`Moments`/`FieldSolution` interface, and the backend/GPU utilities
(`use_cuda!`, `backend_array`, `backend_copy`, etc.) plus its CUDA extension.

`bslLD` itself now holds only the physics built on top of those primitives:

### Kinetics (`src/kinetics/`)
- `species.jl` — `Species{PDT,DG}` pairs a mass/charge with a `DistributionGrid`;
  `thermal_velocity`, `electric_acceleration_scale`, `gyro_frequency`.
- `initialization.jl` — `Distribution(...)` constructors building a `Species` from grid +
  initial-condition functions.
- `advectorCart.jl` / `advectorPolar.jl` — `advectX!`/`advectV!` implement BSL back-tracing
  over a `Species`, dispatching on `CartGrid`/`PolarGrid`. Cartesian uses FFT-based
  (Fourier-mode) interpolation; Polar uses Dierckx splines and is type-constrained to
  `DistributionGrid1d2v{DT,Polar}` (1D config × 2D polar velocity only).
- `moments.jl` — `compute_density`, `compute_current`, `compute_momentum_tensor` reducing a
  `Species`'s distribution over velocity dimensions (Cart and Polar variants).
- `moment_response.jl` — `predict_midpoint_current`/`predict_stage_current` (linear-response
  current predictors for the EM midpoint solvers) and `compute_density_current`.
- `exbBracketCart.jl` — `exb_bracket!`/`exb_euler!`, the E×B Poisson-bracket advection used
  by the hybrid Darwin-Kinetic solvers.

### Maxwell (`src/maxwell/`)
- `solvers_electrostatic.jl` — `PoissonSolver`, `AdiabaticSolver` (both FFT-based, periodic,
  with 2/3-rule dealiasing).
- `solvers_vacuum.jl` — `EMSolverVacuum` with Crank–Nicolson time-stepping; `electromagnetic_energy` and `maxwell_constraints` diagnostics.
- `solvers_hybrid.jl` — `EMSolverDKPol` (Darwin-Kinetic with polarisation drift, 2×2 per-mode
  linear solve + Helmholtz) and `EMSolverDKNoPol` (fully spectral 3×3 Cramer solve), plus
  its midpoint/θ-method time-integration variants (`solve_fields_midpoint!`,
  `solve_fields_damped_midpoint!`) and `apply_faraday!`.
- `cold_plasma.jl` — `ColdIonFluid` for linear benchmarks without a kinetic distribution.

### Sources (`src/sources.jl`)
- `KappaTContext`/`add_kappaT!` — temperature-gradient-driven source terms.

## Key Conventions

- All field types (`ScalarField`, `VectorField`, `MatrixField`) are immutable structs wrapping arrays; mutate via `.data` or in-place kernel calls, not by replacing the struct.
- `solve_fields` (out-of-place) is for electrostatic solvers; `solve_fields!` (in-place, updates `sol.Enew`) is for EM solvers — commit with `sol.E .= sol.Enew` after the call.
- Phase-space configurations (`Cart`/`Polar`) are encoded as type parameters throughout, enabling dispatch without runtime branching.
- Tests live in `test/` (one file per subsystem: `basics_test.jl`, `advection_test.jl`, `index_test.jl`, `differential_operators_test.jl`, `solvers_test.jl`, `hybrid_test.jl`, `maxwell_test.jl`, `gpu_test.jl`).
- Example notebooks are in `examples/` with their own `Project.toml`; run them with a kernel that has `bslLD` on the load path.
