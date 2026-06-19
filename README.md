# bslLD

Low-dimensional Backward Semi-Lagrangian (BSL) solver for testing numerical methods for plasma physics.

This package is a testing ground for numerical methods intended for [BSL6D](https://gitlab.mpcdf.mpg.de/bsl6d/bsl6d), providing 1D/2D configuration space with 2D velocity space at reduced computational cost.

## Features

- **Grid systems**: Cartesian and polar coordinate support
- **Distribution functions**: 1D1v, 1D2v, and 2D2v phase-space configurations
- **Advection**: Fourier-based and spline (Dierckx) interpolation, threaded over velocity slices
- **Field solvers**: electrostatic (Poisson), Darwin/hybrid-kinetic (DK), and vacuum Maxwell — see [Field Solvers](#field-solvers) below
- **Optional GPU acceleration**: KernelAbstractions-based kernels, switchable at runtime via a CUDA weak dependency

## Installation

```julia
] dev /path/to/bslLD
```

To enable GPU execution, load CUDA before using the package:

```julia
using CUDA, bslLD
bslLD.use_cuda!()
```

## Quick Start

```julia
using bslLD

# 1D2v phase space: x ∈ [0, 4π], vx,vy ∈ [-6, 6], background B along z
grid    = bslLD.Grid([0.0, -6.0, -6.0], [4π, 6.0, 6.0], [128, 32, 32], 1, 1.0, 3)
simTime = bslLD.SimulationTime(0.01, 10.0)

initFuncv(v) = exp(-v^2 / 2) / sqrt(2π)
f = bslLD.Distribution(grid, 1e-4; initFuncv = initFuncv)

while bslLD.continue_advection(simTime, true)
    rho = bslLD.compute_density(f, grid)
    sol = bslLD.solve_fields(bslLD.Moments(rho), grid, bslLD.PoissonSolver())
    bslLD.advectX!(f, grid, simTime)
    bslLD.advectV!(f, grid, simTime, sol.E)
    bslLD.advance!(simTime)
end
```

## Field Solvers

All solvers implement the `AbstractFieldSolver` interface via `solve_fields` / `solve_fields!`, which takes a `Moments` struct and returns a `FieldSolution` holding the updated `E` and `B` vector fields.

### Electrostatic solvers

| Solver | Equation solved | Required moments |
|--------|----------------|-----------------|
| `PoissonSolver(factor=1.0)` | $-\nabla^2\phi = \text{factor}\cdot\rho$, $\mathbf{E}=-\nabla\phi$ | `rho` |
| `AdiabaticSolver()` | $\mathbf{E} = -\nabla\rho$ (adiabatic electrons, no Poisson inversion) | `rho` |

Both are FFT-based and support arbitrary 1D/2D periodic domains. A 2/3-rule dealiasing filter is applied.

```julia
sol = bslLD.solve_fields(bslLD.Moments(rho), grid, bslLD.PoissonSolver())
Ex  = sol.E[1]   # ScalarField
```

### Darwin / hybrid-kinetic (EM, low-frequency)

The Darwin model retains electromagnetic effects while filtering out light waves by neglecting the transverse displacement current. Two variants are provided for kinetic-ion / fluid-electron plasmas, parametrised by $\beta_i$ (ion plasma beta) and $\mu = m_e/m_i$ (mass ratio).

| Solver | Description | Required moments |
|--------|-------------|-----------------|
| `EMSolverDKPol(β_i, μ)` | Darwin-Kinetic **with** polarisation-drift correction. Implicit $\mathbf{E}_\perp$ update via a $2\times2$ linear system per Fourier mode; Helmholtz solve for $E_\parallel$. Staggered Faraday for $\mathbf{B}$. | `rho`, `J` (⊥ current), `Pi_diff` (anisotropic pressure) |
| `EMSolverDKNoPol(β_i, μ)` | Darwin-Kinetic **without** polarisation drift. Fully spectral $3\times3$ Cramer solve per Fourier mode; simultaneous Faraday update. | `rho`, `J`, `Pi_diff` |

Both solvers use `solve_fields!(sol, moments, grid, solver, dt)` (in-place, stores result in `sol.Enew`):

```julia
solver  = bslLD.EMSolverDKPol(beta_i, mu)
sol     = bslLD.FieldSolution(E0, B0)
moments = bslLD.Moments(rho, J_perp, Pi_diff)

bslLD.solve_fields!(sol, moments, grid, solver, dt)
sol.E .= sol.Enew   # commit the update
```

The moments `J` and `Pi_diff` are computed from the kinetic distribution:

```julia
rho    = bslLD.compute_density(f, grid)
J_perp = bslLD.compute_current(f, grid, phase)
Pi     = bslLD.compute_momentum_tensor(f, grid, phase)
```

A cold-ion fluid alternative (`ColdIonFluid`) is also provided for linear benchmarking without a kinetic distribution.

### Vacuum Maxwell

| Solver | Description | Moments |
|--------|-------------|---------|
| `EMSolverVacuum(; c, ϵ0, μ0)` | Full Maxwell equations in vacuum, Crank–Nicolson time stepping (unconditionally stable, no CFL constraint on $\Delta t$). Supports 1D/2D domains with 2 or 3 field components. | ignored |

```julia
params = bslLD.VacuumMaxwellParams(c=1.0, ϵ0=1.0, μ0=1.0)
solver = bslLD.EMSolverVacuum(; c=params.c, ϵ0=params.ϵ0, μ0=params.μ0)
bslLD.solve_fields!(sol, moments, grid, solver, dt)
```

Diagnostic helpers: `bslLD.electromagnetic_energy(E, B; params)` and `bslLD.maxwell_constraints(E, B, grid)` ($\nabla\cdot\mathbf{E}$, $\nabla\cdot\mathbf{B}$).

## Physical Models

| Model | Configuration | Field solver | Example notebook |
|-------|--------------|-------------|-----------------|
| Vlasov–Poisson (electrostatic) | 1D1v or 1D2v | `PoissonSolver` | `landau_damping`, `ibw` |
| Ion Bernstein Waves | 1D2v + background $B_0$ | `PoissonSolver` (with gyration) | `ibw` |
| Darwin hybrid-kinetic | 1D2v + background $B_0$ | `EMSolverDKPol` / `EMSolverDKNoPol` | `em_cases` |
| E×B drift verification | 1D2v + external $E$, $B$ | none (imposed fields) | `exb_test` |
| Vacuum EM wave propagation | 1D or 2D | `EMSolverVacuum` | `solve_maxwell` |

## Project Structure

```
bslLD/
├── src/
│   ├── bslLD.jl                       # module entry point, backend management
│   ├── core/
│   │   ├── grid.jl                    # Grid (Cart/Polar), axes
│   │   ├── time.jl                    # SimulationTime, advance!, continue_advection
│   │   ├── indexing.jl                # multi-dim index helpers
│   │   └── fields.jl                  # ScalarField, VectorField, MatrixField
│   ├── kinetics/
│   │   ├── distribution.jl            # DistributionGrid, compute_density/current/Pi
│   │   ├── advectorCart.jl            # BSL advection (Cartesian, KernelAbstractions)
│   │   └── advectorPolar.jl           # BSL advection (polar coordinates)
│   ├── maxwell/
│   │   ├── differential_operators.jl  # spectral grad, curl, div
│   │   ├── spectral.jl                # FFT helpers, wavenumbers
│   │   ├── field_solver.jl            # AbstractFieldSolver, Moments, FieldSolution
│   │   ├── solvers_electrostatic.jl   # PoissonSolver, AdiabaticSolver
│   │   ├── solvers_vacuum.jl          # EMSolverVacuum, VacuumMaxwellParams
│   │   ├── solvers_hybrid.jl          # EMSolverDKPol, EMSolverDKNoPol
│   │   └── cold_plasma.jl             # ColdIonFluid for linear EM benchmarks
│   └── execution.jl                   # use_cpu!, use_cuda!, backend switching
├── examples/                          # Jupyter notebooks (landau_damping, ibw, em_cases, …)
└── test/                              # Unit tests
```

## Dependencies

**Required** (declared in `src/`):

| Package | Purpose |
|---------|---------|
| `FFTW.jl` | Fast Fourier transforms for spectral field solves and advection |
| `AbstractFFTs.jl` | GPU-portable FFT interface |
| `KernelAbstractions.jl` | CPU/GPU kernel abstraction for advection loops |
| `Adapt.jl` | Array transfer between CPU and GPU |
| `Dierckx.jl` | Spline interpolation for BSL back-tracing |
| `StaticArrays.jl` | Fixed-size arrays in hot loops |
| `ProgressMeter.jl` | Progress bars for long simulations |

**Optional / weak dependency:**

| Package | Purpose |
|---------|---------|
| `CUDA.jl` | GPU execution via KernelAbstractions CUDA backend |

**Examples only** (not required by the package itself):

| Package | Purpose |
|---------|---------|
| `CairoMakie.jl` | Plotting in example notebooks |
| `FFTW.jl` | Spectral diagnostics in notebooks (already a core dep) |
| `DSP.jl` | Window functions (Kaiser) for ω–k spectra |

## Backend Selection

The package defaults to CPU execution. Switch at runtime after loading CUDA:

```julia
using CUDA, bslLD
bslLD.use_cuda!()   # move execution to GPU
bslLD.use_cpu!()    # switch back to CPU
```

The CUDA extension is loaded automatically by Julia's extension mechanism when `CUDA` is in scope; no manual activation is needed.

## Related Projects

- **BSL6D** — full 6D Boltzmann solver this package feeds into

## Author

Mario Raeth
