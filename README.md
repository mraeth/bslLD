# bslLD

Low-dimensional Backward Semi-Lagrangian (BSL) solver for testing numerical methods for plasma physics.

This package is a testing ground for numerical methods intended for [BSL6D](https://gitlab.mpcdf.mpg.de/bsl6d/bsl6d), providing 1D/2D configuration space with 2D velocity space at reduced computational cost.

## Features

- **Grid systems**: Cartesian and polar coordinate support
- **Distribution functions**: 1D1v, 1D2v, and 2D2v phase-space configurations
- **Advection**: Fourier (default) or arbitrary-order Lagrange interpolation, selected per call via `method = Lagrange(W)`, with periodic or mirror (specular wall) boundaries; polar advection uses splines (Dierckx)
- **Field solvers**: electrostatic (Poisson), Darwin/hybrid-kinetic (DK), and vacuum Maxwell — see [Field Solvers](#field-solvers) below
- **Optional GPU acceleration**: KernelAbstractions-based kernels, switchable at runtime via a CUDA weak dependency

## Installation

bslLD and its PlasmaCore.jl dependency are distributed through the [BSLRegistry](https://gitlab.mpcdf.mpg.de/bsl6d/BSLRegistry)
(neither is in Julia's General registry). Add the registry once per machine / cluster account:

```julia
pkg> registry add https://gitlab.mpcdf.mpg.de/bsl6d/BSLRegistry.git
```

then install it like any registered package:

```julia
pkg> add bslLD
julia> using bslLD
```

Alternatively, type `using bslLD` directly in the REPL: if the package isn't installed in the
active environment yet, Julia offers to install it.

Pick up new releases with `pkg> registry up` followed by `pkg> up`.

**Development:** clone this repository and run `julia --project=.` in it (`pkg> instantiate` on first
use), or `pkg> dev /path/to/bslLD` into another environment. To test against a local PlasmaCore.jl
checkout, `pkg> dev /path/to/PlasmaCore.jl` in the bslLD environment; this only changes the
(gitignored) `Manifest.toml`, and `pkg> free PlasmaCore` returns to the registered release.

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

### Interpolation method

`advectX!`/`advectV!` take a `method` keyword:

- `Fourier()` (default): spectral shift, exact for periodic data.
- `Lagrange(W)`: centred Lagrange stencil of even width `W ≥ 2`; required for non-periodic boundaries.

```julia
bslLD.advectX!(f, grid, simTime; method = bslLD.Lagrange(24))
bslLD.advectV!(f, grid, simTime, sol.E; method = bslLD.Lagrange(24))
```

Wider stencils damp short wavelengths less: over 200 steps the amplitude retained at `kh = 0.5π` is
0.03 for `W = 8`, 0.85 for `W = 16` and 0.99 for `W = 24`.

The advection runs on the active backend (see [Backend Selection](#backend-selection)) and does not
synchronise the device; synchronise before timing it. Buffers are cached per distribution;
`bslLD.release_advection_buffers!(f)` frees them once `f` is no longer advected.

### Boundary conditions

`advectX!` takes `boundary = Periodic()` (default) or `Mirror()`, a specular wall at the first and
last grid node of each spatial axis:

```julia
bslLD.advectX!(f, grid, simTime; method = bslLD.Lagrange(8), boundary = bslLD.Mirror())
bslLD.advectX!(f, grid, simTime, 1; method = bslLD.Lagrange(8), boundary = bslLD.Mirror())  # one direction
```

- At the wall, particles are reflected with the velocity components the reflection reverses: with
  `B` along `grid.Bdir`, a wall perpendicular to `B` reverses both gyration-plane velocities, a wall
  along `B` the field-aligned one. The wall nodes are symmetrised, `f(x_b, v) == f(x_b, R v)`.
- Mass is conserved to roundoff in the reflecting-domain norm (half weight on the two wall nodes);
  the plain sum over grid points drifts at `O(Δx²)`.
- `Mirror()` needs a `Lagrange` method, applies to spatial axes only (`advectV!` is periodic), and
  needs velocity axes symmetric about zero. All three are checked.

#### Grid for a mirror domain

`Grid` excludes the upper bound `etaMax` from x-axes (the periodic convention) and includes it in
v-axes. For walls at `0` and `Lx` on the `N` nodes `0, Δx, …, Lx`, pass `etaMax = Lx + Δx` with
`Δx = Lx/(N-1)`:

```julia
Lx, N = 2pi, 65                         # N nodes including both walls
dx    = Lx / (N - 1)
grid  = bslLD.Grid([0.0, -vmax, -vmax], [Lx + dx, vmax, vmax], [N, Nv, Nv], 1)
# grid.xaxes[1] == 0:dx:Lx; velocity axes have Nv + 1 nodes, symmetric about 0
```

#### Field solve on a mirror domain

The spectral field solvers assume periodic `x`. `AdiabaticSolver` takes a boundary per spatial axis:
`Mirror()` axes are differentiated by a centred finite difference of even `order` with ghosts
mirrored about the walls (so `E = 0` there), `Periodic()` axes spectrally.

```julia
solver = bslLD.AdiabaticSolver((bslLD.Mirror(), bslLD.Periodic()); order = 8)   # x mirror, y periodic
sol = bslLD.solve_fields(bslLD.Moments(rho), grid, solver)
```

## Field Solvers

All solvers implement the `AbstractFieldSolver` interface via `solve_fields` / `solve_fields!`, which takes a `Moments` struct and returns a `FieldSolution` holding the updated `E` and `B` vector fields.

### Electrostatic solvers

| Solver | Equation solved | Required moments |
|--------|----------------|-----------------|
| `PoissonSolver(factor=1.0)` | $-\nabla^2\phi = \text{factor}\cdot\rho$, $\mathbf{E}=-\nabla\phi$ | `rho` |
| `AdiabaticSolver(boundaries = (); order = 8)` | $\mathbf{E} = -\nabla\rho$ (adiabatic electrons, no Poisson inversion); `Mirror()` axes by finite differences of `order` | `rho` |

Both are FFT-based and support arbitrary 1D/2D periodic domains. A 2/3-rule dealiasing filter is applied.

```julia
sol = bslLD.solve_fields(bslLD.Moments(rho), grid, bslLD.PoissonSolver())
Ex  = sol.E[1]   # ScalarField
```

### Darwin / hybrid-kinetic (EM, low-frequency)

The Darwin model retains electromagnetic effects while filtering out light waves by neglecting the transverse displacement current. Three variants are provided for kinetic-ion / fluid-electron plasmas, parametrised by $\beta_i$ (ion plasma beta) and $\mu = m_e/m_i$ (mass ratio).

| Solver | Description | Required moments |
|--------|-------------|-----------------|
| `EMSolverDKPol(β_i, μ)` | Darwin-Kinetic **with** the electron polarisation current $\mu\partial_t\mathbf{E}_\perp$. Implicit $\mathbf{E}_\perp$ update via a $2\times2$ linear system per Fourier mode; Helmholtz solve for $E_\parallel$. Staggered Faraday for $\mathbf{B}$. | `rho`, `J` (⊥ current), `Pi_diff` (anisotropic pressure) |
| `EMSolverDKNoPol(β_i, μ)` | Darwin-Kinetic **without** polarisation drift. Fully spectral $3\times3$ Cramer solve per Fourier mode; simultaneous Faraday update. | `rho`, `J`, `Pi_diff` |
| `EMSolverDKPolDensity(β_i, μ; w = 1)` | Drift-kinetic electrons with the polarisation **density** only (weight $w$), in charge form: quasineutrality $\sum_s q_s n_s = w\mu[(2/\beta_i)\Delta B_\parallel + \hat b\cdot\nabla\times\mathbf{J}_{\perp}]\,/\,q_e$ replaces $\nabla\cdot\mathbf{J} = 0$. Works on explicit species (see below), any geometry: `step_charge_form!(species, sol, grid, solver, dt; phase, damping_rate)` is one X–C–X step (stage field from a $3\times3$ solve per Fourier mode, `solve_fields_moment_implicit!`; $\theta = 1/2 + \kappa\,dt$ damps the parallel-Ampère defect mode at second order). `quasineutral_drift!` sets a quasineutral start, `quasineutrality_residual` checks it. `charge_form_energy(species, sol, grid, solver)` returns the energy at $t^n$ (magnetic, per species `species_energy`, polarisation from the state, total); it is conserved to $O(dt^2)$ for $\mathbf{k}\perp\hat b$, while for oblique $\mathbf{k}$ the density form leaves a $dt$-independent defect. 1D, $\mathbf{k}\perp\hat b$ only: `quasineutral_B!` and the lagged `solve_fields!`. | explicit species |

**Species and equation-of-motion models.** A species is a mass $m$ and charge $q$ plus a model, `Species(m, q, model)` or `Species(model; q, m)`. The model type carries the equations of motion and owns the whole state; methods dispatch on it (`Species{<:Any,<:Kinetic}`), and its fields are reachable from the species (`sp.dist`, `sp.delta_n`). Models: `Kinetic(dist)` (full orbit; `Species(m, q, dist)` wraps a distribution in it), and the linearised cold fluid `ColdFluid(grid, closure)` with closure `LorentzFluid()` (full cold fluid; `ColdIonFluid(grid)` = cold ions, $q = m = 1$) or `DriftFluid()` ($\mathbf{u}_\perp = \mathbf{E}\times\hat b$, parallel inertia; cold drift-kinetic electrons: `Species(ColdFluid(grid, DriftFluid()); q = -1, m = μ)`), and the linear drift-kinetic model `LinearDK(grid; T_e, Nv = 64, vmax = 6, method = Fourier())` ($\delta F(\mathbf x, v_\parallel)$ on its own 1V grid, $T_{e\perp} = 0$; electrons: `Species(LinearDK(grid; T_e); q = -1, m = μ)`). `LinearDK` streams with the kinetic `advectX!` along $\hat b$ (keyword `vdirs = (Bdir,)`: physical direction of each velocity axis, default `1:NV`) and kicks with `add_linear_dk_kick!` (sources.jl); with Fourier streaming its stage response is exact, with `Lagrange(W)` the corrector applies. Every model provides `push_free!(sp, grid, dt, half; phase, B)` (field-free half step; `B` = $\mathbf B^n$ or $\mathbf B^{n+1}$, used by `LinearDK`, whose streaming acts on $\delta f = \delta F - F_0\delta B_\parallel/b_0$), `push_field!` (kick), `moments_at` (charge density, current) and `stage_response(sp, grid, dt, θ; phase, B)` (linear response to the stage field per Fourier mode), so the charge-form step couples any set of them, e.g. `(ions, electrons)` with kinetic or cold-fluid ions. `response_class(sp)` selects the coupling: `Magnetised()` (⊥ current through $(q^2n/m)\mathbf{E}$; kinetic, Lorentz fluid) or `Drift()` ($E\times B$ current, carries the polarisation density; drift fluid, `LinearDK`).

The rate-form solvers use `solve_fields!(sol, moments, grid, solver, dt)` (in-place, stores result in `sol.Enew`):

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
| Darwin hybrid-kinetic | 1D2v + background $B_0$ | `EMSolverDKPol` / `EMSolverDKNoPol` / `EMSolverDKPolDensity` | `em_cases`, `field_solver_tests.jmd`, `electron_models.jmd` (electron models of the charge form: cold fluid → linear DK → nonlinear; roadmap) |
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
│   │   ├── interpolation.jl           # arbitrary-order Lagrange weights (uniform grid)
│   │   ├── species.jl                 # Species = (m, q, model); AbstractModel, DistributionModel, Kinetic
│   │   ├── advectorCart.jl            # BSL advection (Cartesian, KernelAbstractions)
│   │   └── advectorPolar.jl           # BSL advection (polar coordinates)
│   ├── maxwell/
│   │   ├── differential_operators.jl  # spectral grad, curl, div
│   │   ├── spectral.jl                # FFT helpers, wavenumbers
│   │   ├── field_solver.jl            # AbstractFieldSolver, Moments, FieldSolution
│   │   ├── solvers_electrostatic.jl   # PoissonSolver, AdiabaticSolver
│   │   ├── solvers_vacuum.jl          # EMSolverVacuum, VacuumMaxwellParams
│   │   ├── solvers_hybrid.jl          # EMSolverDKPol, EMSolverDKNoPol, EMSolverDKPolDensity
│   │   ├── cold_plasma.jl             # ColdFluid model (Lorentz / drift closure) for linear EM benchmarks
│   │   ├── charge_form.jl             # species models ↔ charge-form solve, step_charge_form!
│   │   └── dk_electrons.jl            # LinearDK model (linear drift-kinetic electrons)
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
| `KineticDR.jl` | Analytic dispersion references (BSLRegistry) in `field_solver_tests.jmd` |

**PlasmaCore.jl status:** this package depends on `PlasmaCore.jl` via a remote git URL declared in `Project.toml`/`Manifest.toml`, not a local vendored copy, even though `PlasmaCore.jl` sits right next to this repo on disk. If full separation between development streams is later required, the recommended approach is to vendor a copy into `vendor/PlasmaCore.jl` within this project and repoint the dependency to that local path.

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

## License

ISC, see [LICENSE](LICENSE). Part of the [BSL6D](https://gitlab.mpcdf.mpg.de/bsl6d/bsl6d) project; authors in [AUTHORS.md](AUTHORS.md).
