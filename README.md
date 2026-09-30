# bslLD

Low-dimensional Backward Semi-Lagrangian (BSL) solver for testing numerical methods for plasma physics.

This package is a testing ground for numerical methods intended for [BSL6D](https://gitlab.mpcdf.mpg.de/bsl6d/bsl6d), providing 1D/2D configuration space with 2D velocity space at reduced computational cost.

## Features

- **Grid systems**: Cartesian and polar coordinate support
- **Distribution functions**: 1D1v, 1D2v, and 2D2v phase-space configurations
- **Advection**: Fourier (default) or arbitrary-order Lagrange interpolation, selected per call via `method = Lagrange(W)`; polar advection uses splines (Dierckx)
- **Interpolation primitives**: arbitrary even-order Lagrange stencils (`lagrange_weights`, `lagrange_gather`), allocation-free and usable inside KernelAbstractions kernels
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

`advectX!`/`advectV!` take a `method` keyword. The default `Fourier()` is spectrally exact but
imposes periodicity; `Lagrange(W)` performs the shift with a centred Lagrange stencil of even
width `W`, which is a dispatch parameter (each `W` compiles its own unrolled kernel):

```julia
bslLD.advectX!(f, grid, simTime; method = bslLD.Lagrange(24))
bslLD.advectV!(f, grid, simTime, sol.E; method = bslLD.Lagrange(24))
```

Wider stencils hold the spectrum flat out to larger `k`: over 200 steps the amplitude retained at
`kh = 0.5π` is 0.03 for `W = 8`, 0.85 for `W = 16` and 0.99 for `W = 24`. See
`examples/interpolation_spectra.jl`.

When the shift is constant along the advected line (all X/V shifts and the geometry shears), the
Lagrange path runs one thread per line. It computes the weights once and walks the line by
stride, which costs about 3–7 ns per point for `W = 8` on 4 CPU threads. Shifts that depend on the
advected coordinate itself, such as the `CurvedPatch` u_z scaling, keep the per-point kernel.

### Orbit window (rotating grid)

The shift on the rotating logical grid uses `Q` at `simTime.phase`, which is the midpoint rule.
`orbit_window = w` (`advectX!`, `advectV!`, `advect_geometry!`) instead averages `Q` exactly over
a window `w` centred on that phase: the gyration-plane part is multiplied by `sin(Ω w/2)/(Ω w/2)`.
Pass the time span that the substep represents. For a Strang step
`V(h/2) X(h) V(h/2)`, that is phases `t + h/4`, `t + h/2`, `t + 3h/4` with windows `h/2`, `h`, `h/2`.
This removes the O((Ω h)²) gyro-phase error of the midpoint rule. `w = 0` (the default) keeps
the midpoint rule.

### Boundary conditions

`advectX!` also takes `boundary`, either `Periodic()` (default) or `Mirror()`, the specular wall
ported from bsl6d's `105-implement-mirror-boundary-conditions`:

```julia
bslLD.advectX!(f, grid, simTime, 1; method = bslLD.Lagrange(8), boundary = bslLD.Mirror())
```

The halo is filled by reflecting the interior about the first and last grid node while reversing
the velocity components the reflection flips — with `B` along `grid.Bdir`, a mirror perpendicular
to `B` reverses both gyration-plane velocities, a mirror along `B` the field-aligned one — and the
boundary nodes are symmetrised so `f(x_b, v) == f(x_b, R v)`. Mass is then conserved to roundoff in
the reflecting-domain norm (half weight on the two mirror nodes); the plain sum over grid points
drifts at `O(Δx²)` because it over-weights them.

`Mirror()` requires a `Lagrange` method (the spectral path is periodic by construction), applies to
spatial axes only, and needs velocity axes symmetric about zero — all three are checked. The mirror
planes sit at `x[1]` and `x[end]` (node-type axis, as in bsl6d: `… N-1, N | N-1, N-2 …`).
bslLD's Cartesian x-axis excludes its upper endpoint, so to put node `N` on the wall at `L`, pass
`xmax = L + L/(N-1)` to `Grid`.

**Wall sub-cycling.** With fields, a reflecting halo is unstable when a departure point lies
several cells beyond the wall (γ ≈ 0.1 Ω for `Lagrange(8)` at shifts of about 2.5 cells, in bands
of `(dt, Δx)`). bsl6d rejects shifts of one cell or more on mirror axes. bslLD instead sub-cycles
wall axes automatically: `advectX!` splits the sweep into `ceil(max shift / max_wall_shift)`
sub-sweeps, where `max_wall_shift = 1.0` cell by default and the maximum is taken over the
rotated velocity box. Periodic axes are never sub-cycled. Pass `max_wall_shift = Inf` for the
old single-sweep behaviour.

**`Specular()`.** On an axis perpendicular to `B`, `Mirror()` reverses both gyration-plane
components (u → −u). That is invariant under the rotating grid, but it is not the physical
reflection. `Specular()` reflects `v_x → −v_x`. On the rotating grid that is `u' = R(2φ) F u`,
with `F` the u_x index flip and the rotation done as three shears on the halo planes only. It uses
the same node fold and wall-node symmetrisation. Along `B` it is identical to `Mirror()`. Mass in
the reflecting-domain norm is conserved to the velocity-interpolation error of the rotated halo,
not to roundoff. With sub-cycling, both walls are stable, and in the tests so far (passive wall
layer, κ = 0 stability) they give the same result, so `Mirror()` remains the cheaper default.
`AdiabaticSolver((Mirror(),))` or `AdiabaticSolver((Specular(),))` provides the matching
even-extension field.

### Magnetic geometry

`advectX!` takes a `geometry` keyword, and `advect_geometry!` applies the
velocity-space part of a geometry. All geometries assume spatial axis 1 radial
(x), axis 2 binormal (y), axis 3 (if present) along `B`, with `Bdir = 3` and three
velocity components. The rotating logical velocity grid is unchanged.

| Geometry | Spatial characteristic | Velocity space (`advect_geometry!`) |
|---|---|---|
| `Slab()` (default) | `ẋ = Q u` | none |
| `ShearedSlab(; Ls, R0, kcurv)` | `ẏ += (x/Ls) v_∥ − (v_∥² + v⊥²/2)/(Ω R0) cos(kcurv z)` | none |
| `CurvedPatch(; Rc)` | `ż = Rc/(Rc+x) v_z` | residual field `B_z(x) = Rc/(Rc+x)` (exact three-shear rotation) and curvature force `a = (v_z², 0, −v_x v_z)/(Rc+x)` |

```julia
g = bslLD.CurvedPatch(Rc = 50.0)
bslLD.advect_geometry!(f, grid, simTime, g; method = bslLD.Lagrange(8))
bslLD.advectX!(f, grid, simTime, 1; method = bslLD.Lagrange(8), boundary = bslLD.Mirror(), geometry = g)
```

A drift that enters only the y-characteristic does no work against `E_y`.
`add_drift_energy!(f, grid, dt, g::ShearedSlab, E)` adds the linearised
`E_y v_My F_M` term, which recovers the gyrokinetic `(ω − ω*)/(ω − ω_D)` response.
`CurvedPatch` needs no such term. Its characteristics conserve `f`, and mass
is conserved in the measure `J dx dv` with `J = (Rc+x)/Rc`.

`add_kappaT!(f, grid, dt, kappa_T, E; kappa_n = 0)` is the local gradient drive
`dt (E×B)_x (κ_n + κ_T (v²/2 − NV/2)) F_M`.

## Field Solvers

All solvers implement the `AbstractFieldSolver` interface via `solve_fields` / `solve_fields!`, which takes a `Moments` struct and returns a `FieldSolution` holding the updated `E` and `B` vector fields.

### Electrostatic solvers

| Solver | Equation solved | Required moments |
|--------|----------------|-----------------|
| `PoissonSolver(factor=1.0)` | $-\nabla^2\phi = \text{factor}\cdot\rho$, $\mathbf{E}=-\nabla\phi$ | `rho` |
| `AdiabaticSolver(boundaries = ())` | $\mathbf{E} = -\nabla\rho$ (adiabatic electrons, no Poisson inversion); `boundaries[d] = Mirror()` differentiates the even extension along axis `d` | `rho` |

Both are FFT-based and support arbitrary 1D/2D periodic domains. `AdiabaticSolver((Mirror(), Periodic()))` pairs with `Mirror()` advection in x. A 2/3-rule dealiasing filter is applied.

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
│   │   ├── interpolation.jl           # arbitrary-order Lagrange weights/gather (uniform grid)
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

## Author

Mario Raeth
