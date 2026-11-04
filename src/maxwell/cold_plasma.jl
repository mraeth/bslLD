# Linearised cold-fluid model for testing the EM field solvers.
#
# `ColdFluid` is a species model (species.jl): the state δn, δu and a closure of the perpendicular
# motion, linearised around n0 = 1, u = 0 and B0 = b0 ê_{Bdir}; q and m are those of the `Species`:
#   LorentzFluid:  ∂t δu = (q/m) (E + b0 δu × b̂)        (full cold fluid)
#   DriftFluid:    δu⊥ = E × b̂ / b0 (E×B drift),  ∂t δu∥ = (q/m) E∥
# with continuity ∂t δn = −∇·δu. Cold ions: `ColdIonFluid(grid)` = Lorentz, q = m = 1.
# Drift-kinetic cold electrons: `Species(ColdFluid(grid, DriftFluid()); q = -1, m = μ)`.
#
# Moments fed to the rate-form EM solvers (`compute_moments`, Lorentz fluid):
#   rho = n0 + δn, J = q δu⊥ (2-component, velocity-space indexed [1,2]), Pi_diff = 0.
#
# Integration: Strang split mirroring the kinetic step! in em_cases.ipynb
#   V(dt/2) – X(dt/2) – moments – solve_fields! – X(dt/2) – V(dt/2)   (`step_cold!`)
# and the X–C–X building blocks `push_free!` / `push_field!` of the charge form (charge_form.jl).

abstract type FluidClosure end
struct LorentzFluid <: FluidClosure end
struct DriftFluid <: FluidClosure end

struct ColdFluid{C<:FluidClosure,SF<:ScalarField,VF<:VectorField} <: AbstractModel
    closure::C
    delta_n::SF    # δn
    u::VF          # δu, 3 field-space components (DriftFluid: only the ∥ component is state)
end

function ColdFluid(grid::Grid, closure::FluidClosure)
    dims = Tuple(length(ax) for ax in grid.xaxes)
    delta_n = ScalarField(zeros(Float64, dims))
    u = VectorField([ScalarField(zeros(Float64, dims)) for _ = 1:3])
    return ColdFluid(closure, delta_n, u)
end

const FluidSpecies{C} = Species{<:AbstractFloat,<:ColdFluid{C}}
const ColdIonFluid = FluidSpecies{LorentzFluid}
ColdIonFluid(grid::Grid) = Species(ColdFluid(grid, LorentzFluid()))

# Build the Moments struct expected by the rate-form hybrid EM solvers (rho = 1 + δn, n0 = 1).
function compute_moments(fluid::ColdIonFluid, grid::Grid)
    rho = ScalarField(1 .+ fluid.delta_n.data)
    perp_dirs = [d for d = 1:3 if d != grid.Bdir]
    J_perp = VectorField([ScalarField(fluid.q .* fluid.u[d].data) for d in perp_dirs])
    Pi_diff = zero_vectorfield3(grid)
    return Moments(rho, J_perp, Pi_diff)
end

# Half-step velocity update: δu += (q/m)(dt/2) E, then gyrate by (q/m) b0 dt/2 about ê_{Bdir}.
function _advance_u_half!(fluid::ColdIonFluid, E::VectorField, grid::Grid, dt::Real)
    DT = eltype(fluid.u[1].data)
    a = DT(fluid.q / fluid.m * dt / 2)
    for d = 1:3
        fluid.u[d].data .+= a .* E[d].data
    end
    _rotate_u!(fluid, grid, a * DT(grid.b0))
    return nothing
end

# Gyration of δu by the angle θ = (q/m) b0 τ about ê_{Bdir}.
function _rotate_u!(fluid::FluidSpecies, grid::Grid, θ::Real)
    DT = eltype(fluid.u[1].data)
    pz = grid.Bdir
    perp_dirs = [d for d = 1:3 if d != pz]
    d1, d2 = perp_dirs[1], perp_dirs[2]
    parity = DT((d2 % 3 + 1 == pz) ? 1 : -1)  # sign of ε_{d1,d2,Bdir}
    c, s = DT(cos(θ)), DT(sin(θ))
    u1 = copy(fluid.u[d1].data)
    u2 = copy(fluid.u[d2].data)

    fluid.u[d1].data .= c .* u1 .+ parity .* s .* u2
    fluid.u[d2].data .= .-parity .* s .* u1 .+ c .* u2
    return nothing
end

# Half-step continuity update (spectral): δn += -(dt/2) * ∇·δu.
function _advance_n_half!(fluid::FluidSpecies, grid::Grid, dt::Real)
    ndims_x = spatial_ndims(grid)
    # div only over spatial components
    u_spatial = VectorField([fluid.u[d] for d = 1:ndims_x])
    div_u = div(u_spatial, grid)
    DT = eltype(fluid.delta_n.data)
    fluid.delta_n.data .-= DT(dt / 2) .* div_u.data
    return nothing
end

# One full Strang-split step for the Lorentz fluid + EM field solve.
function step_cold!(
    fluid::ColdIonFluid,
    sol::FieldSolution,
    grid::Grid,
    solver::AbstractFieldSolver,
    dt::Real,
)
    # V(dt/2) with E^n
    _advance_u_half!(fluid, sol.E, grid, dt)
    # X(dt/2)
    _advance_n_half!(fluid, grid, dt)

    # Midpoint moments -> field solve
    moments_mid = compute_moments(fluid, grid)
    solve_fields!(sol, moments_mid, grid, solver, dt)

    # X(dt/2)
    _advance_n_half!(fluid, grid, dt)
    # V(dt/2) with E^{n+1}
    _advance_u_half!(fluid, sol.Enew, grid, dt)

    sol.E .= sol.Enew
    return nothing
end

# Overload matching the notebook step!(f_i, sol, grid, simTime, solver) convention.
function step_cold!(
    fluid::ColdIonFluid,
    sol::FieldSolution,
    grid::Grid,
    simTime::SimulationTime,
    solver::AbstractFieldSolver,
)
    step_cold!(fluid, sol, grid, solver, simTime.dt)
end

# ── X–C–X building blocks (charge form) ──────────────────────────────────────
# Field-free half step (`half` = 1 before the kick, 2 after) and the kick by the stage field.
# The cold fluids ignore `phase` (gyration is explicit).

function push_free!(fluid::FluidSpecies{LorentzFluid}, grid::Grid, dt::Real, half::Int; phase = 0, B = nothing)
    θ = fluid.q / fluid.m * grid.b0 * dt / 2
    if half == 1
        _advance_n_half!(fluid, grid, dt)
        _rotate_u!(fluid, grid, θ)
    else
        _rotate_u!(fluid, grid, θ)
        _advance_n_half!(fluid, grid, dt)
    end
    return fluid
end

# Parallel flux only: δu⊥ is not state and stays zero.
push_free!(fluid::FluidSpecies{DriftFluid}, grid::Grid, dt::Real, half::Int; phase = 0, B = nothing) =
    (_advance_n_half!(fluid, grid, dt); fluid)

function push_field!(fluid::FluidSpecies{LorentzFluid}, E::VectorField, grid::Grid, dt::Real; phase = 0)
    a = fluid.q / fluid.m * dt
    foreach(d -> fluid.u[d].data .+= a .* E[d].data, 1:3)
    return fluid
end

# Parallel kick and the compression of the E×B flow, ∂t δn = −∇⊥·(E × b̂)/b0 = −b̂·∇×E / b0.
function push_field!(fluid::FluidSpecies{DriftFluid}, E::VectorField, grid::Grid, dt::Real; phase = 0)
    pz = grid.Bdir
    fluid.u[pz].data .+= (fluid.q / fluid.m * dt) .* E[pz].data
    fluid.delta_n.data .-= (dt / grid.b0) .* curl(E, grid)[pz].data
    return fluid
end

# Charge density q δn and field-indexed current q δu (DriftFluid: the ∥ state current only).
moments_at(fluid::FluidSpecies, grid::Grid, phase::Real) =
    (fluid.q .* fluid.delta_n.data, ntuple(d -> fluid.q .* fluid.u[d].data, 3))

_save(fluid::FluidSpecies) = (copy(fluid.delta_n.data), ntuple(d -> copy(fluid.u[d].data), 3))
function _restore!(fluid::FluidSpecies, s)
    fluid.delta_n.data .= s[1]
    foreach(d -> fluid.u[d].data .= s[2][d], 1:3)
    return fluid
end
