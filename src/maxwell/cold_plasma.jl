# Cold-ion fluid model for testing the EM field solvers.
#
# Linearized cold-ion equations around n0=1, u_i=0, B0=b0*ê_{Bdir}:
#   ∂δn/∂t = −∇·δu_i                          (continuity, spectral)
#   ∂δu_i/∂t = E + Ω_i (δu_i × ê_z)           (momentum, cold ⇒ no pressure)
#
# Moments fed to the EM solvers:
#   rho    = ScalarField with n0 = 1  (constant; keeps density decoupled for linear testing)
#   J      = δu_i_⟂  (2-component, velocity-space indexed [1,2] matching solvers_hybrid.jl)
#   Pi_diff = 0        (cold ions)
#
# Integration: Strang split mirroring the kinetic step! in em_cases.ipynb
#   V(dt/2) – X(dt/2) – moments – solve_fields! – X(dt/2) – V(dt/2)

struct ColdIonFluid{SF<:ScalarField,VF<:VectorField}
    delta_n::SF    # δn (diagnostic only; not fed to field solver)
    u::VF          # δu_i, 3 field-space components
end

function ColdIonFluid(grid::Grid)
    dims = Tuple(length(ax) for ax in grid.xaxes)
    delta_n = ScalarField(zeros(Float64, dims))
    u = VectorField([ScalarField(zeros(Float64, dims)) for _ = 1:3])
    return ColdIonFluid(delta_n, u)
end

# Build the Moments struct expected by EMSolverDKPol / EMSolverDKNoPol.
# Uses constant n0=1 so the field solve stays in the linear regime.
function compute_moments(fluid::ColdIonFluid, grid::Grid)
    dims = size(fluid.delta_n.data)
    rho = ScalarField(ones(Float64, dims))
    perp_dirs = [d for d = 1:3 if d != grid.Bdir]
    J_perp = VectorField([fluid.u[perp_dirs[1]], fluid.u[perp_dirs[2]]])
    Pi_diff = zero_vectorfield3(grid)
    return Moments(rho, J_perp, Pi_diff)
end

# Half-step ion velocity update: δu += (dt/2) * E, then rotate by (dt/2)*Ω_i about ê_{Bdir}.
function _advance_u_half!(fluid::ColdIonFluid, E::VectorField, grid::Grid, dt::Real)
    DT = eltype(fluid.u[1].data)
    half = DT(dt / 2)
    Ω = DT(grid.b0)
    pz = grid.Bdir
    perp_dirs = [d for d = 1:3 if d != pz]
    d1, d2 = perp_dirs[1], perp_dirs[2]
    parity = DT((d2 % 3 + 1 == pz) ? 1 : -1)  # sign of ε_{d1,d2,Bdir}

    # E-kick
    for d = 1:3
        fluid.u[d].data .+= half .* E[d].data
    end

    # Rotation by half*Ω_i
    θ = half * Ω
    c, s = cos(θ), sin(θ)
    u1 = copy(fluid.u[d1].data)
    u2 = copy(fluid.u[d2].data)

    fluid.u[d1].data .= c .* u1 .+ parity .* s .* u2
    fluid.u[d2].data .= .-parity .* s .* u1 .+ c .* u2


    return nothing
end

# Half-step continuity update (spectral): δn += -(dt/2) * ∇·δu_i.
function _advance_n_half!(fluid::ColdIonFluid, grid::Grid, dt::Real)
    ndims_x = spatial_ndims(grid)
    # div only over spatial components
    u_spatial = VectorField([fluid.u[d] for d = 1:ndims_x])
    div_u = div(u_spatial, grid)
    DT = eltype(fluid.delta_n.data)
    fluid.delta_n.data .-= DT(dt / 2) .* div_u.data
    return nothing
end

# One full Strang-split step for the cold-ion fluid + EM field solve.
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
