# Cold-ion fluid model for testing the EM field solvers.
#
# Linearized cold-ion equations around n0=1, u_i=0, B0=b0*ê_{Bdir}:
#   ∂δn/∂t = −∇·δu_i                          (continuity, spectral)
#   ∂δu_i/∂t = E + Ω_i (δu_i × ê_z)           (momentum, cold ⇒ no pressure)
#
# Moments fed to the EM solvers:
#   rho    = n0 + δn  (ignored by EMSolverDKPol / EMSolverDKNoPol; quasineutrality of
#            EMSolverDKPolDensity)
#   J      = δu_i_⟂  (2-component, velocity-space indexed [1,2] matching solvers_hybrid.jl)
#   Pi_diff = 0        (cold ions)
#
# Integration: Strang split mirroring the kinetic step! in em_cases.ipynb
#   V(dt/2) – X(dt/2) – moments – solve_fields! – X(dt/2) – V(dt/2)
# and, for EMSolverDKPolDensity, an X–C–X step with a moment-implicit stage field.

struct ColdIonFluid{SF<:ScalarField,VF<:VectorField}
    delta_n::SF    # δn
    u::VF          # δu_i, 3 field-space components
end

function ColdIonFluid(grid::Grid)
    dims = Tuple(length(ax) for ax in grid.xaxes)
    delta_n = ScalarField(zeros(Float64, dims))
    u = VectorField([ScalarField(zeros(Float64, dims)) for _ = 1:3])
    return ColdIonFluid(delta_n, u)
end

# Build the Moments struct expected by the hybrid EM solvers (rho = 1 + δn, n0 = 1).
function compute_moments(fluid::ColdIonFluid, grid::Grid)
    rho = ScalarField(1 .+ fluid.delta_n.data)
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

    # E-kick
    for d = 1:3
        fluid.u[d].data .+= half .* E[d].data
    end

    _rotate_u!(fluid, grid, half * Ω)
    return nothing
end

# Gyration of δu_i by the angle θ = Ω_i τ about ê_{Bdir}.
function _rotate_u!(fluid::ColdIonFluid, grid::Grid, θ::Real)
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

# X–C–X step in charge form (EMSolverDKPolDensity). X(dt/2) = continuity + gyration, C = kick
# by dt·E^m with the stage field from the moment-implicit solve plus one corrector; B_z^{n+1}
# from quasineutrality with the final density. Background density n0 = 1.
function step_cold!(
    fluid::ColdIonFluid,
    sol::FieldSolution,
    grid::Grid,
    solver::EMSolverDKPolDensity,
    dt::Real,
)
    θ = grid.b0 * dt / 2
    x_half_first!() = (_advance_n_half!(fluid, grid, dt); _rotate_u!(fluid, grid, θ))
    x_half_second!() = (_rotate_u!(fluid, grid, θ); _advance_n_half!(fluid, grid, dt))
    kick!(E) = foreach(d -> fluid.u[d].data .+= dt .* E[d].data, 1:3)
    copy3(F) = VectorField([copy(F[d].data) for d = 1:3])
    moments() = Moments(
        ScalarField(1 .+ fluid.delta_n.data),
        VectorField([copy(fluid.u[1].data), copy(fluid.u[2].data)]),
    )

    x_half_first!()
    u_a, n_a, m_a, B_n = copy3(fluid.u), copy(fluid.delta_n.data), moments(), copy3(sol.B)
    solve_fields_moment_implicit!(sol, m_a, grid, solver, dt; B_n)

    # Corrector: cold-fluid moments after C and the second X(dt/2) with the predicted field
    E_p = copy3(sol.Enew)
    kick!(E_p)
    J_b = moments().J
    x_half_second!()
    m_1 = moments()
    solve_fields_moment_implicit!(
        sol, m_a, grid, solver, dt;
        B_n, corrector = (E = E_p, J_b = J_b, n_1 = m_1.rho, J_1 = m_1.J),
    )

    copyto!(fluid.u, u_a)
    fluid.delta_n.data .= n_a
    kick!(sol.Enew)
    x_half_second!()
    m_1 = moments()
    quasineutral_B!(sol.B, m_1.rho, m_1.J, grid, solver)

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
