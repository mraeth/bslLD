
# First-order frozen-phase midpoint current predictor.
#
# During the C substep all spatial transport is absent and simTime.phase is
# fixed at the midpoint value phase_a.  advectV! therefore applies a uniform
# (v-independent) velocity kick  Δv = eas * E_gyro * dt  where
#   E_gyro = R(Bdir, +eas*phase_a) * E_lab.
# The lab-frame current evolves as
#   J_lab(τ) = J_lab(0) + eas * n * τ * E_lab  (frozen-phase approximation).
# This predictor has O(dt²) error in J^m when E_lab = E^n, which is sufficient
# for a globally second-order scheme.
function predict_midpoint_current(
    n_a::ScalarField,
    J_a::VectorField,
    E::VectorField,
    f::DistributionGrid,
    ::CartGrid,
    dt::Real,
)
    eas = bslLD.electric_acceleration_scale(f)
    τ = dt / 2
    return VectorField([
        J_a[d].data .+ eas .* n_a.data .* τ .* E[d].data
        for d in 1:length(J_a)
    ])
end

# θ-stage current predictor: generalizes predict_midpoint_current to arbitrary θ.
# τ = θ·dt; with θ = 0.5 + κ·dt the O(dt²) error in J^θ is preserved.
function predict_stage_current(
    n_a::ScalarField,
    J_a::VectorField,
    E::VectorField,
    f::DistributionGrid,
    ::CartGrid,
    dt::Real,
    theta::Real,
)
    eas = bslLD.electric_acceleration_scale(f)
    τ = theta * dt
    return VectorField([
        J_a[d].data .+ eas .* n_a.data .* τ .* E[d].data
        for d in 1:length(J_a)
    ])
end

# Combined density + current computation to avoid two separate passes.
function compute_density_current(
    f::DistributionGrid{DT,NX,NV,NXNV,Cart},
    grid::CartGrid,
    phase::Real,
) where {DT,NX,NV,NXNV}
    n = compute_density(f, grid)
    J = compute_current(f, grid, phase)
    return n, J
end
