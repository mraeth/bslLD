# Electrostatic quasineutral solve with massless resistive electrons on a density gradient.
#
# Model (ion units: Ω_i = b0 = 1, lengths ρ_i, φ in T_i/e; b̂ = ẑ, gradient along x):
#   ions:      full orbit (Kinetic, or a ColdFluid for tests), plus the Boussinesq gradient drive
#              −v_E,x n0 [κn + κT (v²/2 − NV/2)] F_M (`add_gradient_drive!`, sources.jl) on kinetic species;
#   electrons: ∂t δn + v_E·∇δn = −κn n0 v_E,x + α_k (n0 φ/τ − δn),  v_E = b̂ × ∇φ / b0,
#              n_e = n0 + δn + μ n0 k⊥² φ / b0² (polarisation density, Fourier form),
#              α_k = α for k_y ≠ 0, α_zonal for k_y = 0 (Hasegawa–Wakatani closure, KineticDR `Resistive`);
#   field:     Σ q n = 0 (electrostatic quasineutrality).
#
# Time scheme: the X–C–X step of the charge form (charge_form.jl) with the stage potential φ^m at
# t^{n+1/2}. The ions' existing `stage_response` gives ρ_i^{n+1} affine in the stage field; its
# −(dt²/2) n0 k² φ^m term is an implicit ion polarisation that keeps every mode regular (no explicit
# lower-hybrid or k²/(κn k_y) time-step limit). The electron update is Crank–Nicolson in α and
# midpoint in the drive, so its response is exact. Quasineutrality at t^{n+1} is one linear equation
# per Fourier mode. One corrector (trial push of the non-exact ion species, fixed defect; nonlinear
# E×B term re-evaluated) as in `step_charge_form!`. Periodic Cartesian grids with Bdir = 3 only.

"""
    ResistiveElectrons(grid; tau, alpha, alpha_zonal = alpha, kappa_n = 0, n0 = 1,
                       polarization = true, nonlinear = false)

Massless isothermal electrons with a resistive parallel coupling on a background density gradient
(the KineticDR `Resistive(α; polarization)` closure at k∥ = 0), for `step_es_qn!`. State: the gyrocentre
density perturbation `delta_n` on the configuration grid. `tau` = T_e/T_i, `alpha` in units of Ω_i,
`alpha_zonal` is used on k_y = 0, `kappa_n` = ∂x ln n0 (also drives the kinetic ions), `n0` the background
density. `D_perp` = ρ_e²ν_e (ρ_i²Ω_i) adds classical cross-field diffusion, which relaxes δn toward the Boltzmann
response at rate D⊥k⊥²: the coupling is α_k = α + D⊥k⊥² (KineticDR: kinetic BGK electrons at k∥ = 0 match it). `nonlinear` adds the E×B advection of δn (2/3-dealiased). Species:
`Species(ResistiveElectrons(grid; ...); q = -1, m = μ)`.
"""
struct ResistiveElectrons{SF<:ScalarField,T<:AbstractFloat} <: AbstractModel
    delta_n::SF
    tau::T
    alpha::T
    alpha_zonal::T
    kappa_n::T
    n0::T
    D_perp::T
    polarization::Bool
    nonlinear::Bool
end

function ResistiveElectrons(grid::Grid; tau, alpha, alpha_zonal = alpha, kappa_n = 0.0, n0 = 1.0, D_perp = 0.0,
                            polarization::Bool = true, nonlinear::Bool = false)
    D_perp >= 0 || throw(ArgumentError("D_perp must be ≥ 0"))
    tau > 0 || throw(ArgumentError("tau must be positive"))
    alpha >= 0 && alpha_zonal >= 0 || throw(ArgumentError("alpha and alpha_zonal must be ≥ 0"))
    dims = Tuple(length(ax) for ax in grid.xaxes)
    return ResistiveElectrons(ScalarField(backend_array(zeros(Float64, dims))), float(tau), float(alpha),
                              float(alpha_zonal), float(kappa_n), float(n0), float(D_perp), polarization,
                              nonlinear)
end

const ResistiveSpecies = Species{<:AbstractFloat,<:ResistiveElectrons}
_is_resistive(sp) = getfield(sp, :model) isa ResistiveElectrons

"""
    ElectrostaticQN(; filter = 1.0)

Electrostatic quasineutral field solve for `step_es_qn!`. `filter` < 1 zeroes φ̂ for |k| > filter·max|k|.
"""
struct ElectrostaticQN{T<:AbstractFloat} <: AbstractFieldSolver
    filter::T
end
ElectrostaticQN(; filter::Real = 1.0) =
    0 < filter <= 1 ? ElectrostaticQN(float(filter)) : throw(ArgumentError("filter must lie in (0, 1]"))

function _check_es_qn_grid(grid::Grid)
    grid.Bdir == 3 && spatial_ndims(grid) == 2 ||
        throw(ArgumentError("step_es_qn! needs a 2D configuration grid (x, y) with Bdir = 3"))
    return nothing
end

# Wavenumbers of the frame (a, b, ∥) = (x, y, z) and k⊥², broadcastable against Fourier arrays.
function _esqn_k(arr, grid)
    r = eltype(arr) <: Complex ? real(arr) : arr
    sw, kx, ky, kz = _cf_kviews(r, grid)
    return sw, kx, ky, @.(kx^2 + ky^2 + kz^2 + 0 * r)
end

# ── Responses: ρ̂^{n+1} = c0 + cφ φ̂^m per Fourier mode ─────────────────────────

# Ion species from its charge-form stage response (E = −i k φ) plus the density moment of the
# gradient drive on kinetic species, q n0 κn dt (i k_y / b0) φ̂.
function _esqn_ion_response(sp, grid, dt, kappa_n; phase = 0, E0 = 0)
    r = stage_response(sp, grid, dt, 0.5; phase)
    c0, ca, cb, cz = r.rho
    sw, kx, ky, _ = _esqn_k(c0, grid)
    cφ = @. -im * (kx * ca + ky * cb)
    # The kicked δf streams over the second X(dt/2) with the ions' velocities, not from rest: its density is
    # the kick's times ⟨e^{−i k·v dt/2}⟩. For a flow u (imposed E0, drag) this is e^{−i k·u dt/2}, a phase the
    # corrector cannot absorb at k u dt ≳ 0.5 (unstable at high k_y). Mean lab-frame flow of the species.
    if sp isa KineticSpecies
        ux, uy = _kinetic_mean_flow(sp, grid, _end_phase(sp, grid, dt, phase))
        (iszero(ux) && iszero(uy)) || (cφ = @.(cφ * exp(-im * (kx * ux + ky * uy) * dt / 2)))
    end
    # the kick by the uniform E0 x̂ on the density n: ρ^{n+1} −= (dt²/2)(q/m) E0 ∂x ρ (ρ^a ≈ c0 to O(dt³))
    iszero(E0) || (c0 = @.(c0 - dt^2 / 2 * sp.q / sp.m * E0 * im * kx * c0))
    if sp isa KineticSpecies && !iszero(kappa_n)
        n0 = real(sum(view(c0, 1:1))) / length(c0) / sp.q          # k = 0 coefficient = Σρ
        @. cφ += sp.q * n0 * kappa_n * dt * im * ky / grid.b0
    end
    return (c0 = copy(c0), cφ = cφ, exact = false)
end

_esqn_alpha(m::ResistiveElectrons, ky, k2) = @. ifelse(iszero(ky), m.alpha_zonal, m.alpha) + m.D_perp * k2

# Electron density update (7 of the S3 plan): δn̂^{n+1} = (A δn̂^n + dt S φ̂ − dt NL̂)/D.
# E0: uniform imposed E_x; its E×B drift v0 = −E0/b0 along y advects δn (−i k_y v0 δn̂, with the same
# Crank–Nicolson factor as α).
function _esqn_electron_coeffs(el, grid, dt, Nn, E0 = 0)
    m = getfield(el, :model)
    sw, kx, ky, k2 = _esqn_k(Nn, grid)
    α = _esqn_alpha(m, ky, k2)
    λ = @. α + im * ky * (-E0 / grid.b0)
    D = @. 1 + λ * dt / 2
    A = @. (1 - λ * dt / 2) / D
    S = @. dt * (im * m.kappa_n * m.n0 * ky / grid.b0 + α * m.n0 / m.tau) / D
    P = m.polarization ? @.(el.m * m.n0 * k2 / grid.b0^2 + 0 * D) : zero(S)
    return A, S, D, P
end

# The polarisation density at t^{n+1} uses φ^{n+1} ≈ (3φ^m − φ^{m−1})/2 (Φprev = φ̂^{m−1}): with φ^m
# alone it lags by dt/2, which makes the scheme first order (relative error ≈ μk² ω dt/2).
function _esqn_electron_response(el, grid, dt, Nn, NL, Φprev, E0 = 0)
    A, S, D, P = _esqn_electron_coeffs(el, grid, dt, Nn, E0)
    c0 = NL === nothing ? @.(el.q * (A * Nn - P * Φprev / 2)) :
         @.(el.q * (A * Nn - dt * NL / D - P * Φprev / 2))
    return (c0 = c0, cφ = @.(el.q * (S + 3P / 2)), exact = true)
end

# E×B advection v_E·∇δn, v_E = (−∂yφ, ∂xφ)/b0, 2/3-dealiased, in Fourier space.
function _esqn_nonlinear(φ, δn, grid)
    sw, kx, ky, _ = _esqn_k(φ, grid)
    F(x) = _fft_c(x, sw)
    Φ, N = F(φ), F(δn)
    real_of(X) = ifft_spatial!(similar(φ), X, sw)
    vx, vy = real_of(@.(-im * ky * Φ / grid.b0)), real_of(@.(im * kx * Φ / grid.b0))
    nx, ny = real_of(@.(im * kx * N)), real_of(@.(im * ky * N))
    NL = F(@.(vx * nx + vy * ny))
    kxm, kym = maximum(abs, kx), maximum(abs, ky)
    @. NL = ifelse((abs(kx) > 2kxm / 3) | (abs(ky) > 2kym / 3), zero(NL), NL)
    return NL
end

# φ̂ = −Σ c0 / Σ cφ per mode; k = 0 and the filtered modes are zero.
function _esqn_solve(responses, grid, solver::ElectrostaticQN)
    C0 = reduce((x, y) -> x .+ y, [r.c0 for r in responses])
    Cφ = reduce((x, y) -> x .+ y, [r.cφ for r in responses])
    _, _, _, k2 = _esqn_k(C0, grid)
    kcut2 = solver.filter^2 * maximum(k2)
    return @. ifelse(iszero(k2) | (k2 > kcut2), zero(C0), -C0 / Cφ)
end

function _esqn_store!(sol, Φ, grid)
    sw, kx, ky, _ = _esqn_k(Φ, grid)
    ifft_spatial!(sol.phi.data, Φ, sw)
    ifft_spatial!(sol.Enew[1].data, @.(-im * kx * Φ), sw)
    ifft_spatial!(sol.Enew[2].data, @.(-im * ky * Φ), sw)
    fill!(sol.Enew[3].data, 0)
    return sol
end

_esqn_drive!(sp, E, grid, dt, m, kappa_T, profile) = sp isa KineticSpecies ?
    add_gradient_drive!(sp.dist, grid, dt, E; kappa_n = m.kappa_n, kappa_T, n0 = _kinetic_n0(sp, grid), profile) : sp

# BGK half steps at both ends (B X C X B): BGK conserves the density pointwise, so it leaves the
# quasineutrality bookkeeping of the step unchanged.
_esqn_bgk!(ions, grid, nu, dt, T) =
    iszero(nu) ? ions : foreach(sp -> sp isa KineticSpecies && add_bgk!(sp.dist, grid, nu, dt; T), ions)

# x–y mean flow (lab frame, physical velocity units) of a kinetic species at gyrophase `phase`; zero below
# 1e-12 so that a species at rest keeps the plain charge-form response.
function _kinetic_mean_flow(sp, grid, phase)
    J = _lab_current_arrays(sp, grid, phase)
    n = sum(compute_density(sp.dist, grid).data)
    u = ntuple(a -> a <= length(J) ? thermal_velocity(sp) * sum(J[a]) / n : 0.0, 2)
    return map(x -> abs(x) < 1e-12 ? zero(x) : x, u)
end

_kinetic_n0(sp, grid) = sum(compute_density(sp.dist, grid).data) / prod(length.(grid.xaxes))

function _esqn_update_electrons!(el, grid, dt, Φ, NL, E0 = 0)
    sw = spectral_workspace(el.delta_n.data, grid)
    Nn = _fft_c(el.delta_n.data, sw)
    A, S, D, _ = _esqn_electron_coeffs(el, grid, dt, Nn, E0)
    N1 = NL === nothing ? @.(A * Nn + S * Φ) : @.(A * Nn + S * Φ - dt * NL / D)
    ifft_spatial!(el.delta_n.data, N1, sw)
    return el
end

"""
    step_es_qn!(species, sol, grid, solver::ElectrostaticQN, dt; phase = 0, kappa_T = 0)

One X–C–X step of the electrostatic quasineutral model: ion species (`Kinetic`, or `ColdFluid` for
tests) and exactly one `ResistiveElectrons` species. Stage potential φ^m from quasineutrality at
t^{n+1} (ion stage response + exact electron update), one corrector (trial push of the ions with
E_p = −∇φ_p; nonlinear term re-evaluated with δn^{n+1/2}), then C with E^m (plus the gradient drive
with the electrons' κn and the ion `kappa_T` on kinetic species) and X(dt/2). `nu > 0` wraps the step in
BGK half steps (`add_bgk!`, target temperature `T_bgk`, at rest in the lab frame) on the kinetic species.
`E0` is a uniform imposed E_x (ion units; its E×B drift is v0 = −E0/b0 along y): it is added to the ion kick
and advects the electron δn. `drive_profile = :mean` drives the ions with the instantaneous x–y mean of f
(a drifting or heated uniform equilibrium) instead of F_M (κn only). Writes E^m to
`sol.E`/`sol.Enew` and φ^m to `sol.phi`. `phase` is the gyrophase of the kinetic species at t^n; the
caller advances it by Ω dt.
"""
function step_es_qn!(species, sol::FieldSolution, grid::Grid, solver::ElectrostaticQN, dt::Real;
                     phase::Real = 0, kappa_T::Real = 0, correctors::Int = 1, nu::Real = 0, T_bgk::Real = 1,
                     E0::Real = 0, drive_profile::Symbol = :maxwellian)
    _check_es_qn_grid(grid)
    # kick field: E^m plus the uniform imposed E0 x̂
    Ek(E) = iszero(E0) ? E : VectorField([ScalarField(E[1].data .+ E0), E[2], E[3]])
    els = [sp for sp in species if _is_resistive(sp)]
    length(els) == 1 || throw(ArgumentError("step_es_qn! needs exactly one ResistiveElectrons species"))
    el, m = els[1], getfield(els[1], :model)
    ions = [sp for sp in species if !_is_resistive(sp)]

    _esqn_bgk!(ions, grid, nu, dt / 2, T_bgk)
    foreach(sp -> push_free!(sp, grid, dt, 1; phase), ions)
    saved = map(_save, ions)
    responses = [_esqn_ion_response(sp, grid, dt, m.kappa_n; phase, E0) for sp in ions]
    sw = spectral_workspace(el.delta_n.data, grid)
    Nn = _fft_c(el.delta_n.data, sw)
    Φprev = _fft_c(sol.phi.data, sw)
    NL = m.nonlinear ? _esqn_nonlinear(sol.phi.data, el.delta_n.data, grid) : nothing
    Φ = _esqn_solve([responses..., _esqn_electron_response(el, grid, dt, Nn, NL, Φprev, E0)], grid, solver)

    for _ = 1:(isempty(ions) && !m.nonlinear ? 0 : correctors)
        Ep = VectorField([ScalarField(similar(sol.E[d].data)) for d = 1:3])
        φp = similar(sol.phi.data)
        tmp = FieldSolution(sol.E, sol.B, Ep, ScalarField(φp))
        _esqn_store!(tmp, Φ, grid)
        for (i, sp) in enumerate(ions)
            push_field!(sp, Ek(Ep), grid, dt; phase)
            _esqn_drive!(sp, Ek(Ep), grid, dt, m, kappa_T, drive_profile)
            push_free!(sp, grid, dt, 2; phase)
            ρ1 = moments_at(sp, grid, _end_phase(sp, grid, dt, phase))[1]
            r = responses[i]
            responses[i] = (c0 = @.(r.c0 + ($(_fft_c(ρ1, sw)) - (r.c0 + r.cφ * Φ))), cφ = r.cφ, exact = false)
            _restore!(sp, saved[i])
        end
        if m.nonlinear
            δp = copy(el.delta_n.data)
            _esqn_update_electrons!(el, grid, dt, Φ, NL, E0)
            half = @. (δp + el.delta_n.data) / 2
            el.delta_n.data .= δp
            NL = _esqn_nonlinear(φp, half, grid)
        end
        Φ = _esqn_solve([responses..., _esqn_electron_response(el, grid, dt, Nn, NL, Φprev, E0)], grid, solver)
    end

    _esqn_store!(sol, Φ, grid)
    for sp in ions
        push_field!(sp, Ek(sol.Enew), grid, dt; phase)
        _esqn_drive!(sp, Ek(sol.Enew), grid, dt, m, kappa_T, drive_profile)
    end
    _esqn_update_electrons!(el, grid, dt, Φ, NL, E0)
    foreach(sp -> push_free!(sp, grid, dt, 2; phase), ions)
    _esqn_bgk!(ions, grid, nu, dt / 2, T_bgk)
    foreach(d -> sol.E[d].data .= sol.Enew[d].data, 1:3)
    return nothing
end

# Charge density of the ions plus that of the electrons with potential φ, in Fourier space.
function _esqn_charge(species, grid, φ; phase = 0)
    el = only([sp for sp in species if _is_resistive(sp)])
    m = getfield(el, :model)
    sw, _, _, k2 = _esqn_k(φ, grid)
    F(x) = _fft_c(x, sw)
    ρ = el.q .* F(el.delta_n.data)
    m.polarization && (ρ .+= (el.q * el.m * m.n0 / grid.b0^2) .* k2 .* F(φ))
    for sp in species
        _is_resistive(sp) || (ρ .+= F(moments_at(sp, grid, phase)[1]))
    end
    return ρ, k2, sw
end

"""
    quasineutral_electrons!(species, sol, grid; phase = 0)

Set δn of the `ResistiveElectrons` species so that Σ q n = 0 with the ions and `sol.phi`; k = 0 of δn
is set to zero. Use it for a quasineutral start.
"""
function quasineutral_electrons!(species, sol::FieldSolution, grid::Grid; phase = 0)
    el = only([sp for sp in species if _is_resistive(sp)])
    ρ, k2, sw = _esqn_charge(species, grid, sol.phi.data; phase)
    N = _fft_c(el.delta_n.data, sw)
    @. N = ifelse(iszero(k2), zero(N), N - ρ / el.q)
    ifft_spatial!(el.delta_n.data, N, sw)
    return species
end

"""
    es_qn_residual(species, sol, grid; phase = 0)

max over x of |Σ q n| without the k = 0 mode, with the electron polarisation density from `sol.phi`.
"""
function es_qn_residual(species, sol::FieldSolution, grid::Grid; phase = 0)
    ρ, k2, sw = _esqn_charge(species, grid, sol.phi.data; phase)
    @. ρ = ifelse(iszero(k2), zero(ρ), ρ)
    return maximum(abs, ifft_spatial!(similar(sol.phi.data), ρ, sw))
end
