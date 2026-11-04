# Charge-form Darwin field solve (EMSolverDKPolDensity) over explicit species, and its X–C–X step.
#
# A species is a charge q and mass m plus an equation-of-motion model: kinetic (`Species`), or a
# cold fluid with Lorentz or drift closure (`ColdFluid`, cold_plasma.jl). Every model provides
#   push_free!(sp, grid, dt, half; phase)    field-free half step, half = 1 before / 2 after C
#   push_field!(sp, E, grid, dt; phase)      kick by the stage field over dt (C)
#   moments_at(sp, grid, phase)              (charge density q n, field-indexed current J)
#   stage_response(sp, grid, dt, θ; phase)   per Fourier mode, the affine map of the stage field
#                                            E^m to J^θ, ρ^{n+1} and J⊥^{n+1} (see below)
# The field solve sums the species responses. Exact responses (drift fluid) need no corrector;
# the others (Lorentz fluid, kinetic) get one kinetic corrector with a fixed defect.

# ── Kinetic species ───────────────────────────────────────────────────────────
# Velocities in units of the species thermal speed 1/√m: the physical current is q/√m times
# `compute_current`, and its response to a kick is (q²/m) n E. The gyration is in the rotating
# frame: the phase points ¼, ½, ¾, 1 of the step follow stepMidpoint!.

function _kinetic_time(sp::Species, grid::Grid, dt, phase, at, fraction)
    t = SimulationTime(dt, dt)
    t.phase = phase + at * gyro_frequency(sp, grid) * dt
    t.fraction_dt = fraction
    return t
end

push_free!(sp::Species, grid::Grid, dt::Real, half::Int; phase = 0) =
    (advectX!(sp, grid, _kinetic_time(sp, grid, dt, phase, half == 1 ? 0.25 : 0.75, 0.5)); sp)

push_field!(sp::Species, E::VectorField, grid::Grid, dt::Real; phase = 0) =
    (advectV!(sp, grid, _kinetic_time(sp, grid, dt, phase, 0.5, 1.0), E); sp)

function moments_at(sp::Species, grid::Grid, phase::Real)
    J = compute_current(sp, grid, phase)
    s = sp.q * thermal_velocity(sp)
    z = zero(J[1].data)
    return (sp.q .* compute_density(sp.dist, grid).data,
            ntuple(d -> d <= length(J) ? s .* J[d].data : z, 3))
end

_save(sp::Species) = copy(sp.dist.data)
_restore!(sp::Species, s) = (sp.dist.data .= s; sp)

# Phase of the stage (½) and of the end of the step (1) for the moments.
_stage_phase(sp::Species, grid, dt, phase) = phase + 0.5 * gyro_frequency(sp, grid) * dt
_end_phase(sp::Species, grid, dt, phase) = phase + gyro_frequency(sp, grid) * dt
_stage_phase(::ColdFluid, grid, dt, phase) = phase
_end_phase(::ColdFluid, grid, dt, phase) = phase

# ── Stage responses ───────────────────────────────────────────────────────────
# Frame (a, b, ∥), right-handed with ∥ = Bdir. An affine response is (c0, ca, cb, c∥): arrays of
# Fourier coefficients with value c0 + ca e_a + cb e_b + c∥ e_∥ for the stage field e. A response
# holds J (stage current, 3 components), rho (charge density at t^{n+1}), Jp (⊥ current at t^{n+1},
# a and b; zero for the drift species, whose ⊥ current is the E×B current) and `drift` = (q, m) of
# a drift species or `nothing`.

function _cf_frame(grid::Grid)
    pz = grid.Bdir
    d1, d2 = [d for d = 1:3 if d != pz]
    return d2 % 3 + 1 == pz ? (d1, d2, pz) : (d2, d1, pz)
end

function _cf_kviews(arr, grid)
    a, b, pz = _cf_frame(grid)
    sw = spectral_workspace(arr, grid)
    kv(d) = _make_kview_or_zero(sw, d, spatial_ndims(grid), ndims(arr), arr)
    return sw, kv(a), kv(b), kv(pz)
end

_affine(c0) = (c0, zero(c0), zero(c0), zero(c0))
_affine_at(A, Ea, Eb, Ez) = @. A[1] + A[2] * Ea + A[3] * Eb + A[4] * Ez
_affine_add(A, B) = map(.+, A, B)

# Lorentz fluid and kinetic species: J^θ = J^a + θ dt α E (α = (q²/m) n0 on the directions that
# have a velocity), ρ^{n+1} = ρ^a − (dt/2) ∇·(J^a + dt α E), J⊥^{n+1} = J⊥^a + dt α E⊥. The model
# ignores the gyration of the second X(dt/2); the corrector supplies it.
function _magnetised_response(ρ, J, α, grid, dt, θ)
    a, b, pz = _cf_frame(grid)
    sw, ka, kb, kz = _cf_kviews(ρ, grid)
    F(x) = _fft_c(x, sw)
    Jh = (F(J[a]), F(J[b]), F(J[pz]))
    Z = zero(Jh[1])
    unit(i, c) = ntuple(j -> j == i ? fill!(similar(Z), c) : Z, 3)
    Jθ = ntuple(i -> (Jh[i], unit(i, θ * dt * α[i])...), 3)
    Jend = ntuple(i -> (Jh[i], unit(i, dt * α[i])...), 3)
    ρh = F(ρ)
    rho = (@.(ρh - dt / 2 * im * (ka * Jh[1] + kb * Jh[2] + kz * Jh[3])),
           @.(-dt^2 / 2 * im * ka * α[1] + 0 * Z), @.(-dt^2 / 2 * im * kb * α[2] + 0 * Z),
           @.(-dt^2 / 2 * im * kz * α[3] + 0 * Z))
    return (J = Jθ, rho = rho, Jp = (Jend[1], Jend[2]), drift = nothing, exact = false)
end

function stage_response(fluid::ColdFluid{LorentzFluid}, grid::Grid, dt, θ; phase = 0)
    ρ, J = moments_at(fluid, grid, phase)
    α = ntuple(_ -> fluid.q^2 / fluid.m, 3)
    return _magnetised_response(ρ, J, α, grid, dt, θ)
end

function stage_response(sp::Species, grid::Grid, dt, θ; phase = 0)
    ρ, J = moments_at(sp, grid, _stage_phase(sp, grid, dt, phase))
    n0 = sum(ρ) / length(ρ) / sp.q
    a, b, pz = _cf_frame(grid)
    nv = length(grid.vaxes)
    α = ntuple(i -> (a, b, pz)[i] <= nv ? sp.q^2 / sp.m * n0 : zero(n0), 3)
    return _magnetised_response(ρ, J, α, grid, dt, θ)
end

# Drift fluid (exact): J^θ = q (E × b̂)/b0 + q (u∥ + θ dt (q/m) E∥) b̂,
# ρ^{n+1} = q [δN^a − dt b̂·∇×E / b0 − (dt/2) ∂∥(u∥ + dt (q/m) E∥)].
function stage_response(fluid::ColdFluid{DriftFluid}, grid::Grid, dt, θ; phase = 0)
    a, b, pz = _cf_frame(grid)
    q, m, b0 = fluid.q, fluid.m, grid.b0
    sw, ka, kb, kz = _cf_kviews(fluid.delta_n.data, grid)
    F(x) = _fft_c(x, sw)
    N, U = F(fluid.delta_n.data), F(fluid.u[pz].data)
    Z = zero(N)
    c(x) = fill!(similar(Z), x)
    J = ((Z, Z, c(q / b0), Z), (Z, c(-q / b0), Z, Z), (q .* U, Z, Z, c(θ * dt * q^2 / m)))
    rho = (@.(q * (N - dt / 2 * im * kz * U)), @.(q * dt / b0 * im * kb + 0 * Z),
           @.(-q * dt / b0 * im * ka + 0 * Z), @.(-dt^2 / 2 * q^2 / m * im * kz + 0 * Z))
    return (J = J, rho = rho, Jp = (_affine(Z), _affine(Z)), drift = (q, m), exact = true)
end

# Defect of a non-exact response: the actual moments after a trial push with the field Ep
# (Ja, Jb: stage current before and after the kick; ρ1, J1 at the end of the step) minus the
# model at Ep. Added to the constant parts.
function _with_defect(r, Ja, Jb, ρ1, J1, Ep, grid, θ)
    a, b, pz = _cf_frame(grid)
    sw, _, _, _ = _cf_kviews(ρ1, grid)
    F(x) = _fft_c(x, sw)
    E = (F(Ep[a].data), F(Ep[b].data), F(Ep[pz].data))
    fix(A, actual) = (A[1] .+ (F(actual) .- _affine_at(A, E...)), A[2], A[3], A[4])
    idx = (a, b, pz)
    J = ntuple(i -> fix(r.J[i], Ja[idx[i]] .+ θ .* (Jb[idx[i]] .- Ja[idx[i]])), 3)
    Jp = ntuple(i -> fix(r.Jp[i], J1[idx[i]]), 2)
    return (J = J, rho = fix(r.rho, ρ1), Jp = Jp, drift = r.drift, exact = r.exact)
end

# ── Field solve ──────────────────────────────────────────────────────────────

# Residual of one Fourier mode for the stage field e: transverse Ampère at the stage
# (c i k × B^θ = J^θ, B^θ = B^n − θ dt i k × e) and, along k̂, quasineutrality at t^{n+1},
# ρ^{n+1} + P [c k² B∥^{n+1} − i (k_a Jp_b − k_b Jp_a)] = 0 with P = w m_d / q_d (k ≠ 0);
# the mean of Ampère's law, J^θ = 0 (k = 0).
@inline function _cf_residual(e, k, p, B, A)
    ea, eb, ez = e
    ka, kb, kz = k
    c, P, dt, θ = p
    at(i) = A[4i-3] + A[4i-2] * ea + A[4i-1] * eb + A[4i] * ez
    Ja, Jb, Jz, ρ1, Pa, Pb = at(1), at(2), at(3), at(4), at(5), at(6)
    k2 = ka^2 + kb^2 + kz^2
    iszero(k2) && return (-Ja, -Jb, -Jz)
    ca, cb, cz = im * (kb * ez - kz * eb), im * (kz * ea - ka * ez), im * (ka * eb - kb * ea)
    Ba, Bb, Bz = B[1] - θ * dt * ca, B[2] - θ * dt * cb, B[3] - θ * dt * cz
    Ra = c * im * (kb * Bz - kz * Bb) - Ja
    Rb = c * im * (kz * Ba - ka * Bz) - Jb
    Rz = c * im * (ka * Bb - kb * Ba) - Jz
    L = (ka * Ra + kb * Rb + kz * Rz) / k2
    Q = (ρ1 + P * (c * k2 * (B[3] - dt * cz) - im * (ka * Pb - kb * Pa))) / (dt * sqrt(k2))
    return (Ra - ka * (L - Q), Rb - kb * (L - Q), Rz - kz * (L - Q))
end

_det3(u, v, w) =
    u[1] * (v[2] * w[3] - v[3] * w[2]) + u[2] * (v[3] * w[1] - v[1] * w[3]) +
    u[3] * (v[1] * w[2] - v[2] * w[1])

# Stage field of one mode: the residual is affine in e; its columns give a 3×3 system (Cramer).
function _cf_solve(ka, kb, kz, p, Ba, Bb, Bz, A...)
    k, B = (ka, kb, kz), (Ba, Bb, Bz)
    z, o = zero(complex(ka)), one(complex(ka))
    r = _cf_residual((z, z, z), k, p, B, A)
    m1 = _cf_residual((o, z, z), k, p, B, A) .- r
    m2 = _cf_residual((z, o, z), k, p, B, A) .- r
    m3 = _cf_residual((z, z, o), k, p, B, A) .- r
    d = _det3(m1, m2, m3)
    return (_det3(.-r, m2, m3) / d, _det3(m1, .-r, m3) / d, _det3(m1, m2, .-r) / d)
end

# P = w m_d / q_d of the (single) drift species; 0 without one.
function _polarisation_coefficient(responses, solver::EMSolverDKPolDensity)
    drifts = [r.drift for r in responses if r.drift !== nothing]
    isempty(drifts) && return 0.0
    length(drifts) == 1 || throw(ArgumentError("the polarisation density needs a single drift species"))
    q, m = drifts[1]
    m ≈ solver.mu || throw(ArgumentError("drift-species mass $m differs from solver.mu = $(solver.mu)"))
    return solver.w * m / q
end

"""
    solve_fields_moment_implicit!(sol, responses, grid, solver::EMSolverDKPolDensity, dt;
                                  theta = 0.5, B_n = sol.B)

Stage field E^m of an X–C–X step in charge form, any grid geometry, from the species
`responses` (`stage_response`, summed). Per Fourier mode it solves transverse Ampère at the
stage t^n + θ dt, (2/β_i) ∇×B^θ = J^θ transverse with B^θ = B^n − θ dt ∇×E^m, and
quasineutrality at t^{n+1} with the polarisation density of the drift species d,
Σ q n = (w m_d / q_d) [(2/β_i) ΔB_∥ + b̂·∇×J⊥] (J⊥ of the other species), B^{n+1} = B^n − dt ∇×E^m.
The k = 0 mode imposes J^θ = 0. θ = 1/2 + κ dt keeps second order and damps the undamped
parallel-Ampère defect mode of θ = 1/2. Writes E^m to `sol.Enew`; `sol.B` is not modified.
"""
function solve_fields_moment_implicit!(
    sol::FieldSolution,
    responses,
    grid::Grid,
    solver::EMSolverDKPolDensity,
    dt::Real;
    theta::Real = 0.5,
    B_n::VectorField = sol.B,
)
    1 / 2 <= theta <= 1 || throw(ArgumentError("theta must lie in [1/2, 1], got $theta"))
    a, b, pz = _cf_frame(grid)
    arr = sol.B[1].data
    DT = eltype(arr)
    sw, ka, kb, kz = _cf_kviews(arr, grid)
    Bh = (_fft_c(B_n[a].data, sw), _fft_c(B_n[b].data, sw), _fft_c(B_n[pz].data, sw))
    total = reduce((x, y) -> map(_affine_add, x, y),
                   [(r.J..., r.rho, r.Jp...) for r in responses])
    p = (DT(2 / solver.beta_i), DT(_polarisation_coefficient(responses, solver)), DT(dt), DT(theta))
    Eh = _cf_solve.(ka, kb, kz, Ref(p), Bh..., Iterators.flatten(total)...)
    for (d, j) in ((a, 1), (b, 2), (pz, 3))
        ifft_spatial!(sol.Enew[d].data, getindex.(Eh, j), sw)
    end
    return sol
end

# ── Quasineutral start and diagnostics ───────────────────────────────────────

# Fourier quasineutrality residual ρ + P [c k² B∥ − i (k_a J_b − k_b J_a)] of the current state,
# without the drift species' own charge density, which is returned separately (q_d, δN̂_d).
function _qn_parts(species, B, grid, solver, phase)
    a, b, pz = _cf_frame(grid)
    sw, ka, kb, kz = _cf_kviews(B[1].data, grid)
    F(x) = _fft_c(x, sw)
    c = 2 / solver.beta_i
    ρ, Ja, Jb = zero(F(B[1].data)), zero(F(B[1].data)), zero(F(B[1].data))
    drift = nothing
    for sp in species
        if sp isa ColdFluid{DriftFluid}
            drift === nothing || throw(ArgumentError("the polarisation density needs a single drift species"))
            drift = sp
            continue
        end
        ρs, Js = moments_at(sp, grid, phase)
        ρ .+= F(ρs); Ja .+= F(Js[a]); Jb .+= F(Js[b])
    end
    P = drift === nothing ? 0.0 : solver.w * drift.m / drift.q
    k2 = @. ka^2 + kb^2 + kz^2
    r = @. ρ + P * (c * k2 * $F(B[pz].data) - im * (ka * Jb - kb * Ja))
    return r, k2, drift, sw
end

"""
    quasineutral_drift!(species, B, grid, solver::EMSolverDKPolDensity; phase = 0)

Set δn of the drift species from quasineutrality with the other species and `B`; k = 0 is kept.
"""
function quasineutral_drift!(species, B::VectorField, grid::Grid, solver::EMSolverDKPolDensity; phase = 0)
    r, k2, drift, sw = _qn_parts(species, B, grid, solver, phase)
    drift === nothing && throw(ArgumentError("no drift species"))
    N = _fft_c(drift.delta_n.data, sw)
    @. N = ifelse(iszero(k2), N, -r / drift.q)
    ifft_spatial!(drift.delta_n.data, N, sw)
    return species
end

"""
    quasineutrality_residual(species, B, grid, solver::EMSolverDKPolDensity; phase = 0)

max over x of |Σ q n − (w m_d / q_d)[(2/β_i) ΔB_∥ + b̂·∇×J⊥]| without the k = 0 mode.
"""
function quasineutrality_residual(species, B::VectorField, grid::Grid, solver::EMSolverDKPolDensity; phase = 0)
    r, k2, drift, sw = _qn_parts(species, B, grid, solver, phase)
    drift === nothing || (r .+= drift.q .* _fft_c(drift.delta_n.data, sw))
    @. r = ifelse(iszero(k2), zero(r), r)
    out = similar(drift === nothing ? B[1].data : drift.delta_n.data)
    ifft_spatial!(out, r, sw)
    return maximum(abs, out)
end

# ── X–C–X step ───────────────────────────────────────────────────────────────

"""
    step_charge_form!(species, sol, grid, solver::EMSolverDKPolDensity, dt; phase = 0, damping_rate = 0)

One X–C–X step in charge form for a tuple of species (kinetic `Species`, `ColdFluid`): X(dt/2),
stage field from the moment-implicit solve, one corrector for the non-exact species (trial push
with the predicted field, fixed defect), C with the stage field, X(dt/2), then Faraday.
`phase` is the gyro-phase of the kinetic species at t^n; `damping_rate` κ gives θ = 1/2 + κ dt.
"""
function step_charge_form!(
    species,
    sol::FieldSolution,
    grid::Grid,
    solver::EMSolverDKPolDensity,
    dt::Real;
    phase::Real = 0,
    damping_rate::Real = 0,
)
    θ = 1 / 2 + damping_rate * dt
    foreach(sp -> push_free!(sp, grid, dt, 1; phase), species)
    saved = map(_save, species)
    responses = [stage_response(sp, grid, dt, θ; phase) for sp in species]
    solve_fields_moment_implicit!(sol, responses, grid, solver, dt; theta = θ)

    if !all(r -> r.exact, responses)
        Ep = VectorField([copy(sol.Enew[d].data) for d = 1:3])
        for (i, sp) in enumerate(species)
            responses[i].exact && continue
            Ja = moments_at(sp, grid, _stage_phase(sp, grid, dt, phase))[2]
            push_field!(sp, Ep, grid, dt; phase)
            Jb = moments_at(sp, grid, _stage_phase(sp, grid, dt, phase))[2]
            push_free!(sp, grid, dt, 2; phase)
            ρ1, J1 = moments_at(sp, grid, _end_phase(sp, grid, dt, phase))
            responses[i] = _with_defect(responses[i], Ja, Jb, ρ1, J1, Ep, grid, θ)
            _restore!(sp, saved[i])
        end
        solve_fields_moment_implicit!(sol, responses, grid, solver, dt; theta = θ)
    end

    foreach(species) do sp
        push_field!(sp, sol.Enew, grid, dt; phase)
        push_free!(sp, grid, dt, 2; phase)
    end
    apply_faraday!(sol.B, sol.Enew, grid, dt)
    sol.E .= sol.Enew
    return nothing
end
