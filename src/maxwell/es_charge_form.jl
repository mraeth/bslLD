# Electrostatic quasineutral charge form with an x-dependent profile, and its X–C–X step.
#
# Quasineutrality with the polarisation density of the drift species d,
#
#   Σ_s q_s n_s + w m_d ∇⊥·(n_d ∇⊥φ)/b0² = 0,
#
# is imposed at t^{n+1}. The species are kicked by the stage field E^m = −∇φ^m with
# φ^m = θ φ^{n+1} + (1 − θ) φ^n, θ = 1/2 + κ dt. The charge density after the step, ρ^{n+1}(φ^m), is
# measured by a trial push (push_field!, push_drift!, push_free!; for FullDK the kick in halves around the
# E×B push, `_push_stage!`) at a predicted φ^p and continued
# linearly with a model operator A of the species responses:
#
#   (A + P/θ) φ^m = −ρ^{n+1}(φ^p) + A φ^p + (1 − θ)/θ P φ^n,     P = w m_d ∇⊥·(n̄_d ∇⊥)/b0²,
#
#   A = Σ_magnetised (dt²/2)(q²/m) ∇·(n̄ ∇)                         kick, then half a free push
#     + Σ_drift [(dt²/2)(q²/m) ∂∥(n̄ ∂∥) + q dt (b̂×∇n̄)·∇/b0]       parallel kick; E×B on n̄(x)
#
# with n̄(x) the species densities averaged over the Fourier axes at the predictor. The E×B term is
# the implicit E×B response; `implicit_drift = false` leaves it out of A (the response is then only in
# the trial push, i.e. lagged). Geometry: b̂ = x̂3; x1 carries the profile (periodic, spectral
# differentiation matrix), x2 (and x3, the parallel axis, if present) are Fourier axes, so A and P are
# one dense x1-operator per (k2, k3). After the step φ^{n+1} solves quasineutrality of the new state.
# `background` is a fixed charge density added to Σ q n (immobile ions: no species, no response in A).
# For a drift species with a 2/3-dealiased E×B push the E×B term of A is the linearisation of that push,
# q dt P[(b̂×∇Pn̄)·∇Pφ]/b0.
#
# `variable_coefficients = true` (nonlinear runs) uses the predictor densities n(x1, x2, x3) instead of
# n̄(x1) in A and P, with the full E×B term q dt (∂1n ∂2φ − ∂2n ∂1φ)/b0, and the full n_d in the
# polarisation density of quasineutrality. The operators then couple the Fourier indices; they are
# applied pseudo-spectrally and solved by GMRES, right-preconditioned with the n̄ operators per index.
# `krylov = false` replaces the GMRES solve of a correction by one step of defect correction with the n̄
# operator (semi-implicit: the stage operator frozen at n̄, the residual with n(x)).

"""
    ESChargeForm(; w = 1, implicit_drift = true, background = nothing, variable_coefficients = false,
                 krylov = true, rtol = 1e-10)

Electrostatic quasineutral charge-form solve with the polarisation density (weight `w`) of the single
drift species and an x1-dependent profile. `implicit_drift` puts the E×B response of the drift
species into the stage operator. `background`: a fixed charge density (array of the spatial shape)
added to the species charge, e.g. immobile ions with only a drift species in `species`.
`variable_coefficients`: the densities n(x) of all axes in the stage operator and the polarisation density
(nonlinear runs), solved by GMRES to the relative residual `rtol` (`krylov = false`: one defect
correction with the averaged operator per correction instead); otherwise their average over the Fourier axes. See [`step_es_charge_form!`](@ref).
"""
Base.@kwdef struct ESChargeForm{BG}
    w::Float64 = 1.0
    implicit_drift::Bool = true
    background::BG = nothing
    variable_coefficients::Bool = false
    krylov::Bool = true
    rtol::Float64 = 1e-10
end

_fixed_charge(s::ESChargeForm) = something(s.background, 0)

# Periodic spectral differentiation matrix on N nodes (N even) of a period L.
function _spectral_diff_matrix(N::Int, L::Real)
    iseven(N) || throw(ArgumentError("the x1 axis needs an even number of nodes, got $N"))
    h = 2π / N
    return [i == j ? 0.0 : 0.5 * (-1)^(i - j) * cot((i - j) * h / 2) * (2π / L) for i in 1:N, j in 1:N]
end

struct _ESGeometry{M,K,B}
    D::M          # d/dx1
    k::K          # (k2, k3) per Fourier index, k3 = 0 without an x3 axis
    nyquist::B    # Nyquist index along x2 or x3
    nx::Int
end

function _es_geometry(grid::Grid)
    grid.Bdir == 3 || throw(ArgumentError("ESChargeForm needs b̂ = x̂3 (Bdir = 3)"))
    nx = length(grid.xaxes)
    2 <= nx <= 3 || throw(ArgumentError("ESChargeForm needs 2 or 3 spatial axes, got $nx"))
    N = length.(grid.xaxes)
    L = [grid.max[d] - grid.min[d] for d in 1:nx]
    kax = [2π .* fftfreq(N[d], N[d] / L[d]) for d in 2:nx]
    I = CartesianIndices(Tuple(N[2:nx]))
    k = map(J -> (kax[1][J[1]], nx == 3 ? kax[2][J[2]] : 0.0), I)
    nyq = map(J -> any(d -> iseven(N[d+1]) && J[d] == N[d+1] ÷ 2 + 1, 1:nx-1), I)
    return _ESGeometry(_spectral_diff_matrix(N[1], L[1]), k, nyq, nx)
end

_es_fft(a, g::_ESGeometry) = fft(complex(a), 2:g.nx)
_es_ifft(a, g::_ESGeometry) = real(ifft(a, 2:g.nx))
_es_mean_x(a, g::_ESGeometry) = vec(sum(a; dims = 2:g.nx)) ./ prod(size(a)[2:g.nx])

# ∇⊥·(n ∇⊥) and ∂∥(n ∂∥) at (k2, k3) as x1-matrices.
_es_diag(v) = [i == j ? v[i] : zero(eltype(v)) for i in eachindex(v), j in eachindex(v)]
_es_perp(D, n, k2) = D * (n .* D) - k2^2 * _es_diag(n)
_es_par(n, k3) = -k3^2 * _es_diag(n)

# `dl`: nothing, or (Mx, keep) for a species whose E×B push is 2/3-dealiased (`_exb_dealiased`): the E×B
# response is then P{Pn̄, Pφ} with Mx the x1 mask and `keep` whether k2 lies in the kept band.
function _es_species_operator(sp, n, D, k2, k3, dt, b0, implicit_drift, dl = nothing)
    c = dt^2 / 2 * sp.q^2 / sp.m
    if is_drift(sp)
        A = complex(c * _es_par(n, k3))
        implicit_drift || return A
        if dl === nothing
            A += sp.q * dt / b0 * im * k2 * _es_diag(D * n)                # −dt u_E·∇n̄, u_E = b̂×∇φ/b0
        elseif dl[2]
            A += sp.q * dt / b0 * im * k2 * (dl[1] * _es_diag(D * (dl[1] * n)) * dl[1])
        end
        return A
    end
    return complex(c * (_es_perp(D, n, k2) + _es_par(n, k3)))
end

# 2/3 mask along x1 as an N×N matrix (the x1 part of exb_dealias!).
function _es_mask_x(N)
    keep = abs.(fftfreq(N, N)) .<= N / 3
    return real(ifft(keep .* fft(_es_diag(ones(N)), 1), 1))
end

_drift_species(species) = (d = filter(is_drift, collect(species)); length(d) == 1 ? only(d) :
    throw(ArgumentError("the polarisation density needs exactly one drift species, got $(length(d))")))

_species_density(sp, grid, phase) = moments_at(sp, grid, phase)[1] ./ sp.q

# Per Fourier index: (A, P) with P = w m_d ∇⊥·(n̄_d ∇⊥)/b0².
function _es_operators(species, grid, solver::ESChargeForm, g::_ESGeometry, dt, phase)
    ns = [_es_mean_x(_species_density(sp, grid, _stage_phase(sp, grid, dt, phase)), g) for sp in species]
    d = findfirst(is_drift, collect(species))
    md = _drift_species(species)
    cp = solver.w * md.m / grid.b0^2
    Mx = _es_mask_x(length(grid.xaxes[1]))
    N2, L2 = length(grid.xaxes[2]), grid.max[2] - grid.min[2]
    return map(g.k) do (k2, k3)
        dl = (Mx, abs(k2) * L2 / 2π <= N2 / 3 + 1e-9)
        A = sum(_es_species_operator(sp, n, g.D, k2, k3, dt, grid.b0, solver.implicit_drift,
                                     _exb_dealiased(sp) ? dl : nothing)
                for (sp, n) in zip(species, ns))
        (A = A, P = complex(cp * _es_perp(g.D, ns[d], k2)))
    end
end

# Solve M x = r per Fourier index, without Nyquist modes: φ has no x1-Nyquist component (−1)^j (in the
# null space of the even-N spectral D, and differentiated differently by the FFT derivatives of the
# pushes), no x1-constant at k2 = 0 (k⊥ = 0: the polarisation density does not fix φ there; with k3 ≠ 0
# this is the k⊥ → 0 limit of the ω_H branch), and no Nyquist mode along x2, x3 (φ̂ = 0 there).
# Galerkin on the complement Π⊥ = 1 − Π: (Π⊥ M Π⊥ + c Π) x = Π⊥ r gives Π x = 0 and Π⊥(M x − r) = 0.
# (Regularising only M + cΠ, with r projected, leaves a Π component in x through M and makes the
# θ-step map non-neutral: |g| up to 1.46 at Δt = 0.5, θ = 1/2, electrons only.)
function _es_factor(M, k, nyquist::Bool)
    nyquist && return nothing
    N = size(M, 1)
    Z = [(-1)^j for j in 0:N-1] ./ sqrt(N)
    iszero(k[1]) && (Z = hcat(ones(N) ./ sqrt(N), Z))
    Π = Z * Z'
    Πc = _es_diag(ones(N)) - Π
    return (Πc * M * Πc + maximum(abs, M) * Π, Πc)
end
_es_solve(F, r) = F === nothing ? zero(r) : F[1] \ (F[2] * r)
_es_solve(M, r, k, nyquist::Bool) = _es_solve(_es_factor(M, k, nyquist), r)

function _es_apply(ops, field, x̂)
    y = similar(x̂)
    for (J, op) in pairs(IndexCartesian(), ops)
        @views y[:, Tuple(J)...] .= getfield(op, field) * x̂[:, Tuple(J)...]
    end
    return y
end

# E = −∇φ: x1 by the differentiation matrix, the Fourier axes spectrally; three components.
function es_field(φ::AbstractArray, grid::Grid, g::_ESGeometry = _es_geometry(grid))
    N = size(φ)
    E1 = -reshape(g.D * reshape(φ, N[1], :), N)
    φ̂ = _es_fft(φ, g)
    kk(i) = reshape(getindex.(g.k, i), 1, size(g.k)...)
    E2 = _es_ifft(-im .* kk(1) .* φ̂, g)
    E3 = g.nx == 3 ? _es_ifft(-im .* kk(2) .* φ̂, g) : zero(E1)
    return VectorField([ScalarField(E1), ScalarField(E2), ScalarField(E3)])
end

# --- Variable coefficients -------------------------------------------------------------------------

# ∂/∂x_d of a real spatial array: x1 by the differentiation matrix, the Fourier axes spectrally.
function _es_deriv(a, g::_ESGeometry, d)
    d == 1 && return reshape(g.D * reshape(a, size(a, 1), :), size(a))
    d > g.nx && return zero(a)
    kk = reshape(getindex.(g.k, d - 1), 1, size(g.k)...)
    return _es_ifft(im .* kk .* _es_fft(a, g), g)
end

_es_perp(n, φ, g::_ESGeometry) = (∂ = (a, d) -> _es_deriv(a, g, d);
    ∂(n .* ∂(φ, 1), 1) .+ ∂(n .* ∂(φ, 2), 2))
_es_par(n, φ, g::_ESGeometry) = (∂ = (a, d) -> _es_deriv(a, g, d); ∂(n .* ∂(φ, 3), 3))
_es_exb(n, φ, g::_ESGeometry) = (∂ = (a, d) -> _es_deriv(a, g, d);
    ∂(n, 1) .* ∂(φ, 2) .- ∂(n, 2) .* ∂(φ, 1))

# A and P of the module header with the predictor densities n(x), as functions on real arrays.
function _es_full_operators(species, grid, solver::ESChargeForm, g::_ESGeometry, dt, phase)
    ns = [_species_density(sp, grid, _stage_phase(sp, grid, dt, phase)) for sp in species]
    nd = ns[findfirst(is_drift, collect(species))]
    cp = solver.w * _drift_species(species).m / grid.b0^2
    function A(φ)
        y = zero(φ)
        for (sp, n) in zip(species, ns)
            c = dt^2 / 2 * sp.q^2 / sp.m
            y .+= c .* _es_par(n, φ, g)
            is_drift(sp) || (y .+= c .* _es_perp(n, φ, g))
            is_drift(sp) && solver.implicit_drift && (y .+= sp.q * dt / grid.b0 .* (_exb_dealiased(sp) ?
                exb_dealias!(_es_exb(exb_dealias!(copy(n)), exb_dealias!(copy(φ)), g)) : _es_exb(n, φ, g)))
        end
        return y
    end
    P(φ) = cp .* _es_perp(nd, φ, g)
    return A, P
end

# Removes the Nyquist modes and the x1-constant at k2 = 0 (see _es_solve) from a real array.
function _es_project(a, g::_ESGeometry)
    â = _es_fft(a, g)
    N = size(a, 1)
    z = [(-1)^j for j in 0:N-1] ./ sqrt(N)
    o = ones(N) ./ sqrt(N)
    for J in CartesianIndices(g.k)
        v = view(â, :, Tuple(J)...)
        g.nyquist[J] && (v .= 0; continue)
        v .-= z .* sum(z .* v)
        iszero(g.k[J][1]) && (v .-= o .* sum(o .* v))
    end
    return _es_ifft(â, g)
end

# Per-index solve with the factors F of the n̄ operator (output in the projected space).
function _es_precondition(r, F, g::_ESGeometry)
    r̂ = _es_fft(r, g)
    for J in CartesianIndices(F)
        @views r̂[:, Tuple(J)...] .= _es_solve(F[J], r̂[:, Tuple(J)...])
    end
    return _es_ifft(r̂, g)
end

# Solves M x = b on the projected space: GMRES (JuliaGMG) on Q M x = Q b, right-preconditioned with
# the per-index solve with the factors F of the n̄ operator (whose range is the projected space).
# Returns (x, iterations, relative residual).
function _es_krylov(M, b, F, g::_ESGeometry; rtol = 1e-10, maxiter = 200)
    K!(y, r) = copyto!(y, vec(_es_precondition(reshape(r, size(b)), F, g)))
    A!(y, x) = copyto!(y, vec(_es_project(M(reshape(x, size(b))), g)))
    Qb = vec(_es_project(b, g))
    x, st, res, _ = JuliaGMG.solve(A!, Qb, K!, JuliaGMG.GMRES(); tol_rel = rtol, tol_abs = 0.0, max_iters = maxiter)
    return reshape(x, size(b)), st.niter, (isempty(res) ? 0.0 : res[end] / sqrt(sum(abs2, Qb)))
end

"""
    es_quasineutral_phi!(sol, species, grid, solver::ESChargeForm; phase = 0)

φ from quasineutrality of the current state, Σ q n + P φ = 0; writes `sol.phi` and `sol.E = −∇φ`.
With `solver.variable_coefficients`, GMRES iteration counts are pushed to `iters` (a vector) if given.
"""
function es_quasineutral_phi!(sol::FieldSolution, species, grid::Grid, solver::ESChargeForm; phase = 0,
                              iters = nothing)
    g = _es_geometry(grid)
    md = _drift_species(species)
    n = _es_mean_x(_species_density(md, grid, phase), g)
    cp = solver.w * md.m / grid.b0^2
    ρ = sum(moments_at(sp, grid, phase)[1] for sp in species) .+ _fixed_charge(solver)
    F = map(J -> _es_factor(complex(cp * _es_perp(g.D, n, g.k[J][1])), g.k[J], g.nyquist[J]), CartesianIndices(g.k))
    if solver.variable_coefficients
        nd = _species_density(md, grid, phase)
        φ, it, _ = _es_krylov(φ -> cp .* _es_perp(nd, φ, g), -ρ, F, g; solver.rtol)
        iters === nothing || push!(iters, it)
        sol.phi.data .= φ
    else
        ρ̂ = _es_fft(ρ, g)
        φ̂ = similar(ρ̂)
        for J in CartesianIndices(g.k)
            @views φ̂[:, Tuple(J)...] .= _es_solve(F[J], -ρ̂[:, Tuple(J)...])
        end
        sol.phi.data .= _es_ifft(φ̂, g)
    end
    E = es_field(sol.phi.data, grid, g)
    foreach(d -> sol.E[d].data .= E[d].data, 1:3)
    return sol
end

# C of one species at the stage field: kick, then E×B push. For FullDK the parallel kick is split in halves
# around the E×B push: the two do not commute for k_z ≠ 0 at finite amplitude (their commutator is the E×B
# advection of E∥), and kick-then-push is first order in it.
function _push_stage!(sp, E, φ, grid, dt; phase)
    push_field!(sp, E, grid, dt; phase)
    push_drift!(sp, φ, grid, dt)
end
function _push_stage!(sp::FullDKSpecies, E, φ, grid, dt; phase)
    push_field!(sp, E, grid, dt / 2; phase)
    push_drift!(sp, φ, grid, dt)
    push_field!(sp, E, grid, dt / 2; phase)
end

function _es_trial_rho(species, saved, φ, E, grid, dt, phase)
    ρ = zero(φ)
    for (i, sp) in enumerate(species)
        _push_stage!(sp, E, ScalarField(φ), grid, dt; phase)
        push_free!(sp, grid, dt, 2; phase)
        ρ .+= moments_at(sp, grid, _end_phase(sp, grid, dt, phase))[1]
        _restore!(sp, saved[i])
    end
    return ρ
end

"""
    step_es_charge_form!(species, sol, grid, solver::ESChargeForm, dt; phase = 0, damping_rate = 0,
                         corrections = 2, iters = nothing)

One electrostatic X–C–X step: X(dt/2); stage potential φ^m from quasineutrality at t^{n+1} with
φ^m = θφ^{n+1} + (1 − θ)φ^n, θ = 1/2 + κ dt (`damping_rate` κ), from `corrections` trial pushes (the
first at φ^p = φ^n) continued by the model operator; C (kick, E×B; for FullDK half kick, E×B, half kick) with
φ^m; X(dt/2); then φ^{n+1}
from quasineutrality of the new state (`sol.phi`, `sol.E`). `sol.phi` must hold φ^n on entry
([`es_quasineutral_phi!`](@ref) before the first step). With `solver.variable_coefficients`, the GMRES
iteration counts of the stage solves and of the final solve are pushed to `iters` if given. Returns φ^m.
"""
function step_es_charge_form!(species, sol::FieldSolution, grid::Grid, solver::ESChargeForm, dt::Real;
                              phase::Real = 0, damping_rate::Real = 0, corrections::Int = 2, iters = nothing)
    θ = 1 / 2 + damping_rate * dt
    1 / 2 <= θ <= 1 || throw(ArgumentError("θ = 1/2 + κ dt must lie in [1/2, 1], got $θ"))
    g = _es_geometry(grid)
    foreach(sp -> push_free!(sp, grid, dt, 1; phase, B = sol.B), species)
    ops = _es_operators(species, grid, solver, g, dt, phase)
    saved = map(_save, species)
    φm = copy(sol.phi.data)
    if solver.variable_coefficients
        A, P = _es_full_operators(species, grid, solver, g, dt, phase)
        F = map(J -> _es_factor(ops[J].A + ops[J].P / θ, g.k[J], g.nyquist[J]), CartesianIndices(g.k))
        Pφn = P(sol.phi.data)
        for _ in 1:corrections
            ρ = _es_trial_rho(species, saved, φm, es_field(φm, grid, g), grid, dt, phase) .+ _fixed_charge(solver)
            M(φ) = A(φ) .+ P(φ) ./ θ
            rhs = -ρ .+ A(φm) .+ (1 - θ) / θ .* Pφn
            if solver.krylov
                φm, it, _ = _es_krylov(M, rhs, F, g; solver.rtol)
                iters === nothing || push!(iters, it)
            else
                φm = _es_project(φm, g) .+ _es_precondition(rhs .- M(φm), F, g)
            end
        end
    else
        Pφn = _es_apply(ops, :P, _es_fft(sol.phi.data, g))
        for _ in 1:corrections
            ρ = _es_trial_rho(species, saved, φm, es_field(φm, grid, g), grid, dt, phase) .+ _fixed_charge(solver)
            rhs = -_es_fft(ρ, g) .+ _es_apply(ops, :A, _es_fft(φm, g)) .+ (1 - θ) / θ .* Pφn
            φ̂m = similar(rhs)
            for (J, op) in pairs(IndexCartesian(), ops)
                @views φ̂m[:, Tuple(J)...] .= _es_solve(op.A + op.P / θ, rhs[:, Tuple(J)...], g.k[J], g.nyquist[J])
            end
            φm = _es_ifft(φ̂m, g)
        end
    end
    E = es_field(φm, grid, g)
    for sp in species
        _push_stage!(sp, E, ScalarField(φm), grid, dt; phase)
        push_free!(sp, grid, dt, 2; phase)
    end
    es_quasineutral_phi!(sol, species, grid, solver; phase, iters)
    return φm
end
