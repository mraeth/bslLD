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
#   A = Σ_magnetised (dt²/2)(q²/m) ∇·(n̄ ∇)                              kick, then half a free push
#     + Σ_drift G(k∥) [(dt²/2)(q²/m) ∂∥(n̄ ∂∥) + q dt (b̂×∇n̄)·∇/b0]       parallel kick; E×B on n̄(x)
#
# with n̄(x) the species densities averaged over the Fourier axes at the predictor. The E×B term is
# the implicit E×B response; `implicit_drift = false` leaves it out of A (the response is then only in
# the trial push, i.e. lagged). G(k∥) = ⟨e^{−i k∥ v∥ dt/2}⟩ over the Maxwellian of the drift species on its
# v∥ grid (`parallel_streaming = true`; 1 otherwise): the kick and the E×B push perturb f by ∂_v F_M and
# F_M shapes, which the second X(dt/2) streams along b̂. Without it the corrections converge only slowly
# once k∥ v_t dt ~ 1. Geometry: b̂ = x̂3; x1 carries the profile (periodic, spectral
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
#
# Backends: the solve runs on the backend of the field arrays (`sol.phi.data`), the species' backend. The
# per-index n̄ operators (N1 × N1 per (k2, k3)) are built and inverted on the host from the x1 profiles and
# uploaded once per solve as one (N1, N1, modes) array, applied by a batched kernel; everything else
# (pseudo-spectral operators, projections, GMRES vectors) stays on the backend.

"""
    ESChargeForm(; w = 1, implicit_drift = true, background = nothing, variable_coefficients = false,
                 krylov = true, rtol = 1e-10, parallel_streaming = true, x_boundary = :periodic, anderson = 0)

Electrostatic quasineutral charge-form solve with the polarisation density (weight `w`) of the single
drift species and an x1-dependent profile. `implicit_drift` puts the E×B response of the drift
species into the stage operator. `background`: a fixed charge density (array of the spatial shape)
added to the species charge, e.g. immobile ions with only a drift species in `species`.
`variable_coefficients`: the densities n(x) of all axes in the stage operator and the polarisation density
(nonlinear runs), solved by GMRES to the relative residual `rtol` (`krylov = false`: one defect
correction with the averaged operator per correction instead); otherwise their average over the Fourier axes.
`parallel_streaming`: the drift species' response in the stage operator includes its parallel streaming over
the second half step (the factor G(k∥)). `x_boundary`: `:periodic`, or `:dirichlet` for walls on the first and last
x1 node (the `Mirror()` grid, Δx = L/(N−1)) with φ = 0 there. `anderson`: depth m of the Anderson acceleration of the
corrections (0: plain corrections). See [`step_es_charge_form!`](@ref).
"""
Base.@kwdef struct ESChargeForm{BG}
    w::Float64 = 1.0
    implicit_drift::Bool = true
    background::BG = nothing
    variable_coefficients::Bool = false
    krylov::Bool = true
    rtol::Float64 = 1e-10
    parallel_streaming::Bool = true
    x_boundary::Symbol = :periodic
    anderson::Int = 0
end

_to_host(a::Array) = a
_to_host(a::AbstractArray) = Array(a)

# `a` (a host array) on the backend of `like`.
_on(like::Array, a::Array) = a
_on(like, a::AbstractArray) = copyto!(similar(like, eltype(a), size(a)), a)

_fixed_charge(s::ESChargeForm) = s.background === nothing ? 0 : _to_host(s.background)
_fixed_charge(s::ESChargeForm, g) = s.background === nothing ? 0 : _on(g.Dd, _to_host(s.background))

# Periodic spectral differentiation matrix on N nodes (N even) of a period L.
function _spectral_diff_matrix(N::Int, L::Real)
    iseven(N) || throw(ArgumentError("the x1 axis needs an even number of nodes, got $N"))
    h = 2π / N
    return [i == j ? 0.0 : 0.5 * (-1)^(i - j) * cot((i - j) * h / 2) * (2π / L) for i in 1:N, j in 1:N]
end

# x1 with walls (`x_boundary = :dirichlet`). The N nodes of the `Mirror()` grid, walls on nodes 1 and N, extend to the
# periodic grid of 2(N−1) nodes, even (s = 1: n, f) or odd (s = −1: φ, zero on the walls):
# ã_j = a_j (j ≤ N), ã_{2N−j} = s a_j (2 ≤ j ≤ N−1). E_s is that extension as a 2(N−1) × N matrix, R the restriction.
function _wall_extension(N::Int, s)
    E = zeros(2N - 2, N)
    for j in 1:N
        E[j, j] = 1
    end
    for j in 2:N-1
        E[2N-j, j] = s
    end
    s < 0 && (E[:, 1] .= 0; E[:, N] .= 0)
    return E
end

struct _ESGeometry{M,K,B,DD,KK,MK,UV,UW}
    D::M          # d/dx1 on odd inputs (φ; host); periodic: the spectral matrix
    De::M         # d/dx1 on even inputs (n, f; host); periodic: = D
    Mxo::M        # 2/3 mask along x1 on odd inputs (host)
    Mxe::M        # 2/3 mask along x1 on even inputs (host)
    k::K          # (k2, k3) per Fourier index, k3 = 0 without an x3 axis (host)
    nyquist::B    # Nyquist index along x2 or x3 (host)
    nx::Int
    wall::Bool    # x1 walls (Dirichlet φ)
    kax::Vector{Vector{Float64}}  # k2 (and k3) axes (host)
    Dd::DD        # D on the backend of the field arrays
    Ded::DD       # De on the backend
    kk::KK        # (k2, k3) on the backend, shaped (1, N2[, N3])
    keep::MK      # 0 at the Nyquist indices, 1 elsewhere, shaped as kk
    u::UV         # x1 vectors projected out of φ (normalised), each shaped (N1, 1[, 1]), on the backend
    w::UW         # their weights per index: 1 or an array shaped as kk
    uh::Vector{Vector{Float64}}   # u on the host
    wh::Vector{Array{Float64}}    # w per index on the host (shape of k)
end

"""
    _es_geometry(grid, like = Float64[]; x_boundary = :periodic)

Geometry of the ES solve; its backend arrays live on the backend of `like` (an array, e.g. `sol.phi.data`).
Periodic x1: the projected vectors are the zigzag (−1)^j (all indices) and the constant (k2 = 0). Walls
(`:dirichlet`): the wall nodes, so that φ = 0 there and quasineutrality holds on the interior nodes.
"""
function _es_geometry(grid::Grid, like::AbstractArray = Float64[]; x_boundary::Symbol = :periodic)
    grid.Bdir == 3 || throw(ArgumentError("ESChargeForm needs b̂ = x̂3 (Bdir = 3)"))
    x_boundary in (:periodic, :dirichlet) || throw(ArgumentError("x_boundary must be :periodic or :dirichlet"))
    nx = length(grid.xaxes)
    2 <= nx <= 3 || throw(ArgumentError("ESChargeForm needs 2 or 3 spatial axes, got $nx"))
    N = length.(grid.xaxes)
    L = [grid.max[d] - grid.min[d] for d in 1:nx]
    kax = [collect(2π .* fftfreq(N[d], N[d] / L[d])) for d in 2:nx]
    I = CartesianIndices(Tuple(N[2:nx]))
    k = map(J -> (kax[1][J[1]], nx == 3 ? kax[2][J[2]] : 0.0), I)
    nyq = map(J -> any(d -> iseven(N[d+1]) && J[d] == N[d+1] ÷ 2 + 1, 1:nx-1), I)
    up(a) = _on(like, a)
    ms = (1, size(I)...)
    kk = ntuple(d -> up(reshape(getindex.(k, d), ms)), 2)
    keep = up(reshape(Float64.(.!nyq), ms))
    vs = (N[1], ntuple(_ -> 1, nx - 1)...)
    N1 = N[1]
    wall = x_boundary === :dirichlet
    if wall
        h = grid.delta[1]
        Ee, Eo = _wall_extension(N1, 1), _wall_extension(N1, -1)
        R = hcat(_es_diag(ones(N1)), zeros(N1, N1 - 2))     # restriction to nodes 1…N
        Dt = _spectral_diff_matrix(2N1 - 2, (2N1 - 2) * h)
        Mt = _es_mask_x(2N1 - 2)
        D, De, Mxo, Mxe = R * Dt * Eo, R * Dt * Ee, R * Mt * Eo, R * Mt * Ee
        e(j) = (v = zeros(N1); v[j] = 1; v)
        uh = [e(1), e(N1)]
        wh = Array{Float64}[ones(size(I)), ones(size(I))]
    else
        D = _spectral_diff_matrix(N1, L[1])
        De = D
        Mxo = Mxe = _es_mask_x(N1)
        uh = [[(-1.0)^j for j in 0:N1-1] ./ sqrt(N1), ones(N1) ./ sqrt(N1)]
        wh = Array{Float64}[ones(size(I)), Float64.(iszero.(getindex.(k, 1)))]
    end
    u = Tuple(up(reshape(v, vs)) for v in uh)
    w = Tuple(all(isone, x) ? 1 : up(reshape(x, ms)) for x in wh)
    return _ESGeometry(D, De, Mxo, Mxe, k, nyq, nx, wall, kax, up(D), up(De), kk, keep, u, w, uh, wh)
end

_es_geometry(grid::Grid, like::AbstractArray, solver::ESChargeForm) = _es_geometry(grid, like; solver.x_boundary)

_es_fft(a, g::_ESGeometry) = fft(complex(a), 2:g.nx)
_es_ifft(a, g::_ESGeometry) = real(ifft(a, 2:g.nx))
_es_mean_x(a, g::_ESGeometry) = vec(sum(a; dims = 2:g.nx)) ./ prod(size(a)[2:g.nx])

# ∂/∂x1 by the differentiation matrix for an input of x1-parity s (−1: φ, 1: densities; only relevant with walls):
# BLAS on host arrays, a kernel on device arrays (AMDGPU 1.x has no dense matrix product on Julia ≥ 1.13).
_es_dx1(a::Array, g::_ESGeometry, s = -1) = reshape((s < 0 ? g.D : g.De) * reshape(a, size(a, 1), :), size(a))
function _es_dx1(a, g::_ESGeometry, s = -1)
    X = reshape(a, size(a, 1), :)
    Y = similar(X)
    be = get_backend(X)
    _es_matmul_kernel!(be)(Y, s < 0 ? g.Dd : g.Ded, X; ndrange = size(X))
    KernelAbstractions.synchronize(be)
    return reshape(Y, size(a))
end

@kernel function _es_matmul_kernel!(y, D, x)
    i, c = @index(Global, NTuple)
    acc = zero(eltype(y))
    for j in axes(D, 2)
        acc += D[i, j] * x[j, c]
    end
    y[i, c] = acc
end

# ∇⊥·(n ∇⊥) and ∂∥(n ∂∥) at (k2, k3) as x1-matrices.
_es_diag(v) = [i == j ? v[i] : zero(eltype(v)) for i in eachindex(v), j in eachindex(v)]
_es_perp(D, n, k2) = D * (n .* D) - k2^2 * _es_diag(n)
_es_perp(g::_ESGeometry, n, k2) = g.De * (n .* g.D) - k2^2 * _es_diag(n)
_es_par(n, k3) = -k3^2 * _es_diag(n)

# `dl`: nothing, or (Mx, keep) for a species whose E×B push is 2/3-dealiased (`_exb_dealiased`): the E×B
# response is then P{Pn̄, Pφ} with Mx the x1 mask and `keep` whether k2 lies in the kept band.
# (Single-index form, without the streaming factor; `_es_operators` assembles all indices.)
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

# G(k∥) = ⟨e^{−i k∥ v∥ dt/2}⟩ over the drift species' Maxwellian on its v∥ grid: its parallel streaming over the
# second X(dt/2) (real: the v∥ grid is symmetric). 1 for species without one.
_streaming_factor(sp, k3, dt) = 1.0
function _streaming_factor(sp::FullDKSpecies, k3, dt)
    md = getfield(sp, :model)
    v = collect(md.grid.vaxes[1])
    w = exp.(-v .^ 2 ./ (2md.T_e))
    return real(sum(w .* cis.(-k3 * thermal_velocity(sp) .* v .* dt / 2)) / sum(w))
end
_streaming_factor(sp, k3, dt, solver::ESChargeForm) =
    is_drift(sp) && solver.parallel_streaming ? _streaming_factor(sp, k3, dt) : 1.0

_drift_species(species) = (d = filter(is_drift, collect(species)); length(d) == 1 ? only(d) :
    throw(ArgumentError("the polarisation density needs exactly one drift species, got $(length(d))")))

# Charge density q n of a species (no currents), on its backend.
_charge_density(sp, grid, phase) = moments_at(sp, grid, phase)[1]
_charge_density(sp::KineticSpecies, grid, phase) = sp.q .* compute_density(sp.dist, grid).data
_charge_density(sp::FullDKSpecies, grid, phase) =
    (md = getfield(sp, :model); sp.q .* compute_density(md.dist, md.grid).data)

_species_density(sp, grid, phase) = _charge_density(sp, grid, phase) ./ sp.q

# n̄ operators of all Fourier indices, (A, P) as host arrays (N1, N1, modes), P = w m_d ∇⊥·(n̄_d ∇⊥)/b0².
# The x1 parts that do not depend on the index (D n̄ D, the E×B diagonal) are formed once per species.
function _es_operators(species, grid, solver::ESChargeForm, g::_ESGeometry, dt, phase)
    ns = [_to_host(_es_mean_x(_species_density(sp, grid, _stage_phase(sp, grid, dt, phase)), g)) for sp in species]
    d = findfirst(is_drift, collect(species))
    md = _drift_species(species)
    cp = solver.w * md.m / grid.b0^2
    D, De, Mxo, Mxe = g.D, g.De, g.Mxo, g.Mxe
    N1 = size(D, 1)
    N2, L2 = length(grid.xaxes[2]), grid.max[2] - grid.min[2]
    parts = map(zip(species, ns)) do (sp, n)
        c = dt^2 / 2 * sp.q^2 / sp.m
        ex = !(is_drift(sp) && solver.implicit_drift) ? nothing :
             _exb_dealiased(sp) ? Mxe * _es_diag(De * (Mxe * n)) * Mxo : _es_diag(De * n)
        (sp = sp, c = c, DnD = is_drift(sp) ? nothing : (g.wall ? D : De) * (n .* D), N = _es_diag(n), ex = ex,
         dealiased = _exb_dealiased(sp))
    end
    DnDd = De * (ns[d] .* D)
    Nd = _es_diag(ns[d])
    modes = vec(g.k)
    A = zeros(ComplexF64, N1, N1, length(modes))
    P = zeros(ComplexF64, N1, N1, length(modes))
    Threads.@threads for m in eachindex(modes)
        k2, k3 = modes[m]
        Am = view(A, :, :, m)
        for p in parts
            if is_drift(p.sp)
                G = _streaming_factor(p.sp, k3, dt, solver)
                Am .+= (G * p.c * -k3^2) .* p.N
                p.ex === nothing || (p.dealiased && abs(k2) * L2 / 2π > N2 / 3 + 1e-9) ||
                    (Am .+= (G * p.sp.q * dt / grid.b0 * im * k2) .* p.ex)
            else
                Am .+= p.c .* (p.DnD .- (k2^2 + k3^2) .* p.N)
            end
        end
        P[:, :, m] .= cp .* (DnDd .- k2^2 .* Nd)
    end
    return (A = A, P = P)
end

# Solve M x = r per Fourier index, without Nyquist modes: φ has no x1-Nyquist component (−1)^j (in the
# null space of the even-N spectral D, and differentiated differently by the FFT derivatives of the
# pushes), no x1-constant at k2 = 0 (k⊥ = 0: the polarisation density does not fix φ there; with k3 ≠ 0
# this is the k⊥ → 0 limit of the ω_H branch), and no Nyquist mode along x2, x3 (φ̂ = 0 there).
# Galerkin on the complement Π⊥ = 1 − Π: (Π⊥ M Π⊥ + c Π) x = Π⊥ r gives Π x = 0 and Π⊥(M x − r) = 0.
# (Regularising only M + cΠ, with r projected, leaves a Π component in x through M and makes the
# θ-step map non-neutral: |g| up to 1.46 at Δt = 0.5, θ = 1/2, electrons only.)
function _es_factor(M, k, nyquist::Bool)
    N = size(M, 1)
    Z = [(-1)^j for j in 0:N-1] ./ sqrt(N)
    iszero(k[1]) && (Z = hcat(ones(N) ./ sqrt(N), Z))
    return _es_factor(M, Z, nyquist)
end
# With the projected vectors Z (columns) of an index; `_es_Z(g, J)` gives them for a geometry.
function _es_factor(M, Z::AbstractMatrix, nyquist::Bool)
    nyquist && return nothing
    N = size(M, 1)
    Π = Z * Z'
    Πc = _es_diag(ones(N)) - Π
    return (Πc * M * Πc + maximum(abs, M) * Π, Πc)
end
_es_solve(F, r) = F === nothing ? zero(r) : F[1] \ (F[2] * r)
_es_solve(M, r, k, nyquist::Bool) = _es_solve(_es_factor(M, k, nyquist), r)
_es_Z(g::_ESGeometry, J) = reduce(hcat, [g.uh[a] for a in eachindex(g.uh) if g.wh[a][J] != 0])

# The solution operators W = (Π⊥ M Π⊥ + cΠ)⁻¹ Π⊥ (0 at Nyquist indices) of the host operators M (N1, N1, modes),
# on the backend of the geometry: LAPACK per index on the host; on a device M is uploaded, the Galerkin matrices
# are formed there (`_es_galerkin`) and a batched Gauss–Jordan kernel solves them.
function _es_inverses(M::Array{<:Complex,3}, g::_ESGeometry)
    if g.Dd isa Array
        W = zeros(ComplexF64, size(M))
        I = vec(CartesianIndices(g.k))
        Threads.@threads for m in eachindex(I)
            Fm = _es_factor(M[:, :, m], _es_Z(g, I[m]), g.nyquist[I[m]])
            Fm === nothing || (W[:, :, m] .= Fm[1] \ Fm[2])
        end
        return W
    end
    G, Wd = _es_galerkin(_on(g.Dd, M), g)
    _es_gauss_jordan!(G, Wd)
    return Wd
end

# (Π⊥ M Π⊥ + cΠ, Π⊥) of all indices as in _es_factor, on the backend of M (N1, N1, modes), by broadcasts:
# Π = Σ_a w_a u_a u_aᵀ with the geometry's vectors u and weights w (periodic: zigzag, and the constant at k2 = 0;
# walls: the wall nodes), c = max|M| per index, and
# Π⊥MΠ⊥ = M − ΠM − MΠ + ΠMΠ. At Nyquist indices (identity, 0), so that the solve gives W = 0 there.
function _es_galerkin(M, g::_ESGeometry)
    N = size(M, 1)
    u = map(v -> reshape(v, N, 1, 1), g.u)
    ut = map(v -> reshape(v, 1, N, 1), g.u)
    w = map(x -> x isa Number ? x : reshape(x, 1, 1, :), g.w)
    keep = reshape(g.keep, 1, 1, :)
    Id = _on(g.Dd, reshape(_es_diag(ones(ComplexF64, N)), N, N, 1))
    A = eachindex(u)
    r = map(a -> sum(u[a] .* M; dims = 1), A)                 # u_aᵀ M, (1, N, modes)
    c = map(a -> sum(M .* ut[a]; dims = 2), A)                # M u_a, (N, 1, modes)
    s(a, b) = sum(r[a] .* ut[b]; dims = 2)                    # u_aᵀ M u_b, (1, 1, modes)
    Π = reduce((p, q) -> p .+ q, (w[a] .* u[a] .* ut[a] for a in A))
    F = copy(M)
    for a in A
        F .-= w[a] .* (u[a] .* r[a] .+ c[a] .* ut[a])
    end
    for a in A, b in A
        F .+= w[a] .* w[b] .* u[a] .* s(a, b) .* ut[b]
    end
    F .+= maximum(abs, M; dims = (1, 2)) .* Π
    return keep .* F .+ (1 .- keep) .* Id, keep .* (Id .- Π)
end

# Solves A[:, :, m] X = B[:, :, m] for every m in place (B ← X, A destroyed): Gauss–Jordan with partial pivoting,
# one workgroup per m. The first thread of the group finds the pivot and swaps the rows; rows are then eliminated
# in parallel.
@kernel function _es_gauss_jordan_kernel!(A, B, pv)
    m = @index(Group, Linear)
    t = @index(Local, Linear)
    T = @groupsize()[1]
    N = size(A, 1)
    R = size(B, 2)
    for k in 1:N
        if t == 1
            p = k
            amax = abs(A[k, k, m])
            for i in k+1:N
                a = abs(A[i, k, m])
                a > amax && (amax = a; p = i)
            end
            if p != k
                for j in 1:N
                    A[k, j, m], A[p, j, m] = A[p, j, m], A[k, j, m]
                end
                for j in 1:R
                    B[k, j, m], B[p, j, m] = B[p, j, m], B[k, j, m]
                end
            end
            pv[m] = A[k, k, m]
        end
        @synchronize
        for j in k+t-1:T:N
            A[k, j, m] /= pv[m]
        end
        for j in t:T:R
            B[k, j, m] /= pv[m]
        end
        @synchronize
        for i in t:T:N
            if i != k
                f = A[i, k, m]
                for j in k:N
                    A[i, j, m] -= f * A[k, j, m]
                end
                for j in 1:R
                    B[i, j, m] -= f * B[k, j, m]
                end
            end
        end
        @synchronize
    end
end

function _es_gauss_jordan!(A, B)
    be = get_backend(A)
    T = min(size(A, 1), 256)
    pv = similar(A, size(A, 3))
    _es_gauss_jordan_kernel!(be, T)(A, B, pv; ndrange = T * size(A, 3))
    KernelAbstractions.synchronize(be)
    return B
end

@kernel function _es_batched_matvec_kernel!(y, W, x)
    i, m = @index(Global, NTuple)
    acc = zero(eltype(y))
    for j in axes(W, 2)
        acc += W[i, j, m] * x[j, m]
    end
    y[i, m] = acc
end

# y[:, J] = W[:, :, J] x̂[:, J] for every Fourier index J (x̂ of shape (N1, modes...)).
function _es_batched_apply(W, x̂)
    X = reshape(x̂, size(x̂, 1), :)
    Y = similar(X)
    be = get_backend(X)
    _es_batched_matvec_kernel!(be)(Y, W, X; ndrange = size(X))
    KernelAbstractions.synchronize(be)
    return reshape(Y, size(x̂))
end

# E = −∇φ: x1 by the differentiation matrix, the Fourier axes spectrally; three components.
function es_field(φ::AbstractArray, grid::Grid, g::_ESGeometry = _es_geometry(grid, φ))
    E1 = -_es_dx1(φ, g, -1)
    φ̂ = _es_fft(φ, g)
    E2 = _es_ifft(-im .* g.kk[1] .* φ̂, g)
    E3 = g.nx == 3 ? _es_ifft(-im .* g.kk[2] .* φ̂, g) : zero(E1)
    return VectorField([ScalarField(E1), ScalarField(E2), ScalarField(E3)])
end

# --- Variable coefficients -------------------------------------------------------------------------

# ∂/∂x_d of a real spatial array of x1-parity s (−1: φ-like, 1: density-like): x1 by the differentiation matrix, the
# Fourier axes spectrally.
function _es_deriv(a, g::_ESGeometry, d, s = -1)
    d == 1 && return _es_dx1(a, g, s)
    d > g.nx && return zero(a)
    return _es_ifft(im .* g.kk[d-1] .* _es_fft(a, g), g)
end

# `sf`: x1-parity taken for the flux n ∂xφ: 1 (even, the polarisation of the drift species), −1 (zero on the walls:
# the response of magnetised species, which `Mirror()` reflects).
_es_perp(n, φ, g::_ESGeometry, sf = 1) = (∂ = (a, d, s) -> _es_deriv(a, g, d, s);
    ∂(n .* ∂(φ, 1, -1), 1, sf) .+ ∂(n .* ∂(φ, 2, -1), 2, 1))
_es_par(n, φ, g::_ESGeometry) = (∂ = (a, d, s) -> _es_deriv(a, g, d, s); ∂(n .* ∂(φ, 3, -1), 3, 1))
_es_exb(n, φ, g::_ESGeometry) = (∂ = (a, d, s) -> _es_deriv(a, g, d, s);
    ∂(n, 1, 1) .* ∂(φ, 2, -1) .- ∂(n, 2, 1) .* ∂(φ, 1, -1))

# 2/3 truncation (exb_dealias!) of a copy of a spatial array of x1-parity s; with walls on its extension (1).
_es_dealias(a, g::_ESGeometry, s) =
    g.wall ? _x1_restrict(exb_dealias!(_x1_extend(a, s)), size(a, 1)) : exb_dealias!(copy(a))

# A and P of the module header with the predictor densities n(x), as functions on real arrays (on the backend
# of the geometry). The drift species' response is filtered by G(k∥) along x3.
function _es_full_operators(species, grid, solver::ESChargeForm, g::_ESGeometry, dt, phase)
    ns = [_species_density(sp, grid, _stage_phase(sp, grid, dt, phase)) for sp in species]
    nd = ns[findfirst(is_drift, collect(species))]
    cp = solver.w * _drift_species(species).m / grid.b0^2
    G = map(species) do sp
        (g.nx == 3 && is_drift(sp) && solver.parallel_streaming) || return nothing
        _on(g.Dd, reshape([_streaming_factor(sp, k3, dt) for k3 in g.kax[2]], 1, 1, :))
    end
    stream(y, Gs) = Gs === nothing ? y : _es_ifft(Gs .* _es_fft(y, g), g)
    function A(φ)
        y = zero(φ)
        for (sp, n, Gs) in zip(species, ns, G)
            c = dt^2 / 2 * sp.q^2 / sp.m
            if is_drift(sp)
                yd = c .* _es_par(n, φ, g)
                solver.implicit_drift && (yd .+= sp.q * dt / grid.b0 .* (_exb_dealiased(sp) ?
                    _es_dealias(_es_exb(_es_dealias(n, g, 1), _es_dealias(φ, g, -1), g), g, 1) : _es_exb(n, φ, g)))
                y .+= stream(yd, Gs)
            else
                y .+= c .* (_es_par(n, φ, g) .+ _es_perp(n, φ, g, -1))
            end
        end
        return y
    end
    P(φ) = cp .* _es_perp(nd, φ, g)
    return A, P
end

# Removes the Nyquist modes and the x1-constant at k2 = 0 (see _es_solve) from a real array.
function _es_project(a, g::_ESGeometry)
    â = _es_fft(a, g)
    for (u, w) in zip(g.u, g.w)
        â .-= u .* (w .* sum(u .* â; dims = 1))
    end
    â .*= g.keep
    return _es_ifft(â, g)
end

# Per-index solve with the solution operators W of the n̄ operator (output in the projected space).
_es_precondition(r, W, g::_ESGeometry) = _es_ifft(_es_batched_apply(W, _es_fft(r, g)), g)

# Solves M x = b on the projected space: GMRES (JuliaGMG) on Q M x = Q b, right-preconditioned with
# the per-index solve with the solution operators W of the n̄ operator (whose range is the projected space).
# The Krylov vectors live on the backend of b. Returns (x, iterations, relative residual).
function _es_krylov(M, b, W, g::_ESGeometry; rtol = 1e-10, maxiter = 200)
    K!(y, r) = copyto!(y, vec(_es_precondition(reshape(r, size(b)), W, g)))
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
    g = _es_geometry(grid, sol.phi.data, solver)
    md = _drift_species(species)
    n = _to_host(_es_mean_x(_species_density(md, grid, phase), g))
    cp = solver.w * md.m / grid.b0^2
    ρ = sum(_charge_density(sp, grid, phase) for sp in species) .+ _fixed_charge(solver, g)
    Pm = zeros(ComplexF64, length(n), length(n), length(g.k))
    DnD = g.De * (n .* g.D)
    Nd = _es_diag(n)
    for (m, k) in enumerate(vec(g.k))
        Pm[:, :, m] .= cp .* (DnD .- k[1]^2 .* Nd)
    end
    W = _es_inverses(Pm, g)
    if solver.variable_coefficients
        nd = _species_density(md, grid, phase)
        φ, it, _ = _es_krylov(φ -> cp .* _es_perp(nd, φ, g), -ρ, W, g; solver.rtol)
        iters === nothing || push!(iters, it)
    else
        φ = _es_precondition(-ρ, W, g)
    end
    copyto!(sol.phi.data, φ)
    E = es_field(φ, grid, g)
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
        ρ .+= _charge_density(sp, grid, _end_phase(sp, grid, dt, phase))
        _restore!(sp, saved[i])
    end
    return ρ
end

# Anderson acceleration of the corrections φ ↦ G(φ) (one trial push and model solve each). With the residuals
# r = G(φ) − φ of the last m + 1 corrections, the next iterate is G(φ_k) − Σ_j γ_j ΔG_j with γ minimising
# ‖r_k − Σ_j γ_j ΔR_j‖ (ΔR, ΔG: differences of successive r and G). The vectors stay on the backend; the m × m normal
# equations are solved on the host. m = 0 returns G(φ_k).
struct _Anderson{V}
    m::Int
    G::Vector{V}
    R::Vector{V}
end
_Anderson(m, like) = _Anderson(m, typeof(like)[], typeof(like)[])

function _anderson!(acc::_Anderson, φ, Gφ)
    acc.m == 0 && return Gφ
    push!(acc.G, Gφ)
    push!(acc.R, Gφ .- φ)
    length(acc.G) > acc.m + 1 && (popfirst!(acc.G); popfirst!(acc.R))
    k = length(acc.G)
    k == 1 && return Gφ
    dR = [acc.R[j+1] .- acc.R[j] for j in 1:k-1]
    dG = [acc.G[j+1] .- acc.G[j] for j in 1:k-1]
    H = [sum(dR[i] .* dR[j]) for i in 1:k-1, j in 1:k-1]
    b = [sum(dR[i] .* acc.R[k]) for i in 1:k-1]
    γ = (H + 1e-14 * maximum(abs, H) * _es_diag(ones(k - 1))) \ b
    out = copy(Gφ)
    for j in 1:k-1
        out .-= γ[j] .* dG[j]
    end
    return out
end

"""
    step_es_charge_form!(species, sol, grid, solver::ESChargeForm, dt; phase = 0, damping_rate = 0,
                         corrections = 2, iters = nothing)

One electrostatic X–C–X step: X(dt/2); stage potential φ^m from quasineutrality at t^{n+1} with
φ^m = θφ^{n+1} + (1 − θ)φ^n, θ = 1/2 + κ dt (`damping_rate` κ), from `corrections` trial pushes (the
first at φ^p = φ^n) continued by the model operator, optionally Anderson-accelerated (`solver.anderson`); C (kick,
E×B; for FullDK half kick, E×B, half kick) with
φ^m; X(dt/2); then φ^{n+1}
from quasineutrality of the new state (`sol.phi`, `sol.E`). `sol.phi` must hold φ^n on entry
([`es_quasineutral_phi!`](@ref) before the first step). With `solver.variable_coefficients`, the GMRES
iteration counts of the stage solves and of the final solve are pushed to `iters` if given. Returns φ^m
(on the backend of `sol.phi`).
"""
function step_es_charge_form!(species, sol::FieldSolution, grid::Grid, solver::ESChargeForm, dt::Real;
                              phase::Real = 0, damping_rate::Real = 0, corrections::Int = 2, iters = nothing)
    θ = 1 / 2 + damping_rate * dt
    1 / 2 <= θ <= 1 || throw(ArgumentError("θ = 1/2 + κ dt must lie in [1/2, 1], got $θ"))
    g = _es_geometry(grid, sol.phi.data, solver)
    foreach(sp -> push_free!(sp, grid, dt, 1; phase, B = sol.B), species)
    ops = _es_operators(species, grid, solver, g, dt, phase)
    W = _es_inverses(ops.A .+ ops.P ./ θ, g)
    saved = map(_save, species)
    φn = copy(sol.phi.data)
    φm = copy(φn)
    ρfix = _fixed_charge(solver, g)
    acc = _Anderson(solver.anderson, φm)
    if solver.variable_coefficients
        A, P = _es_full_operators(species, grid, solver, g, dt, phase)
        Pφn = P(φn)
        for _ in 1:corrections
            ρ = _es_trial_rho(species, saved, φm, es_field(φm, grid, g), grid, dt, phase) .+ ρfix
            M(φ) = A(φ) .+ P(φ) ./ θ
            rhs = -ρ .+ A(φm) .+ (1 - θ) / θ .* Pφn
            if solver.krylov
                Gφ, it, _ = _es_krylov(M, rhs, W, g; solver.rtol)
                iters === nothing || push!(iters, it)
            else
                Gφ = _es_project(φm, g) .+ _es_precondition(rhs .- M(φm), W, g)
            end
            φm = _anderson!(acc, φm, Gφ)
        end
    else
        Ad, Pd = _on(g.Dd, ops.A), _on(g.Dd, ops.P)
        Pφn = _es_batched_apply(Pd, _es_fft(φn, g))
        for _ in 1:corrections
            ρ = _es_trial_rho(species, saved, φm, es_field(φm, grid, g), grid, dt, phase) .+ ρfix
            rhs = -_es_fft(ρ, g) .+ _es_batched_apply(Ad, _es_fft(φm, g)) .+ (1 - θ) / θ .* Pφn
            φm = _anderson!(acc, φm, _es_ifft(_es_batched_apply(W, rhs), g))
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
