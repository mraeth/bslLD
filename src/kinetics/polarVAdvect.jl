# Advection on the polar velocity grid (`PolarVGrid`): x free streaming (exact Fourier
# multiplier) and two E-kick methods, `PolarSL` (K1, semi-Lagrangian) and `PolarTaylor`
# (K2, Eulerian, mass conserving). Conventions are those of advectorCart.jl, rotating
# frame with angle θ = electric_acceleration_scale(sp) * simTime.phase:
#
#   x step:  f(x, v) <- f(x - τ vth (vx cosθ + vy sinθ), v) = f(x - τ vth r cos(φ - θ), v)
#   v kick:  f(x, v) <- f(x, v - δ),  δ = τ (q/√m) R(θ) E(x),  R(θ) = [cosθ -sinθ; sinθ cosθ]
#
# with τ = simTime.dt * simTime.fraction_dt.

using FFTW: plan_rfft, plan_irfft
import AbstractFFTs: mul!

const SpeciesPolarV = Species{PDT,<:DistributionGridImpl{1,2,PolarV}} where {PDT}

# ── x advection ──────────────────────────────────────────────────────────────

struct _PolarXWorkspace{P,IP,B,K}
    plan::P
    iplan::IP
    buf::B
    k::K
end

function _polar_x_workspace(grid::PolarVGrid, data::AbstractMatrix{T}) where {T}
    return _polar_buffer!(grid, (:x, size(data), T)) do
        nx = size(data, 1)
        buf = Matrix{Complex{T}}(undef, nx ÷ 2 + 1, size(data, 2))
        tmp = similar(data)
        plan = plan_rfft(tmp, 1; flags = FFTW.ESTIMATE | FFTW.UNALIGNED)
        iplan = plan_irfft(buf, nx, 1; flags = FFTW.ESTIMATE | FFTW.UNALIGNED)
        L = nx * grid.xgrid.delta[1]
        k = T[2pi * (i - 1) / L for i = 1:(nx÷2+1)]
        _PolarXWorkspace(plan, iplan, buf, k)
    end::_PolarXWorkspace
end

"""
    advectX!(sp, grid::PolarVGrid, simTime)

Free streaming over `simTime.dt * simTime.fraction_dt` in the rotating frame of phase
`simTime.phase`: exact Fourier multiplier `exp(-i k τ vth r cos(φ - θ))` per velocity
node, `θ = (q/√m) phase` (same convention as the Cartesian `advectX!`).
"""
function advectX!(sp::SpeciesPolarV, grid::PolarVGrid, simTime::SimulationTime)
    f = sp.dist.data
    T = eltype(f)
    ws = _polar_x_workspace(grid, f)
    th = T(electric_acceleration_scale(sp) * simTime.phase)
    tau = T(_effective_dt(simTime))
    vth = T(thermal_velocity(sp))
    mul!(ws.buf, ws.plan, f)
    buf, k = ws.buf, ws.k
    vr, vphi = grid.vr, grid.vphi
    @threads for iv = 1:grid.Nv
        c = tau * vth * vr[iv] * cos(vphi[iv] - th)
        @inbounds for ik in eachindex(k)
            buf[ik, iv] *= cis(-k[ik] * c)
        end
    end
    mul!(f, ws.iplan, buf)
    return nothing
end

# ── Kick displacement ───────────────────────────────────────────────────────

@inline function _kick_delta(e::VectorField, ix, th, scale)
    ex = e[1].data[ix]
    ey = length(e) >= 2 ? e[2].data[ix] : zero(ex)
    c, s = cos(th), sin(th)
    return scale * (c * ex - s * ey), scale * (s * ex + c * ey)
end

# Chunks of the x range, one workspace each (threads over x).
_nchunks(nx) = max(1, min(nx, Threads.nthreads()))
_chunk(c, nc, nx) = (((c - 1) * nx) ÷ nc + 1):((c * nx) ÷ nc)   # (`div` is PlasmaCore's here)

# ── K1: semi-Lagrangian kick ────────────────────────────────────────────────

"""
    PolarSL(W = 8)

Semi-Lagrangian E-kick on `PolarVGrid` (scheme K1): f̂_m(s_j) by FFT in φ, departure
point v - δ per node, centred Lagrange interpolation of even width `W` in s with
parity ghosts f̂_m(s_{1-j}) = (-1)^m f̂_m(s_j) and zeros beyond the outer edge, then the
Fourier sum at the departure angle. Cost ∝ Nr Nφ (Nφ/2) W per x. Not conservative.
"""
struct PolarSL{W}
    function PolarSL{W}() where {W}
        W isa Int && W >= 2 && iseven(W) ||
            throw(ArgumentError("PolarSL stencil width must be an even Int >= 2, got $W"))
        return new{W}()
    end
end
PolarSL(W::Integer = 8) = PolarSL{Int(W)}()

struct _K1Workspace{T,P}
    g::Matrix{T}                 # (Ns, Nφ) slice
    G::Matrix{Complex{T}}        # (Ns, M) rfft along φ
    Fext::Matrix{Complex{T}}     # (M, Ns + 2H) modes with parity ghosts / outer zeros
    acc::Vector{Complex{T}}      # (M)
    out::Matrix{T}               # (Ns, Nφ) result
    plan::P
end

function _k1_workspaces(grid::PolarVGrid, ::Type{T}, ::Val{W}, nc) where {T,W}
    nphi = _require_uniform(grid)
    return _polar_buffer!(grid, (:k1, T, W, nc)) do
        Ns = grid.Ns
        M = nphi ÷ 2 + 1
        H = W
        map(1:nc) do _
            g = Matrix{T}(undef, Ns, nphi)
            G = Matrix{Complex{T}}(undef, Ns, M)
            plan = plan_rfft(g, 2; flags = FFTW.ESTIMATE)
            _K1Workspace(
                g,
                G,
                zeros(Complex{T}, M, Ns + 2H),
                zeros(Complex{T}, M),
                Matrix{T}(undef, Ns, nphi),
                plan,
            )
        end
    end::Vector
end

# Modes with real-reconstruction factors c_m (2 for 0 < m < Nφ/2, else 1), parity ghosts
# on the left, zeros on the right. Column H + n holds node n (n = 1-H .. Ns+H).
function _fill_modes!(Fext, G, Ns, nphi, H)
    M = size(G, 2)
    @inbounds for n = 1:Ns, m = 1:M
        cm = (m == 1 || (iseven(nphi) && m == M)) ? 1 : 2
        Fext[m, H+n] = G[n, m] * (cm / nphi)
    end
    # outer zeros first: if Ns < H, the ghost of a node beyond Ns must read a zero
    @inbounds for n = (Ns+1):(Ns+H), m = 1:M
        Fext[m, H+n] = 0
    end
    @inbounds for n = 0:-1:(1-H), m = 1:M
        Fext[m, H+n] = isodd(m - 1) ? -Fext[m, H+1-n] : Fext[m, H+1-n]
    end
    return Fext
end

function _k1_slice!(ws::_K1Workspace{T}, grid::PolarVGrid, dx, dy, ::Val{W}) where {T,W}
    Ns = grid.Ns
    nphi = size(ws.g, 2)
    M = size(ws.G, 2)
    H = W
    mul!(ws.G, ws.plan, ws.g)
    Fext = _fill_modes!(ws.Fext, ws.G, Ns, nphi, H)
    acc = ws.acc
    inv_h = 1 / grid.ds
    mp = grid.map
    @inbounds for l = 1:nphi
        cphi, sphi = cospi(2 * (l - 1) / nphi), sinpi(2 * (l - 1) / nphi)
        for j = 1:Ns
            x = grid.r[j] * cphi - dx
            y = grid.r[j] * sphi - dy
            rd = hypot(x, y)
            t = inv_radius(mp, rd) * inv_h + T(0.5)   # fractional node index (node n at t = n)
            i0 = floor(Int, t)
            if i0 + 1 - W ÷ 2 > Ns
                ws.out[j, l] = 0
                continue
            end
            w = lagrange_weights(Val(W), 1 - (t - i0), T)  # node i0+1, alpha = 1 - frac
            c0 = H + i0 - W ÷ 2                               # column of stencil node m=1 minus 1
            for m = 1:M
                acc[m] = 0
            end
            for q = 1:W
                wq = w[q]
                col = c0 + q
                @simd for m = 1:M
                    acc[m] += wq * Fext[m, col]
                end
            end
            z = rd > 0 ? complex(x / rd, y / rd) : one(Complex{T})
            zm = one(Complex{T})
            val = zero(T)
            for m = 1:M
                a = acc[m]
                val += real(a) * real(zm) - imag(a) * imag(zm)
                zm *= z
            end
            ws.out[j, l] = val
        end
    end
    return ws.out
end

# ── K2: Eulerian conservative kick (Taylor in time) ─────────────────────────

"""
    PolarTaylor(W = 8; order = 4, cfl = 1.0)

Eulerian E-kick on `PolarVGrid` (scheme K2). Solves ∂_t f + δ·∇_v f = 0 on t ∈ [0, 1]
in angular Fourier space, where with δ = δx + iδy

    (δ·∇f)_m = ½ δ̄ (F'_{m-1} - (m-1) F_{m-1}/r) + ½ δ (F'_{m+1} + (m+1) F_{m+1}/r),

F' = (1/r'(s)) ∂_s F by centred finite differences of order `W` in s (parity ghosts,
zeros beyond the edge), F_{-1} = conj(F_1). Ring j carries the modes m < Nφ_j/2 (ragged
Nφ allowed, Nφ_j non-decreasing in j) and its Nyquist mode is left unchanged. Where an
inner ring does not carry mode m, the value F_m used by the radial stencil and by the
m∓1 coupling terms is (`extrap`): 0 → zero; 1 → F_m(r_{j0}) (r/r_{j0})^m from the
innermost carrying ring j0; 2 (default) → r^m (a₀ + a₁ r²) fitted to rings j0, j0+1
(the regular small-r form of an angular mode of a smooth f). It is never stored or
evolved there: the stored truncation error is the dropped F_m itself.

Time: Taylor series of `order` ∈ {3, 4, 7, 8} (the orders whose stability region
contains a segment of the imaginary axis, |y| ≤ 1.73, 2.83, 1.77, 3.40) on n sub-steps,
n = ⌈|δ| λ / cfl⌉, λ = σ/(Δs min r') + max_j m_j/r_j (σ ≥ max |FD symbol|);
`cfl` must not exceed the order's limit. n is chosen per x slice from the local |δ|;
slices with δ = 0 are skipped (exact no-op). The ring's Nyquist mode (even Nφ_j) is
frozen, unlike in `PolarSL` which evaluates it.

Mass. The semi-discrete scheme does not conserve the quadrature mass
M = Σ_j ω_j F_{0,j} (ω_j = 2π Δs r_j r'_j (1+γ_j)) by itself: the node form of the m = 0
equation, ∂_t F_0 = -Re(δ(F_1' + F_1/r)), does not telescope against ω (Σ_j ω_j L_jn ≠ 0 also
in the interior for a non-uniform map), and the edge stencil lets mass through r_max. The
mass rate is split at an interior face J+1/2 (J = min(K+6, Ns/2), K = quadrature order)
using the conservative face flux Φ̂_{J+1/2} of u = r F_1 (Shu-type flux of the centred
stencil; for K3 the spectral flux function), which carries the exact flux through that
face to the order of the scheme:
- pole defect = Σ_{j≤J} ω_j ∂_t F_{0,j} + 2π Re(δ Φ̂_{J+1/2}) — truncation-size, born at the
  pole closure; removed by `massfix` (`:pole`, default: along the smooth bump
  (1 - (r/r_{J+1/2})²)² inside the face — a single ring K would amplify the O(Δs^W)
  defect by 1/ω_K ~ Δs⁻² and cost two orders; `:profile`: along exp(-r²/2); `:none`: no
  correction at all);
- edge-side rate = the rest, i.e. the flux through r_max plus the outer-region
  truncation defect; with `edge = :wall` (default) it is held back on the outermost ring
  (mass conserved to round-off; r_max is a closed wall), with `edge = :outflow` it leaves.
Neither is ever moved onto the pole, so mass at r_max does not create pole errors.
`polar_mass_diagnostics(grid)` returns the accumulated mass change, pole correction and
edge correction. This is a bookkeeping correction, not a summation-by-parts operator.
Cost ∝ order · Σ_j (Nφ_j/2) (W + 6) per x per sub-step.
"""
struct PolarTaylor{W}
    order::Int
    cfl::Float64
    massfix::Symbol
    extrap::Int
    edge::Symbol
    function PolarTaylor{W}(
        order::Integer,
        cfl::Real,
        massfix::Symbol,
        extrap::Integer,
        edge::Symbol,
    ) where {W}
        W isa Int && W >= 2 && iseven(W) ||
            throw(ArgumentError("PolarTaylor FD order must be an even Int >= 2, got $W"))
        _check_taylor(order, cfl)
        _check_massfix_extrap(massfix, extrap, edge)
        return new{W}(Int(order), Float64(cfl), massfix, Int(extrap), edge)
    end
end

# Imaginary-axis stability limit y_max of the Taylor polynomial Σ_{k≤p} z^k/k!. Orders
# 1, 2, 5, 6 have none (|R(iy)| > 1 for all small y ≠ 0).
const _TAYLOR_YMAX = Dict(3 => 1.73, 4 => 2.83, 7 => 1.77, 8 => 3.40)

function _check_taylor(order, cfl)
    haskey(_TAYLOR_YMAX, order) || throw(
        ArgumentError(
            "Taylor order $order is unstable on the imaginary axis; use 3, 4, 7 or 8",
        ),
    )
    0 < cfl <= _TAYLOR_YMAX[order] || throw(
        ArgumentError("cfl = $cfl exceeds the stability limit $(_TAYLOR_YMAX[order]) of order $order"),
    )
    return nothing
end

function _check_massfix_extrap(massfix, extrap, edge)
    massfix in (:pole, :profile, :none) ||
        throw(ArgumentError("massfix must be :pole, :profile or :none"))
    extrap in (0, 1, 2) || throw(ArgumentError("extrap must be 0, 1 or 2"))
    edge in (:wall, :outflow) || throw(ArgumentError("edge must be :wall or :outflow"))
    return nothing
end
PolarTaylor(
    W::Integer = 8;
    order::Integer = 4,
    cfl::Real = 1.0,
    massfix::Symbol = :pole,
    extrap::Integer = 2,
    edge::Symbol = :wall,
) = PolarTaylor{Int(W)}(order, cfl, massfix, extrap, edge)

# Centred first-derivative weights c_k, k = 1..W/2 (c_{-k} = -c_k), order W.
function _central_d1(W::Int)
    q = W ÷ 2
    # Σ_k c_k 2 k^{2p-1} = δ_{p1}, p = 1..q
    A = Rational{BigInt}[2 * big(k)^(2p - 1) for p = 1:q, k = 1:q]
    b = Rational{BigInt}[p == 1 ? 1 : 0 for p = 1:q]
    return Float64.(_rational_solve(A, b))
end

# Per-ring real FFT of length n (shared by all rings with the same Nφ_j).
struct _RingFFT{T,P,IP}
    g::Vector{T}
    G::Vector{Complex{T}}
    plan::P
    iplan::IP
end

function _RingFFT(::Type{T}, n::Int) where {T}
    g = Vector{T}(undef, n)
    G = Vector{Complex{T}}(undef, n ÷ 2 + 1)
    return _RingFFT(
        g,
        G,
        plan_rfft(g; flags = FFTW.ESTIMATE),
        plan_irfft(G, n; flags = FFTW.ESTIMATE),
    )
end

struct _K2Workspace{T,R}
    rings::Vector{R}           # ring j -> FFT of length Nφ_j
    U::Matrix{Complex{T}}      # (Ns, Mmax) modes 0..Mmax-1, zero beyond each ring's cutoff
    A1::Matrix{Complex{T}}     # Taylor term
    A2::Matrix{Complex{T}}     # Taylor term (next)
    D::Matrix{Complex{T}}      # radial derivative of each mode
    nyq::Vector{Complex{T}}    # frozen Nyquist coefficient per ring (raw rfft value)
    u::Vector{Complex{T}}      # u = r F_1 (conservative m = 0 flux)
    du::Vector{Complex{T}}     # ∂_s u
    m0::Vector{T}              # F_0 before the kick (diagnostics)
    diag::Vector{T}            # accumulated [mass change (outflow), |pole correction|]
end

struct _K2Static{T}
    c::Vector{T}          # FD weights
    invrh::Vector{T}      # 1 / (r'(s_j) h)
    invr::Vector{T}       # 1 / r_j
    mc::Vector{Int}       # evolved modes per ring (m < mc[j])
    prof::Vector{T}       # mass-fix profile, Σ ω prof = 1 (zeros for :none)
    omega::Vector{T}      # ring weights
    sigma::T              # 2 Σ |c_k| (bound of the FD symbol)
    angmax::T             # max_j mc_j / r_j
    j0::Vector{Int}       # per mode column: innermost ring carrying it
    ea::Matrix{T}         # F_m(r_j) ≈ ea[j,col] F_m(r_j0) + eb[j,col] F_m(r_j0+1), j < j0
    eb::Matrix{T}
    r::Vector{T}          # r_j
    invrrp::Vector{T}     # 1 / (r_j r'_j)
    hgam::Vector{T}       # Δs γ_j (endpoint corrections of the quadrature)
    fix::Bool             # massfix != :none
    wall::Bool            # edge = :wall (edge flux held back on the outermost ring)
    jmid::Int             # interior face J+1/2 splitting pole-side and edge-side defects
end

# Extrapolation weights for the modes an inner ring does not carry (see `PolarTaylor`).
function _k2_extrap_tables(grid::PolarVGrid, ::Type{T}, mc, extrap) where {T}
    Ns = grid.Ns
    Mmax = maximum(mc)
    issorted(mc) || throw(ArgumentError("ragged K2 needs Nφ_j non-decreasing in j"))
    j0 = [something(findfirst(>=(col), mc), Ns + 1) for col = 1:Mmax]
    ea = zeros(T, Ns, Mmax)
    eb = zeros(T, Ns, Mmax)
    extrap == 0 && return j0, ea, eb
    r = grid.r
    for col = 1:Mmax, j = 1:min(j0[col] - 1, Ns)
        m = col - 1
        a = j0[col]
        if extrap == 1 || a + 1 > Ns || mc[a+1] < col
            ea[j, col] = (r[j] / r[a])^m
        else
            b = a + 1
            x, x0, x1 = r[j]^2, r[a]^2, r[b]^2
            ea[j, col] = (r[j] / r[a])^m * (x1 - x) / (x1 - x0)
            eb[j, col] = (r[j] / r[b])^m * (x - x0) / (x1 - x0)
        end
    end
    return j0, ea, eb
end

# Fill the not-carried inner-ring modes of a stage array from the carried ones.
@inline function _k2_fill_extrap!(U, st::_K2Static)
    Ns, Mc = size(U)
    @inbounds for col = 1:Mc
        a = st.j0[col]
        a > 1 || continue
        if a > Ns
            for j = 1:Ns
                U[j, col] = 0
            end
            continue
        end
        b = min(a + 1, Ns)
        for j = 1:(a-1)
            U[j, col] = st.ea[j, col] * U[a, col] + st.eb[j, col] * U[b, col]
        end
    end
    return U
end

_ring_modes(n) = iseven(n) ? n ÷ 2 : n ÷ 2 + 1

# Face J+1/2 of the pole-side mass bookkeeping: just outside the pole closure (the K
# corrected weights plus a stencil), and at most half the radial grid, so that defects
# born at the outer edge are never attributed to the pole.
_bookkeeping_face(grid::PolarVGrid) =
    clamp(count(!iszero, grid.gamma) + 6, 1, max(1, grid.Ns ÷ 2))

function _k2_static(
    grid::PolarVGrid,
    ::Type{T},
    ::Val{W},
    massfix::Symbol,
    extrap::Int,
    edge::Symbol,
) where {T,W}
    return _polar_buffer!(grid, (:k2static, T, W, massfix, extrap, edge)) do
        Ns = grid.Ns
        c = T.(_central_d1(W))
        om = T.(grid.ring_weights)
        if massfix === :profile
            e = T[exp(-grid.r[j]^2 / 2) for j = 1:Ns]
            prof = e ./ sum(om .* e)
        elseif massfix === :pole
            # smooth bump inside the bookkeeping disk r < r_{J+1/2}: Σ ω prof = 1 with an
            # O(1) normalisation (a single ring K would amplify the O(Δs^W) defect by
            # 1/ω_K ~ 1/Δs² and cost two orders of accuracy)
            J = _bookkeeping_face(grid)
            rJ = radius(grid.map, J * grid.ds)
            b = T[j <= J ? (1 - (grid.r[j] / rJ)^2)^2 : zero(T) for j = 1:Ns]
            prof = b ./ sum(om .* b)
        else
            prof = zeros(T, Ns)
        end
        mc = [_ring_modes(n) for n in grid.nphi]
        invr = T[1 / grid.r[j] for j = 1:Ns]
        j0, ea, eb = _k2_extrap_tables(grid, T, mc, extrap)
        _K2Static(
            c,
            T[1 / (grid.drds[j] * grid.ds) for j = 1:Ns],
            invr,
            mc,
            prof,
            om,
            T(2 * sum(abs, c)),
            maximum(mc .* invr),
            j0,
            ea,
            eb,
            T.(grid.r),
            T[1 / (grid.r[j] * grid.drds[j]) for j = 1:Ns],
            T[grid.ds * grid.gamma[j] for j = 1:Ns],
            massfix !== :none,
            edge === :wall,
            _bookkeeping_face(grid),
        )
    end::_K2Static{T}
end

function _k2_workspaces(grid::PolarVGrid, ::Type{T}, nc) where {T}
    return _polar_buffer!(grid, (:k2, T, nc)) do
        Ns = grid.Ns
        Mmax = maximum(_ring_modes, grid.nphi)
        map(1:nc) do _
            ffts = Dict(n => _RingFFT(T, n) for n in unique(grid.nphi))
            _K2Workspace(
                [ffts[n] for n in grid.nphi],
                zeros(Complex{T}, Ns, Mmax),
                zeros(Complex{T}, Ns, Mmax),
                zeros(Complex{T}, Ns, Mmax),
                zeros(Complex{T}, Ns, Mmax),
                zeros(Complex{T}, Ns),
                zeros(Complex{T}, Ns),
                zeros(Complex{T}, Ns),
                zeros(T, Ns),
                zeros(T, 3),
            )
        end
    end::Vector
end

"""
    polar_mass_diagnostics(grid::PolarVGrid; reset = false)
        -> (; outflow, pole_correction, edge_correction)

Accumulated over all `PolarTaylor`/`PolarSpectral` kicks on `grid` since the last reset,
summed over x nodes (Σ_x of ∫ f dv changes, as `sum(compute_density(...).data)`):
`outflow` is the net mass change of the kicks (zero to round-off with `edge = :wall`);
`pole_correction` and `edge_correction` estimate (first-stage rate × sub-step) the
absolute mass moved by the pole-defect correction and held back at the wall.
"""
function polar_mass_diagnostics(grid::PolarVGrid; reset::Bool = false)
    out, pole, edge = 0.0, 0.0, 0.0
    lock(_polar_cache_lock) do
        for (key, val) in grid.cache
            key isa Tuple && first(key) === :k2 || continue
            for ws in val
                out += ws.diag[1]
                pole += ws.diag[2]
                edge += ws.diag[3]
                reset && fill!(ws.diag, 0)
            end
        end
    end
    return (; outflow = out, pole_correction = pole, edge_correction = edge)
end

# D[:, k] = ∂_r F_k  (k = 0..Mc-1 stored at column k+1), parity ghosts and outer zeros.
@inline function _k2_ddr!(D, U, st::_K2Static{T}, ::Val{W}) where {T,W}
    Ns, Mc = size(U)
    q = W ÷ 2
    c = st.c
    @inbounds for col = 1:Mc
        par = isodd(col - 1) ? -one(T) : one(T)
        for j = 1:Ns
            col > st.mc[j] + 1 && continue   # never read by _k2_rhs! (ragged rings)
            a = zero(Complex{T})
            for k = 1:q
                jp = j + k
                jm = j - k
                up = jp <= Ns ? U[jp, col] : zero(Complex{T})
                # ghost 1-jm beyond the last ring (Ns < W/2) is outside the domain: zero
                um = jm >= 1 ? U[jm, col] : (1 - jm <= Ns ? par * U[1-jm, col] : zero(Complex{T}))
                a += c[k] * (up - um)
            end
            D[j, col] = a * st.invrh[j]
        end
    end
    return D
end

# Even extension of u (u_{1-n} = u_n, zero beyond Ns).
@inline _ueven(u, n, Ns) =
    n >= 1 ? (n <= Ns ? u[n] : zero(eltype(u))) : (1 - n <= Ns ? u[1-n] : zero(eltype(u)))

# Conservative face flux Φ̂_{J+1/2} (J = st.jmid) of the even vector u = r F_1 for the
# centred stencil: Φ̂_{j+1/2} = Σ_k c_k Σ_{i=1-k}^{k} u_{j+i}, so that the centred difference
# is Φ̂_{j+1/2} - Φ̂_{j-1/2} exactly (Shu's conservative finite-difference form).
function _k2_face_flux(u, st::_K2Static{T}, ::Val{W}) where {T,W}
    Ns = length(u)
    J = st.jmid
    a = zero(Complex{T})
    @inbounds for k = 1:(W÷2), i = (1-k):k
        a += st.c[k] * _ueven(u, J + i, Ns)
    end
    return a
end

# out = -(δ·∇) U   (modes 0..Mc-1); m = 0 in conservative form with only the pole
# defect corrected (see `PolarTaylor`). Returns the pole-defect mass rate removed.
# `op` selects the radial derivative: Val(W) (centred FD, K2) or a `_SpecDeriv` (K3).
function _k2_rhs!(out, U, D, st::_K2Static{T}, dlt::Complex{T}, op, u, du) where {T}
    Ns, Mc = size(U)
    _k2_fill_extrap!(U, st)      # U is a stage copy; the state itself is never filled
    _k2_ddr!(D, U, st, op)
    hd = dlt / 2
    hdb = conj(dlt) / 2
    # m = 0 node values: ∂_t F_0 = -Re(δ (F_1' + F_1/r))  (F_1/r exact on the nodes)
    mdot_in = zero(T)
    mdot = zero(T)
    @inbounds for j = 1:Ns
        out[j, 1] = Mc >= 2 ? -real(dlt * (D[j, 2] + st.invr[j] * U[j, 2])) : zero(T)
        w = st.omega[j] * real(out[j, 1])
        mdot += w
        j <= st.jmid && (mdot_in += w)
    end
    pole_rate = zero(T)
    edge_rate = zero(T)
    if st.fix
        # mass bookkeeping across the interior face J+1/2 (J = st.jmid): the inner disk
        # gains -2π Re(δ Φ̂_{J+1/2}) exactly; the excess is the pole defect -> ring K
        @inbounds for j = 1:Ns
            u[j] = Mc >= 2 ? st.r[j] * U[j, 2] : zero(Complex{T})
        end
        phi = _k2_face_flux(u, st, op)
        pole_rate = mdot_in + 2T(pi) * real(dlt * phi)
        @inbounds for j = 1:Ns
            out[j, 1] -= pole_rate * st.prof[j]
        end
        if st.wall        # edge = :wall: hold the outer-side rate back on the last ring
            edge_rate = mdot - pole_rate
            @inbounds out[Ns, 1] -= edge_rate / st.omega[Ns]
        end
    end
    @inbounds for col = 2:Mc
        m = col - 1
        for j = 1:Ns
            if col > st.mc[j]          # beyond this ring's mode cutoff
                out[j, col] = 0
                continue
            end
            ir = st.invr[j]
            lo = D[j, col-1] - (m - 1) * ir * U[j, col-1]        # F_{m-1} term
            hi = col < Mc ? D[j, col+1] + (m + 1) * ir * U[j, col+1] : zero(Complex{T})
            out[j, col] = -(hdb * lo + hd * hi)
        end
    end
    return pole_rate, edge_rate
end

function _k2_slice!(
    ws::_K2Workspace{T},
    st::_K2Static{T},
    grid::PolarVGrid,
    f::AbstractMatrix{T},
    ix::Int,
    dx,
    dy,
    method,
    op,
) where {T}
    Ns = grid.Ns
    U = ws.U
    # gather: per-ring rfft, normalised coefficients F_m = G_m / Nφ_j
    @inbounds for j = 1:Ns
        rf = ws.rings[j]
        n = grid.nphi[j]
        o = grid.offset[j]
        for l = 1:n
            rf.g[l] = f[ix, o+l]
        end
        mul!(rf.G, rf.plan, rf.g)
        for col = 1:size(U, 2)
            U[j, col] = col <= st.mc[j] ? rf.G[col] / n : zero(Complex{T})
        end
        ws.nyq[j] = iseven(n) ? rf.G[n÷2+1] : zero(Complex{T})
    end
    dlt = complex(T(dx), T(dy))
    lam = abs(dlt) * (_radial_sigma(op, st) * maximum(st.invrh) + st.angmax)
    nsub = max(1, ceil(Int, lam / method.cfl))
    tau = one(T) / nsub
    A1, A2, D = ws.A1, ws.A2, ws.D
    @inbounds for j = 1:Ns
        ws.m0[j] = real(U[j, 1])
    end
    pole = zero(T)
    edge = zero(T)
    for _ = 1:nsub
        copyto!(A1, U)
        for k = 1:method.order
            pr, er = _k2_rhs!(A2, A1, D, st, dlt, op, ws.u, ws.du)
            k == 1 && (pole += abs(pr) * tau; edge += abs(er) * tau)
            fac = tau / k
            @inbounds for i in eachindex(A2)
                A1[i] = fac * A2[i]
                U[i] += A1[i]
            end
        end
    end
    dm = zero(T)
    @inbounds for j = 1:Ns
        dm += st.omega[j] * (real(U[j, 1]) - ws.m0[j])
    end
    ws.diag[1] += dm
    ws.diag[2] += pole
    ws.diag[3] += edge
    # scatter: per-ring inverse rfft, Nyquist coefficient unchanged
    @inbounds for j = 1:Ns
        rf = ws.rings[j]
        n = grid.nphi[j]
        o = grid.offset[j]
        for col = 1:st.mc[j]
            rf.G[col] = U[j, col] * n
        end
        iseven(n) && (rf.G[n÷2+1] = ws.nyq[j])
        mul!(rf.g, rf.iplan, rf.G)
        for l = 1:n
            f[ix, o+l] = rf.g[l]
        end
    end
    return f
end

# ── K3: K2 with a spectral radial derivative (DCT/DST in s) ─────────────────

"""
    PolarSpectral(; order = 4, cfl = 1.0, massfix = :pole, extrap = 2)

Eulerian E-kick K3: identical to `PolarTaylor` (K2) except that the radial derivative
∂_s F_m is spectral. On the offset nodes s_j = (j-1/2)Δs the coefficient F_m(s) has exact
parity (-1)^m about s = 0, i.e. it is half-sample symmetric (m even) or antisymmetric
(m odd) there. Even modes: DCT-II → multiply by -κ_k → DST-III; odd modes: DST-II →
multiply by κ_k → DCT-III, κ_k = πk/S, S = Ns Δs (FFTW REDFT10/RODFT01 and RODFT10/REDFT01,
applied to real and imaginary parts). The same transforms imply a reflection at the outer
edge s = S (even for even m, odd for odd m), harmless while f(r_max) ≈ 0. The highest
sine/cosine term (k = Ns) is dropped (its derivative is undefined on the nodes). Chain
rule d/dr = (1/r'(s)) d/ds. Sub-steps per x slice from the local
|δ| (π/(Δs min r') + max_j m_j/r_j) ≤ `cfl`; `order` ∈ {3, 4, 7, 8} with the cfl limits of
`PolarTaylor`; δ = 0 slices skipped; Nyquist mode frozen. The conservative m = 0 form
mass bookkeeping uses the spectral flux function Ψ(s) = a_0 + Σ_k a_k σ_k cos κ_k s,
σ_k = (πk/2Ns)/sin(πk/2Ns), whose midpoint differences equal Δs ∂_s u on the nodes
exactly, at the bookkeeping face. Being global, K3 is polluted everywhere (Gibbs) when f
is not ≈ 0 at r_max; use K1/K2 for such cases.
Mass fix and ragged-ring handling as in `PolarTaylor` (ragged is supported by the code
path but tested only for uniform Nφ).
"""
struct PolarSpectral
    order::Int
    cfl::Float64
    massfix::Symbol
    extrap::Int
    edge::Symbol
    function PolarSpectral(order::Integer, cfl::Real, massfix::Symbol, extrap::Integer, edge::Symbol)
        _check_taylor(order, cfl)
        _check_massfix_extrap(massfix, extrap, edge)
        return new(Int(order), Float64(cfl), massfix, Int(extrap), edge)
    end
end
PolarSpectral(;
    order::Integer = 4,
    cfl::Real = 1.0,
    massfix::Symbol = :pole,
    extrap::Integer = 2,
    edge::Symbol = :wall,
) = PolarSpectral(order, cfl, massfix, extrap, edge)

struct _SpecDeriv{T,P1,P2,P3,P4}
    Re::Matrix{T}        # (Ns, 2 n_even): Re/Im of the even-m modes
    Ro::Matrix{T}        # (Ns, 2 n_odd)
    pef::P1              # DCT-II   (REDFT10), in place along s
    peb::P2              # DST-III  (RODFT01)
    pof::P3              # DST-II   (RODFT10)
    pob::P4              # DCT-III  (REDFT01)
    kap::Vector{T}       # κ_k / (2 Ns), k = 1..Ns-1 (includes the DCT/DST normalisation)
    invrp::Vector{T}     # 1 / r'(s_j)
    ub::Matrix{T}        # (Ns, 2): Re/Im of u = r F_1 (conservative m = 0 flux)
    puf::P1              # DCT-II on ub
    sig::Vector{T}       # σ_k = (πk/2Ns)/sin(πk/2Ns): flux function Ψ = Σ a_k σ_k cos κ_k s
    cosJ::Vector{T}      # cos(κ_k J Δs) at the bookkeeping face J = Ns ÷ 2
end

function _SpecDeriv(grid::PolarVGrid, ::Type{T}, Mc::Int) where {T}
    Mc >= 2 || throw(ArgumentError("PolarSpectral needs at least modes m = 0, 1 (Nφ >= 3)"))
    Ns = grid.Ns
    ne = cld(Mc, 2)          # m = 0, 2, 4, ...
    no = Mc ÷ 2              # m = 1, 3, ...
    Re = zeros(T, Ns, 2ne)
    Ro = zeros(T, Ns, 2no)
    S = Ns * grid.ds
    ub = zeros(T, Ns, 2)
    return _SpecDeriv(
        Re,
        Ro,
        FFTW.plan_r2r!(Re, FFTW.REDFT10, 1; flags = FFTW.ESTIMATE),
        FFTW.plan_r2r!(Re, FFTW.RODFT01, 1; flags = FFTW.ESTIMATE),
        FFTW.plan_r2r!(Ro, FFTW.RODFT10, 1; flags = FFTW.ESTIMATE),
        FFTW.plan_r2r!(Ro, FFTW.REDFT01, 1; flags = FFTW.ESTIMATE),
        T[pi * k / S / (2Ns) for k = 1:(Ns-1)],
        T[1 / grid.drds[j] for j = 1:Ns],
        ub,
        FFTW.plan_r2r!(ub, FFTW.REDFT10, 1; flags = FFTW.ESTIMATE),
        T[(pi * k / (2Ns)) / sin(pi * k / (2Ns)) for k = 1:(Ns-1)],
        T[cos(pi * k * _bookkeeping_face(grid) / Ns) for k = 1:(Ns-1)],
    )
end

# Spectral analogue: the flux function Ψ(s) = a_0 + Σ_k a_k σ_k cos κ_k s of the cosine
# series of u, for which Δs Σ_{j≤J} (∂_s u)_j = Ψ(JΔs) - Ψ(0) exactly
# (Σ_{j=1}^J sin((j-1/2)θ) = sin²(Jθ/2)/sin(θ/2)); returns Ψ at the face J+1/2 = JΔs.
function _k2_face_flux(u, st::_K2Static{T}, sp::_SpecDeriv{T}) where {T}
    Ns = length(u)
    ub = sp.ub
    @inbounds for j = 1:Ns
        ub[j, 1] = real(u[j])
        ub[j, 2] = imag(u[j])
    end
    sp.puf * ub                                   # A_k at row k+1
    p = complex(ub[1, 1], ub[1, 2]) / (2Ns)      # a_0
    @inbounds for k = 1:(Ns-1)
        p += complex(ub[k+1, 1], ub[k+1, 2]) / Ns * sp.sig[k] * sp.cosJ[k]
    end
    return p
end

struct _K3Workspace{A,B}
    k2::A
    spec::B
end

function _k3_workspaces(grid::PolarVGrid, ::Type{T}, nc) where {T}
    k2w = _k2_workspaces(grid, T, nc)
    return _polar_buffer!(grid, (:k3, T, nc)) do
        [_K3Workspace(k2w[c], _SpecDeriv(grid, T, size(k2w[c].U, 2))) for c = 1:nc]
    end::Vector
end

# D[:, col] = ∂_r F_{col-1}, spectral in s (see `PolarSpectral`).
function _k2_ddr!(D, U, st::_K2Static{T}, sp::_SpecDeriv{T}) where {T}
    Ns, Mc = size(U)
    Re, Ro, kap = sp.Re, sp.Ro, sp.kap
    @inbounds for col = 1:Mc
        if iseven(col - 1)
            c = 2 * ((col - 1) ÷ 2) + 1
            for j = 1:Ns
                Re[j, c] = real(U[j, col])
                Re[j, c+1] = imag(U[j, col])
            end
        else
            c = 2 * ((col - 2) ÷ 2) + 1
            for j = 1:Ns
                Ro[j, c] = real(U[j, col])
                Ro[j, c+1] = imag(U[j, col])
            end
        end
    end
    sp.pef * Re                       # A_k at row k+1 (k = 0..Ns-1)
    sp.pof * Ro                       # B at row k (wavenumber k = 1..Ns)
    @inbounds for c = 1:size(Re, 2)
        for i = 1:(Ns-1)              # DST-III input X_{i-1} (wavenumber i) = -κ_i A_i / 2Ns
            Re[i, c] = -kap[i] * Re[i+1, c]
        end
        Re[Ns, c] = 0                 # wavenumber Ns: no cosine partner on the nodes
    end
    @inbounds for c = 1:size(Ro, 2)
        for i = (Ns-1):-1:1           # DCT-III input X_i (wavenumber i) = κ_i B_{i-1} / 2Ns
            Ro[i+1, c] = kap[i] * Ro[i, c]
        end
        Ro[1, c] = 0                  # sin(π Ns s/S) has zero derivative on the nodes
    end
    sp.peb * Re
    sp.pob * Ro
    @inbounds for col = 1:Mc
        if iseven(col - 1)
            c = 2 * ((col - 1) ÷ 2) + 1
            for j = 1:Ns
                D[j, col] = complex(Re[j, c], Re[j, c+1]) * sp.invrp[j]
            end
        else
            c = 2 * ((col - 2) ÷ 2) + 1
            for j = 1:Ns
                D[j, col] = complex(Ro[j, c], Ro[j, c+1]) * sp.invrp[j]
            end
        end
    end
    return D
end

# ── Kick driver ──────────────────────────────────────────────────────────────

function _kick_slice!(ws::_K1Workspace, grid, f, ix, dx, dy, ::PolarSL{W}, _) where {W}
    nphi, Ns = size(ws.g, 2), grid.Ns
    @inbounds for l = 1:nphi, j = 1:Ns
        ws.g[j, l] = f[ix, grid.offset[j]+l]
    end
    out = _k1_slice!(ws, grid, dx, dy, Val(W))
    @inbounds for l = 1:nphi, j = 1:Ns
        f[ix, grid.offset[j]+l] = out[j, l]
    end
    return f
end
_kick_slice!(ws::_K2Workspace, grid, f, ix, dx, dy, method::PolarTaylor{W}, st) where {W} =
    _k2_slice!(ws, st, grid, f, ix, dx, dy, method, Val(W))
_kick_slice!(ws::_K3Workspace, grid, f, ix, dx, dy, method::PolarSpectral, st) =
    _k2_slice!(ws.k2, st, grid, f, ix, dx, dy, method, ws.spec)

_kick_workspaces(grid, ::Type{T}, ::PolarSL{W}, nc) where {T,W} =
    _k1_workspaces(grid, T, Val(W), nc)
_kick_workspaces(grid, ::Type{T}, ::PolarTaylor, nc) where {T} = _k2_workspaces(grid, T, nc)
_kick_workspaces(grid, ::Type{T}, ::PolarSpectral, nc) where {T} = _k3_workspaces(grid, T, nc)
_kick_static(grid, ::Type{T}, ::PolarSL) where {T} = nothing
_kick_static(grid, ::Type{T}, m::PolarTaylor{W}) where {T,W} =
    _k2_static(grid, T, Val(W), m.massfix, m.extrap, m.edge)
# (the FD weights in the static data are unused by K3; W = 2 keeps them trivial)
_kick_static(grid, ::Type{T}, m::PolarSpectral) where {T} =
    _k2_static(grid, T, Val(2), m.massfix, m.extrap, m.edge)

_radial_sigma(::Val, st) = st.sigma
_radial_sigma(::_SpecDeriv, st) = oftype(st.sigma, pi)   # spectral k_max Δs = π

"""
    polar_kick!(f::AbstractMatrix, grid::PolarVGrid, dx, dy; method = PolarSL(8))

Translate every x slice of `f` (layout `(Nx, Nv)`) in velocity: `f(v) <- f(v - δ)`,
`δ = (dx[ix], dy[ix])` (vectors of length Nx, or scalars). Threaded over x.
"""
function polar_kick!(f::AbstractMatrix{T}, grid::PolarVGrid, dx, dy; method = PolarSL(8)) where {T}
    nx = size(f, 1)
    nc = _nchunks(nx)
    wss = _kick_workspaces(grid, T, method, nc)   # K1 asserts uniform Nφ here
    st = _kick_static(grid, T, method)
    @threads for c = 1:nc
        ws = wss[c]
        for ix in _chunk(c, nc, nx)
            ddx = T(dx isa Number ? dx : dx[ix])
            ddy = T(dy isa Number ? dy : dy[ix])
            (ddx == 0 && ddy == 0) && continue          # exact no-op
            _kick_slice!(ws, grid, f, ix, ddx, ddy, method, st)
        end
    end
    return f
end

"""
    advectV!(sp, grid::PolarVGrid, simTime, e::VectorField; method = PolarSL(8))

E-kick over `simTime.dt * simTime.fraction_dt` in the rotating frame of
`simTime.phase`: `f(v) <- f(v - δ)`, `δ = τ (q/√m) R(θ) E`, θ = (q/√m) phase — the
convention of the Cartesian `advectV!`. `method` is `PolarSL(W)` (K1),
`PolarTaylor(W; order)` (K2, FD radial derivative) or `PolarSpectral(; order)` (K3,
DCT/DST radial derivative).
"""
function advectV!(
    sp::SpeciesPolarV,
    grid::PolarVGrid,
    simTime::SimulationTime,
    e::VectorField;
    method = PolarSL(8),
)
    f = sp.dist.data
    T = eltype(f)
    es = electric_acceleration_scale(sp)
    th = es * simTime.phase
    scale = _effective_dt(simTime) * es
    nx = size(f, 1)
    dx = Vector{T}(undef, nx)
    dy = Vector{T}(undef, nx)
    for ix = 1:nx
        dx[ix], dy[ix] = _kick_delta(e, ix, th, scale)
    end
    polar_kick!(f, grid, dx, dy; method = method)
    return nothing
end
