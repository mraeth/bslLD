# Polar perpendicular-velocity grid for 1d2v ("P2" scheme), bslLD-local.
#
# Velocity v = (r cos φ, r sin φ) in the rotating frame of the Cartesian path. Radial
# nodes are offset in a computational coordinate s, s_j = (j - 1/2) Δs, j = 1..Ns, and
# mapped by an odd smooth map r = r(s). Because the map is odd, the ghost node s_{1-j} = -s_j
# sits at r = -r_j, where the parity relation  f̂_m(-r) = (-1)^m f̂_m(r)  of the angular
# Fourier coefficients is exact. φ is uniform with Nφ_j nodes on ring j.
#
# Data layout: f.data is (Nx, Nv), Nv = Σ_j Nφ_j; velocity index iv = offset[j] + l
# (l = 1..Nφ_j, φ_l = (l-1) 2π/Nφ_j). Each ring is a contiguous block, so a ragged Nφ(r)
# only changes `nphi`/`offset`. Only uniform Nφ is implemented by the advectors so far.

"""
    PolarV

Distribution tag of the polar 1d2v velocity grid (`PolarVGrid`). Distinct from
PlasmaCore's `Polar`, which belongs to the old prototype in `advectorPolar.jl`.
"""
abstract type PolarV end

const DistributionPolarV{DT,AT} = DistributionGrid{DT,1,2,2,PolarV,AT}

# ── Radial maps ──────────────────────────────────────────────────────────────

"""
    SinhMap(a)

Odd radial map `r(s) = a sinh(s/a)`, `dr/ds = cosh(s/a)`, explicit inverse
`s(r) = a asinh(r/a)`. `dr/ds = 1` at the pole and `≈ r/a` for `r ≫ a`: nodes are
uniform (spacing Δs) below `r ≈ a` and stretch geometrically in the tail. `a → ∞` is
the uniform grid. The explicit inverse matters: the semi-Lagrangian kick inverts the
map once per grid point and kick.
"""
struct SinhMap{T<:AbstractFloat}
    a::T
end
SinhMap(a::Real) = SinhMap(float(a))

"""
    IdentityMap()

Uniform radial nodes, `r = s`.
"""
struct IdentityMap end

@inline radius(m::SinhMap, s) = m.a * sinh(s / m.a)
@inline dradius(m::SinhMap, s) = cosh(s / m.a)
@inline inv_radius(m::SinhMap, r) = m.a * asinh(r / m.a)
@inline radius(::IdentityMap, s) = s
@inline dradius(::IdentityMap, s) = one(s)
@inline inv_radius(::IdentityMap, r) = r

# ── Endpoint-corrected midpoint rule at s = 0 ────────────────────────────────
#
# ∫ f dvx dvy = ∫_0^∞ ∫_0^2π f r dr dφ = 2π ∫_0^∞ g(s) ds,  g = F_0(r(s)) r(s) r'(s),
# F_0 the angular mean. F_0 is even in r, r odd, r' even  ⇒  g is ODD in s, so
# g^{(2k-1)}(0) ≠ 0 and the plain midpoint rule on s_j = (j-1/2)h is only O(h²)
# (Euler–Maclaurin). The left-end Euler–Maclaurin terms of the midpoint rule are
#
#   ∫_0^∞ g ds = h Σ_j g_j + Σ_{k≥1} B_{2k}(1/2) h^{2k}/(2k)! g^{(2k-1)}(0),
#
# B_{2k}(1/2) = -(1 - 2^{1-2k}) B_{2k}. With g(s) = Σ_q a_q s^{2q+1} fitted through the
# first K nodes (odd Vandermonde, exact for odd polynomials of degree 2K-1), the
# correction is Σ_k B_{2k}(1/2) h^{2k} a_{k-1}/(2k) = h Σ_{j≤K} γ_j g_j with
# γ = Ṽ^{-T} b,  Ṽ[j,q] = (j-1/2)^{2q+1},  b_k = B_{2k}(1/2)/(2k)  (independent of h).
# Truncation: O(h^{2K+2}) for smooth g decaying at the outer edge.

const _BERNOULLI_EVEN = (
    1 // 6,
    -1 // 30,
    1 // 42,
    -1 // 30,
    5 // 66,
    -691 // 2730,
    7 // 6,
    -3617 // 510,
)

# Solve A x = b exactly (Gaussian elimination in rationals; A small and nonsingular).
function _rational_solve(A::Matrix{Rational{BigInt}}, b::Vector{Rational{BigInt}})
    n = length(b)
    M = hcat(copy(A), copy(b))
    for c = 1:n
        p = findfirst(i -> M[i, c] != 0, c:n) + c - 1
        M[c, :], M[p, :] = M[p, :], M[c, :]
        for i = 1:n
            i == c && continue
            fac = M[i, c] / M[c, c]
            M[i, :] .-= fac .* M[c, :]
        end
    end
    return [M[i, n+1] / M[i, i] for i = 1:n]
end

"""
    polar_endpoint_corrections(K) -> Vector{Rational{BigInt}}

Relative corrections γ_j (j = 1..K) of the midpoint weights at the first K offset nodes
for integrands that are odd in s: `∫_0^∞ g ds ≈ h Σ_j (1 + γ_j) g(s_j)`, error
`O(h^{2K+2})`. `K = 0` is the plain midpoint rule. Supported `0 <= K <= 8`.
"""
function polar_endpoint_corrections(K::Integer)
    0 <= K <= length(_BERNOULLI_EVEN) ||
        throw(ArgumentError("quadrature correction order K must be in 0:8, got $K"))
    K == 0 && return Rational{BigInt}[]
    t = [big(2j - 1) // 2 for j = 1:K]
    VT = Rational{BigInt}[t[j]^(2q + 1) for q = 0:(K-1), j = 1:K]   # Ṽ^T
    b = Rational{BigInt}[
        -(1 - 1 // big(2)^(2k - 1)) * _BERNOULLI_EVEN[k] / (2k) for k = 1:K
    ]
    return _rational_solve(VT, b)
end

# ── Grid ─────────────────────────────────────────────────────────────────────

"""
    PolarVGrid(xmin, xmax, Nx, Ns, Nphi; rmax = 6.0, map = SinhMap(2.0), b0 = 1.0,
               Bdir = 3, quad_order = 6)

1d2v grid with a periodic x axis (`Nx` nodes on `[xmin, xmax)`, as the Cartesian
`Grid`) and a polar perpendicular velocity grid: `Ns` offset radial nodes
`s_j = (j-1/2)Δs`, `Δs = s(rmax)/Ns`, radii `r_j = r(s_j)` from the odd `map`, and
`Nphi` uniform angles per ring (an Int, a vector with one entry per ring, or a rule
`r -> Nφ` such as `RaggedNphi(c)`). `rmax` is the outer edge (`r(Ns Δs) = rmax`, beyond
which f is taken as zero). `quad_order = K` selects the endpoint-corrected midpoint rule
(`polar_endpoint_corrections`); the weights in `grid.weights` integrate `∫ f dvx dvy`.

`grid.xgrid` is a PlasmaCore `Grid` with the same x axis; the field solvers use it
(`solve_fields(m, grid::PolarVGrid, solver)` forwards to it).
"""
struct PolarVGrid{T<:AbstractFloat,XG<:Grid,XA,MP}
    xgrid::XG
    xaxes::XA
    map::MP
    Ns::Int
    ds::T
    rmax::T
    s::Vector{T}
    r::Vector{T}
    drds::Vector{T}
    nphi::Vector{Int}
    offset::Vector{Int}
    Nv::Int
    vr::Vector{T}
    vphi::Vector{T}
    weights::Vector{T}
    ring_weights::Vector{T}   # Σ_l weights on ring j  ( = 2π h r_j r'_j (1+γ_j) )
    gamma::Vector{T}
    b0::T
    Bdir::Int
    cache::Dict{Any,Any}
end

function PolarVGrid(
    xmin::Real,
    xmax::Real,
    Nx::Integer,
    Ns::Integer,
    Nphi;   # Integer (uniform), vector (one per ring) or callable r -> Nφ (e.g. RaggedNphi)
    rmax::Real = 6.0,
    map = SinhMap(2.0),
    b0::Real = 1.0,
    Bdir::Integer = 3,
    quad_order::Integer = 6,
)
    T = Float64
    Bdir == 3 || throw(ArgumentError("PolarVGrid supports Bdir = 3 (B along z) only"))
    xgrid = Grid([T(xmin), -1.0, -1.0], [T(xmax), 1.0, 1.0], [Int(Nx), 2, 2], 1, T(b0), 3)
    ds = T(inv_radius(map, T(rmax)) / Ns)
    s = [(j - T(0.5)) * ds for j = 1:Ns]
    r = [T(radius(map, sj)) for sj in s]
    drds = [T(dradius(map, sj)) for sj in s]
    nphi =
        Nphi isa Integer ? fill(Int(Nphi), Ns) :
        Nphi isa AbstractVector ? collect(Int, Nphi) : Int[Nphi(rj) for rj in r]
    length(nphi) == Ns || throw(ArgumentError("need one Nphi per ring ($Ns)"))
    all(>=(1), nphi) || throw(ArgumentError("Nphi must be positive"))
    offset = cumsum(vcat(0, nphi[1:(end-1)]))
    Nv = sum(nphi)
    gam = zeros(T, Ns)
    K = min(Int(quad_order), Ns)
    gq = polar_endpoint_corrections(K)
    for j = 1:K
        gam[j] = T(gq[j])
    end
    vr = Vector{T}(undef, Nv)
    vphi = similar(vr)
    weights = similar(vr)
    ring_weights = Vector{T}(undef, Ns)
    for j = 1:Ns
        ring_weights[j] = 2T(pi) * ds * r[j] * drds[j] * (1 + gam[j])
        for l = 1:nphi[j]
            iv = offset[j] + l
            vr[iv] = r[j]
            vphi[iv] = 2T(pi) * (l - 1) / nphi[j]
            weights[iv] = ring_weights[j] / nphi[j]
        end
    end
    return PolarVGrid(
        xgrid,
        xgrid.xaxes,
        map,
        Int(Ns),
        ds,
        T(rmax),
        s,
        r,
        drds,
        nphi,
        offset,
        Nv,
        vr,
        vphi,
        weights,
        ring_weights,
        gam,
        T(b0),
        Int(Bdir),
        Dict{Any,Any}(),
    )
end

"""
    RaggedNphi(c; n0 = 0, nmin = 8, nmax = 64)

Ring-size rule for `PolarVGrid(..., Nphi = RaggedNphi(c; n0))`:
`Nφ_j = clamp(2 ceil((n0 + c r_j) / 2), nmin, nmax)` (even).

The angular modes needed at radius r to reach an error ε are m ≲ κ r + m₀(ε), with κ the
velocity-space wavenumber of f (for a mode J_m(κ r): (κr/2)^m/m! < ε), so the rule is
affine: c ≈ 2κ and n0 ≈ 2 m₀ (n0 ≈ 12-16 for ε ~ 1e-6). The pure `max(nmin, c r)` form
(n0 = 0) is not enough: a mode dropped at small r still enters the m∓1 equations
through m F_m / r (see trials/polar_p2_design.md, §9). Only `PolarTaylor` (K2), x
advection, moments and init support ragged rings; `PolarSL` (K1) needs uniform Nφ.
"""
struct RaggedNphi
    c::Float64
    n0::Float64
    nmin::Int
    nmax::Int
end
# nmin is rounded up and nmax down to even values, so every Nφ_j is even.
function RaggedNphi(c::Real; n0::Real = 0, nmin::Integer = 8, nmax::Integer = 64)
    lo, hi = 2 * cld(nmin, 2), 2 * fld(nmax, 2)
    2 <= lo <= hi || throw(ArgumentError("RaggedNphi needs 2 <= nmin <= nmax (even-rounded)"))
    return RaggedNphi(c, n0, lo, hi)
end
(rule::RaggedNphi)(r) =
    clamp(2 * ceil(Int, (rule.n0 + rule.c * r) / 2), rule.nmin, rule.nmax)

uniform_nphi(grid::PolarVGrid) = all(==(grid.nphi[1]), grid.nphi)

function _require_uniform(grid::PolarVGrid)
    uniform_nphi(grid) ||
        throw(ArgumentError("ragged Nφ(r) is not implemented by the polar advectors yet"))
    return grid.nphi[1]
end

const _polar_cache_lock = ReentrantLock()
_polar_buffer!(make, grid::PolarVGrid, key) =
    lock(() -> get!(make, grid.cache, key), _polar_cache_lock)

Base.show(io::IO, g::PolarVGrid) = print(
    io,
    "PolarVGrid(Nx=$(length(g.xaxes[1])), Ns=$(g.Ns), Nφ=$(uniform_nphi(g) ? g.nphi[1] : "ragged"), rmax=$(g.rmax), map=$(g.map), K=$(count(!iszero, g.gamma)))",
)

gyro_frequency(s::Species, grid::PolarVGrid) = abs(s.q) / s.m * grid.b0

# The electrostatic and other field solvers only see the x grid.
solve_fields(moments::Moments, grid::PolarVGrid, solver::AbstractFieldSolver) =
    solve_fields(moments, grid.xgrid, solver)

# ── Initialisation ───────────────────────────────────────────────────────────

"""
    Distribution(grid::PolarVGrid, epsilon; m = 1.0, q = 1.0,
                 initFuncx = x -> 1 + epsilon*cos(2π x / L),
                 initFuncv = (r, φ) -> exp(-r^2/2) / (2π))

Species on the polar grid, `f(x, r, φ) = initFuncx(x) * initFuncv(r, φ)`. The default
is the 2-D Maxwellian in units of the thermal speed (`∫ f dvx dvy = 1`), seeded with
`cos(k x)`, `k = 2π/L` (the Cartesian default uses `sin`).
"""
function Distribution(
    grid::PolarVGrid,
    epsilon;
    m = 1.0,
    q = 1.0,
    initFuncx = (x -> 1.0 + epsilon * cos(2pi / (grid.xaxes[1][end] + grid.xgrid.delta[1] - grid.xaxes[1][1]) * (x - grid.xaxes[1][1]))),
    initFuncv = ((r, phi) -> exp(-r^2 / 2) / (2pi)),
)
    x = grid.xaxes[1]
    data = Float64[initFuncx(x[ix]) * initFuncv(grid.vr[iv], grid.vphi[iv]) for ix in eachindex(x), iv = 1:grid.Nv]
    sf = ScalarField(data)
    dist = DistributionGridImpl{1,2,PolarV,typeof(sf)}(sf)
    return Species(m, q, dist)
end

# ── Moments ──────────────────────────────────────────────────────────────────

function compute_density(f::DistributionGridImpl{1,2,PolarV}, grid::PolarVGrid)
    d = f.data
    nx = size(d, 1)
    rho = zeros(eltype(d), nx)
    w = grid.weights
    @inbounds for iv = 1:grid.Nv
        wv = w[iv]
        @simd for ix = 1:nx
            rho[ix] += wv * d[ix, iv]
        end
    end
    return ScalarField(rho)
end

# Current (∫ v f) in the rotating-frame velocity coordinates, as the Cartesian
# `compute_current(f, grid)`.
function compute_current(f::DistributionGridImpl{1,2,PolarV}, grid::PolarVGrid)
    d = f.data
    nx = size(d, 1)
    jx = zeros(eltype(d), nx)
    jy = zeros(eltype(d), nx)
    @inbounds for iv = 1:grid.Nv
        wx = grid.weights[iv] * grid.vr[iv] * cos(grid.vphi[iv])
        wy = grid.weights[iv] * grid.vr[iv] * sin(grid.vphi[iv])
        @simd for ix = 1:nx
            jx[ix] += wx * d[ix, iv]
            jy[ix] += wy * d[ix, iv]
        end
    end
    return VectorField([jx, jy])
end

# Lab-frame current: rotate back by -q/sqrt(m) * phase (as the Cartesian method).
function compute_current(sp::Species, grid::PolarVGrid, phase::Real)
    J = compute_current(sp.dist, grid)
    th = -electric_acceleration_scale(sp) * phase
    c, s = cos(th), sin(th)
    jx, jy = J[1].data, J[2].data
    return VectorField([c .* jx .- s .* jy, s .* jx .+ c .* jy])
end
