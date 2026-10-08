# Magnetic geometry on top of the rotating-grid slab (see Notes, "Implementation
# Plan: From the Slab Model to a Local Tokamak Flux Tube").
#
# Axis convention for every geometry here: spatial axis 1 is radial (x), axis 2
# binormal/poloidal (y), axis 3 (if present) along the reference field (z), with
# `grid.Bdir == 3` and three velocity components. The rotating logical grid
# v = Q(t) u about z stays untouched: v_par = u_z and |v_perp|^2 = u_x^2 + u_y^2
# are invariant under Q. Terms that act along a fixed physical direction in the
# perpendicular plane (CurvedPatch: metric, curvature force, δB_2) need the phase.

abstract type AbstractGeometry end

"""
    Slab()

Uniform field along `grid.Bdir`; no geometric terms. The default.
"""
struct Slab <: AbstractGeometry end

"""
    ShearedSlab(; Ls = Inf, R0 = Inf, kcurv = 0.0)

Steps 1 and 2 of the plan. Adds to the binormal (axis 2) characteristic

    dy/dt += (x / Ls) v_par + v_My,     v_My = -(v_par^2 + v_perp^2/2) / (Ω R0) * K(z)

with `K(z) = cos(kcurv * z)` (`K = 1` for a 2D grid). `Ls = Inf` switches the
reduced shear off, `R0 = Inf` the prescribed magnetic drift. The Lorentz force
keeps the uniform field. The shift stays constant along y, so both the Fourier
and the Lagrange path apply.
"""
Base.@kwdef struct ShearedSlab{T<:AbstractFloat} <: AbstractGeometry
    Ls::T = Inf
    R0::T = Inf
    kcurv::T = 0.0
end

"""
    CurvedPatch(; Rc, r0 = Inf, q0 = Inf, shat = 0.0, R0 = Rc - r0)

Step 3 of the plan: a frozen outboard patch (x = r - r0, poloidal arc y_θ = r0 θ,
toroidal arc y_φ = Rc φ) with the field

    B(x) = B_θ e_θ + B_φ e_φ,   B_φ ∝ Rc/(Rc+x),   B_θ/B_φ = (r0+x)/(q(x) R0),
    q(x) = q0 (1 + shat x / r0),

normalised to `|B(0)| = b0`, and the geometric acceleration

    a_x = v_θ^2/(r0+x) + v_φ^2/(Rc+x),  a_θ = -v_x v_θ/(r0+x),  a_φ = -v_x v_φ/(Rc+x).

The grid is field-aligned at x = 0: velocity axis 3 is `b0 = B(0)/|B(0)|`,
axis 2 is `b0 × x̂`, and the spatial axes 2, 3 are the arcs rotated by the
pitch angle α, `tan α = r0/(q0 R0)`. The metric then reads

    (dy/dt, dz/dt) = M(x) (v_2, v_3),
    M = [c²m_θ + s²m_φ   cs(m_θ - m_φ);  cs(m_θ - m_φ)   s²m_θ + c²m_φ],

with `m_θ = r0/(r0+x)`, `m_φ = Rc/(Rc+x)`, `c, s = cos α, sin α`. The same `M`
maps the logical gradient to the physical field ([`apply_metric!`](@ref)). The
magnetic shear comes out of `B(x)` and `M`; there is no reduced `x v_∥/Ls` term.

`advectX!` applies `M` on axes 2 and 3; [`advect_geometry!`](@ref) applies the
residual field `B(x) - B(0)` and `a_g` in velocity space. Characteristics
conserve `f`; mass is conserved in the measure `J dx dv`,
`J = (r0+x)/r0 · (Rc+x)/Rc`. The defaults `r0 = q0 = Inf` give the
large-aspect-ratio patch `B = Rc/(Rc+x) e_z`, `a_g = (v_z², 0, -v_x v_z)/(Rc+x)`.
"""
struct CurvedPatch{T<:AbstractFloat} <: AbstractGeometry
    Rc::T
    r0::T
    q0::T
    shat::T
    R0::T
end

function CurvedPatch(; Rc, r0 = Inf, q0 = Inf, shat = 0.0, R0 = isfinite(r0) ? Rc - r0 : Rc)
    isinf(q0) || isfinite(r0) && R0 > 0 ||
        throw(ArgumentError("CurvedPatch: a finite q0 needs a finite r0 and R0 > 0"))
    T = float(promote_type(typeof(Rc), typeof(r0), typeof(q0), typeof(shat), typeof(R0)))
    return CurvedPatch{T}(Rc, r0, q0, shat, R0)
end

# tan α = B_θ/B_φ at x = 0
_pitch(g::CurvedPatch) = isinf(g.q0) ? zero(g.Rc) : g.r0 / (g.q0 * g.R0)

"`M(x)` of [`CurvedPatch`](@ref) as `(M22, M23, M33)`."
function _patch_metric(g::CurvedPatch, x)
    s, c = sincos(atan(_pitch(g)))
    mt = isinf(g.r0) ? one(x) : g.r0 / (g.r0 + x)
    mp = g.Rc / (g.Rc + x)
    return (c^2 * mt + s^2 * mp, s * c * (mt - mp), s^2 * mt + c^2 * mp)
end

"`B(x)/b0` of [`CurvedPatch`](@ref) on velocity axes 2 and 3 (axis 1 is zero)."
function _patch_field(g::CurvedPatch, x)
    s, c = sincos(atan(_pitch(g)))
    bphi = c * g.Rc / (g.Rc + x)
    p = isinf(g.q0) ? zero(x) : (g.r0 + x) / (g.q0 * (1 + g.shat * x / g.r0) * g.R0)
    return (bphi * (c * p - s), bphi * (s * p + c))
end

function _check_patch(g::CurvedPatch, x)
    ok = all(xi -> g.Rc + xi > 0 && g.r0 + xi > 0, x) &&
         (isinf(g.q0) || all(xi -> 1 + g.shat * xi / g.r0 > 0, x))
    ok || throw(ArgumentError("CurvedPatch: Rc + x, r0 + x and q(x) must stay positive " *
                              "on the grid, x ∈ [$(minimum(x)), $(maximum(x))]"))
    return nothing
end

function _check_geometry(grid::CartGrid)
    NX, NV = length(grid.xaxes), length(grid.vaxes)
    grid.Bdir == 3 && NV == 3 && NX >= 2 || throw(ArgumentError(
        "geometry needs >= 2 spatial axes, three velocity axes and Bdir == 3, " *
        "got NX = $NX, NV = $NV, Bdir = $(grid.Bdir)"))
    return nothing
end

# --- Spatial characteristics ----------------------------------------------

# Kernel-side data of a geometry for the spatial shift, or `nothing` for Slab.
struct ShearedSlabShift{T,XT,ZT}
    inv_Ls::T
    drift::T     # vth / (Ω R0): v_My / vth per (u_par^2 + u_perp^2/2)
    kcurv::T
    x::XT
    z::ZT        # axis-3 coordinates, or `nothing` on a 2D grid
end

struct CurvedPatchShift{XT}
    myy::XT      # M22 - 1 per radial index
    myz::XT      # M23
    mzz::XT      # M33 - 1
end

Adapt.@adapt_structure ShearedSlabShift
Adapt.@adapt_structure CurvedPatchShift

_spatial_geometry(::Slab, sp, grid, DT) = nothing

function _spatial_geometry(g::ShearedSlab, sp, grid, DT)
    _check_geometry(grid)
    z = length(grid.xaxes) >= 3 ? backend_vector(grid.xaxes[3]) : nothing
    return ShearedSlabShift(
        DT(inv(g.Ls)),
        DT(thermal_velocity(sp) / (gyro_frequency(sp, grid) * g.R0)),
        DT(g.kcurv),
        backend_vector(grid.xaxes[1]),
        z,
    )
end

function _spatial_geometry(g::CurvedPatch, sp, grid, DT)
    _check_geometry(grid)
    x = collect(grid.xaxes[1])
    _check_patch(g, x)
    m = _patch_metric.(Ref(g), x)
    return CurvedPatchShift(backend_vector(DT.(getindex.(m, 1) .- 1)),
        backend_vector(DT.(getindex.(m, 2))), backend_vector(DT.(getindex.(m, 3) .- 1)))
end

@inline _curvature_profile(::Nothing, g, ixs) = one(g.kcurv)
@inline _curvature_profile(z, g, ixs) = cos(g.kcurv * z[ixs[3]])

# Correction to the axis-`ctx.dir` displacement (per unit vth) on top of the
# rotated logical velocity.
@inline _geometric_displacement(::Nothing, ctx, ixs, ivs) = zero(eltype(ctx.k))

@inline function _geometric_displacement(g::ShearedSlabShift, ctx, ixs, ivs)
    ctx.dir == 2 || return zero(g.inv_Ls)
    u1, u2, u3 = ctx.vaxes[1][ivs[1]], ctx.vaxes[2][ivs[2]], ctx.vaxes[3][ivs[3]]
    shear = g.x[ixs[1]] * g.inv_Ls * u3
    drift = -g.drift * (u3 * u3 + (u1 * u1 + u2 * u2) / 2) * _curvature_profile(g.z, g, ixs)
    return shear + drift
end

@inline function _geometric_displacement(g::CurvedPatchShift, ctx, ixs, ivs)
    ctx.dir == 1 && return zero(eltype(g.myy))
    rot = R(ctx.grid.Bdir, -ctx.electric_scale * ctx.phi)
    v2 = (ctx.vaxes[1][ivs[1]] * rot[2, 1] + ctx.vaxes[2][ivs[2]] * rot[2, 2]) * ctx.orbit
    v3 = ctx.vaxes[3][ivs[3]]
    i = ixs[1]
    return ctx.dir == 2 ? g.myy[i] * v2 + g.myz[i] * v3 : g.myz[i] * v2 + g.mzz[i] * v3
end

# `has_z = false`: no spatial axis 3, the logical E_3 is zero (the solvers leave
# that component of their reused buffer untouched).
@kernel function metric_field_kernel!(E2, E3, m, has_z)
    I = @index(Global, Cartesian)
    i = I[1]
    e2, e3 = E2[I], has_z ? E3[I] : zero(eltype(E3))
    @inbounds E2[I] = e2 + m.myy[i] * e2 + m.myz[i] * e3
    @inbounds E3[I] = e3 + m.myz[i] * e2 + m.mzz[i] * e3
end

"""
    apply_metric!(E, grid, geometry)

Turn the logical field `E = -∇φ` (derivatives along the grid axes, as returned
by the field solvers) into the physical field on the velocity axes, in place.
For [`CurvedPatch`](@ref), `(E_2, E_3) ← M(x) (E_2, E_3)`; a no-op otherwise.
Call it after `solve_fields` and before `advectV!`/`add_kappaT!`.
"""
apply_metric!(E::VectorField, grid::CartGrid, ::AbstractGeometry; exec = bslLD.backend()) = E

function apply_metric!(E::VectorField, grid::CartGrid, g::CurvedPatch; exec = bslLD.backend())
    E2, E3 = E[2].data, E[3].data
    m = _spatial_geometry(g, nothing, grid, eltype(E2))
    metric_field_kernel!(exec)(E2, E3, m, length(grid.xaxes) >= 3; ndrange = size(E2))
    KernelAbstractions.synchronize(exec)
    return E
end

# --- Generic line shifts --------------------------------------------------

"""
    LineShiftContext{S}(sizes_x, sizes_v, dir, shift)

Advection along spatial (`S = :x`) or velocity (`S = :v`) axis `dir` with a
departure point given per grid point by the callable `shift(ixs, ivs)` (in
cells). Unlike the X/V contexts the shift may depend on the advected
coordinate itself; `lagrange_shift_kernel!` computes the weights per point.
"""
struct LineShiftContext{S,SXT,SVT,FT}
    sizes_x::SXT
    sizes_v::SVT
    dir::Int
    shift::FT
end

LineShiftContext{S}(sizes_x, sizes_v, dir, shift) where {S} =
    LineShiftContext{S,typeof(sizes_x),typeof(sizes_v),typeof(shift)}(
        sizes_x, sizes_v, dir, shift)

Adapt.adapt_structure(to, c::LineShiftContext{S}) where {S} =
    LineShiftContext{S}(c.sizes_x, c.sizes_v, c.dir, Adapt.adapt(to, c.shift))

@inline _shift_cells(ctx::LineShiftContext, ixs, ivs) = ctx.shift(ixs, ivs)

@inline _line_length(ctx::LineShiftContext{:x}) = ctx.sizes_x[ctx.dir]
@inline _line_length(ctx::LineShiftContext{:v}) = ctx.sizes_v[ctx.dir]
@inline _line_index(ctx::LineShiftContext{:x}, ixs, ivs) = ixs[ctx.dir]
@inline _line_index(ctx::LineShiftContext{:v}, ixs, ivs) = ivs[ctx.dir]
@inline _line_linear_index(ctx::LineShiftContext{:x}, ixs, ivs, i) =
    index_combined_to_1d(Base.setindex(ixs, i, ctx.dir), ivs, ctx.sizes_x, ctx.sizes_v)
@inline _line_linear_index(ctx::LineShiftContext{:v}, ixs, ivs, i) =
    index_combined_to_1d(ixs, Base.setindex(ivs, i, ctx.dir), ctx.sizes_x, ctx.sizes_v)
@inline _line_stride(ctx::LineShiftContext{:x}) = prod(ctx.sizes_x[1:(ctx.dir-1)])
@inline _line_stride(ctx::LineShiftContext{:v}) =
    prod(ctx.sizes_x) * prod(ctx.sizes_v[1:(ctx.dir-1)])
@inline _line_sizes(ctx::LineShiftContext{:x}) =
    (Base.setindex(ctx.sizes_x, 1, ctx.dir), ctx.sizes_v)
@inline _line_sizes(ctx::LineShiftContext{:v}) =
    (ctx.sizes_x, Base.setindex(ctx.sizes_v, 1, ctx.dir))
# The u_z scaling depends on the advected coordinate itself: per-point weights.
@inline _constant_along_line(ctx::LineShiftContext) = _constant_along_line(ctx.shift)

function _line_shift!(sp::Species, grid::CartGrid, ::Val{S}, dir, shift, method) where {S}
    sizes_x, sizes_v = _cartesian_axis_sizes(grid)
    ctx = LineShiftContext{S}(sizes_x, sizes_v, dir, shift)
    plan = _get_plan(sp.dist, grid)
    _advect_dir!(sp.dist, ctx, plan, method, Periodic(), nothing, nothing)
    return nothing
end

# --- Velocity-space subflows of CurvedPatch --------------------------------

# du_a/dt-type shear: departure u_a - coef(x) * u_b, in cells of axis a.
struct VelocityShear{CT,VT}
    coef::CT      # per radial index, already divided by delta_v[a]
    vb::VT
    b::Int
end

@inline (s::VelocityShear)(ixs, ivs) = s.coef[ixs[1]] * s.vb[ivs[s.b]]

# Shear of u_z by the in-plane projection q = a1 u_x + a2 u_y: departure
# u_z - coef(x) * q, in cells of axis 3.
struct ProjectedShear{CT,VT,T}
    coef::CT
    v1::VT
    v2::VT
    a1::T
    a2::T
end

@inline (s::ProjectedShear)(ixs, ivs) =
    s.coef[ixs[1]] * (s.a1 * s.v1[ivs[1]] + s.a2 * s.v2[ivs[2]])

# Geometric acceleration on the logical grid, du/dt = Q^T a_g(Q u), along axis D
# with the other two components frozen. The departure point comes from one RK4
# step backwards over `tau`; it depends on u_D itself (per-point weights).
struct CurvatureSweep{D,CT,VT,T}
    kr::CT        # vth / (r0 + x) per radial index (0 for r0 = ∞)
    kR::CT        # vth / (Rc + x)
    v1::VT
    v2::VT
    v3::VT
    c::T          # Q^T x̂ = (c, s), times the orbit factor
    s::T
    ca::T         # cos, sin of the pitch angle
    sa::T
    tau::T
    inv_dv::T
end

CurvatureSweep{D}(kr, kR, v1, v2, v3, c, s, ca, sa, tau, inv_dv) where {D} =
    CurvatureSweep{D,typeof(kr),typeof(v1),typeof(c)}(kr, kR, v1, v2, v3, c, s, ca, sa,
        tau, inv_dv)

Adapt.adapt_structure(to, s::CurvatureSweep{D}) where {D} = CurvatureSweep{D}(
    Adapt.adapt(to, s.kr), Adapt.adapt(to, s.kR), Adapt.adapt(to, s.v1),
    Adapt.adapt(to, s.v2), Adapt.adapt(to, s.v3), s.c, s.s, s.ca, s.sa, s.tau, s.inv_dv)

@inline function _curvature_rate(s::CurvatureSweep{D}, i, u1, u2, u3) where {D}
    vx = s.c * u1 + s.s * u2          # physical v_1, v_2 (v_3 = u_3)
    v2 = s.c * u2 - s.s * u1
    vt = s.ca * v2 + s.sa * u3        # poloidal and toroidal components
    vp = s.ca * u3 - s.sa * v2
    kr, kR = s.kr[i], s.kR[i]
    ax = kr * vt * vt + kR * vp * vp
    at = -kr * vx * vt
    ap = -kR * vx * vp
    a2 = s.ca * at - s.sa * ap
    D == 1 && return s.c * ax - s.s * a2
    D == 2 && return s.s * ax + s.c * a2
    return s.sa * at + s.ca * ap
end

@inline function (s::CurvatureSweep{D})(ixs, ivs) where {D}
    i = ixs[1]
    u = (s.v1[ivs[1]], s.v2[ivs[2]], s.v3[ivs[3]])
    rate(y) = _curvature_rate(s, i, Base.setindex(u, y, D)...)
    y, h = u[D], -s.tau
    k1 = rate(y)
    k2 = rate(y + h / 2 * k1)
    k3 = rate(y + h / 2 * k2)
    k4 = rate(y + h * k3)
    return -h / 6 * (k1 + 2k2 + 2k3 + k4) * s.inv_dv
end

@inline _constant_along_line(::CurvatureSweep) = false

Adapt.@adapt_structure VelocityShear
Adapt.@adapt_structure ProjectedShear

# Right-handed rotation of (u_x, u_y) about ẑ by a(x), as Sx Sy Sx.
function _rotate_z!(sp, grid, a, dv, vaxes, method)
    sx = VelocityShear(backend_vector(@. -tan(a / 2) / dv[1]), vaxes[2], 2)
    sy = VelocityShear(backend_vector(@. sin(a) / dv[2]), vaxes[1], 1)
    _line_shift!(sp, grid, Val(:v), 1, sx, method)
    _line_shift!(sp, grid, Val(:v), 2, sy, method)
    _line_shift!(sp, grid, Val(:v), 1, sx, method)
end

# Right-handed rotation about ê = Q^T e_2 = (-s, c, 0) by b(x). It acts on
# (q, u_z), q = c u_x + s u_y, as Sz(-tan b/2) Sq(sin b) Sz(-tan b/2); the q-shear
# is two commuting translations of u_x and u_y that depend on u_z only.
function _rotate_e!(sp, grid, b, c, s, dv, vaxes, method)
    sz = ProjectedShear(backend_vector(@. -tan(b / 2) / dv[3]), vaxes[1], vaxes[2], c, s)
    _line_shift!(sp, grid, Val(:v), 3, sz, method)
    _line_shift!(sp, grid, Val(:v), 1,
        VelocityShear(backend_vector(@. c * sin(b) / dv[1]), vaxes[3], 3), method)
    _line_shift!(sp, grid, Val(:v), 2,
        VelocityShear(backend_vector(@. s * sin(b) / dv[2]), vaxes[3], 3), method)
    _line_shift!(sp, grid, Val(:v), 3, sz, method)
end

# Exact residual-field rotation over h: du/dt = Ω_s u × n, n = Q^T δB/b0 =
# β2 ê + (β3 - 1) ẑ, i.e. rotation by the vector ρ = -Ω_s h n. Written as
# R_z(a) R_ê(b) R_z(a) with sin(b/2) = sin(|ρ|/2) ρ_e/|ρ| and
# tan a = tan(|ρ|/2) ρ_z/|ρ| (equal quaternions); a single R_z if β2 = 0.
function _residual_rotation!(sp, grid, g::CurvedPatch, h, c, s, o, method)
    DT = eltype(sp.dist.data)
    NX = length(grid.xaxes)
    x = collect(DT, grid.xaxes[1])
    dv = ntuple(d -> DT(grid.delta[NX+d]), Val(3))
    vaxes = map(backend_vector, grid.vaxes)
    omega = DT(sign(sp.q) * gyro_frequency(sp, grid))
    beta = _patch_field.(Ref(g), x)
    rz = @. -omega * h * (last(beta) - 1)
    re = @. -omega * h * o * first(beta)
    if all(iszero, re)
        _rotate_z!(sp, grid, rz, dv, vaxes, method)
        return nothing
    end
    rho = hypot.(re, rz)
    k = @. ifelse(iszero(rho), DT(1 / 2), sin(rho / 2) / rho)
    a = @. atan(k * rz, cos(rho / 2))
    b = @. 2 * asin(k * re)
    _rotate_z!(sp, grid, a, dv, vaxes, method)
    _rotate_e!(sp, grid, b, DT(c), DT(s), dv, vaxes, method)
    _rotate_z!(sp, grid, a, dv, vaxes, method)
    return nothing
end

# a_g as Strang sweeps u_x(h/2) u_y(h/2) u_z(h) u_y(h/2) u_x(h/2).
function _curvature_force!(sp, grid, g::CurvedPatch, h, c, s, method)
    DT = eltype(sp.dist.data)
    NX = length(grid.xaxes)
    x = collect(DT, grid.xaxes[1])
    dv = ntuple(d -> DT(grid.delta[NX+d]), Val(3))
    vaxes = map(backend_vector, grid.vaxes)
    vth = DT(thermal_velocity(sp))
    kr = backend_vector(isinf(g.r0) ? zero(x) : @. vth / (DT(g.r0) + x))
    kR = backend_vector(@. vth / (DT(g.Rc) + x))
    sa, ca = DT.(sincos(atan(_pitch(g))))
    for (d, tau) in ((1, h / 2), (2, h / 2), (3, h), (2, h / 2), (1, h / 2))
        sweep = CurvatureSweep{d}(kr, kR, vaxes..., DT(c), DT(s), ca, sa, DT(tau),
            inv(dv[d]))
        _line_shift!(sp, grid, Val(:v), d, sweep, method)
    end
    return nothing
end

"""
    advect_geometry!(sp, grid, simTime, geometry; method = Lagrange(8), orbit_window = 0)

Velocity-space part of the geometry over `h = dt * fraction_dt`, at the
logical-grid phase `simTime.phase` (the phase-dependent directions averaged
over `orbit_window`, as in `advectX!`). A no-op for `Slab` and `ShearedSlab`.
For [`CurvedPatch`](@ref) it is the symmetric split `B(h/2) A(h) B(h/2)`:

- `B`: residual field `B(x) - B(0)`, an x-dependent rotation about an axis in
  the plane of `ẑ` and `Q^T e_2`, done exactly as `R_z R_ê R_z`, each factor
  three one-dimensional shears with shifts constant along their lines (one
  `R_z` when there is no pitch);
- `A`: geometric acceleration `Q^T a_g(Q u)`, as one-dimensional sweeps
  `u_x u_y u_z u_y u_x` with RK4 departure points (per-point weights).
"""
advect_geometry!(sp::Species, grid::CartGrid, simTime::SimulationTime, ::AbstractGeometry;
    method = Lagrange(8), orbit_window = 0) = nothing

function advect_geometry!(
    sp::Species,
    grid::CartGrid,
    simTime::SimulationTime,
    g::CurvedPatch;
    method = Lagrange(8),
    orbit_window = 0,
)
    _check_geometry(grid)
    _check_patch(g, grid.xaxes[1])
    h = _effective_dt(simTime)
    phi = electric_acceleration_scale(sp) * simTime.phase
    c, s = cos(phi), sin(phi)
    o = _orbit_factor(sp, grid, orbit_window)
    _residual_rotation!(sp, grid, g, h / 2, c, s, o, method)
    _curvature_force!(sp, grid, g, h, c * o, s * o, method)
    _residual_rotation!(sp, grid, g, h / 2, c, s, o, method)
    return nothing
end

# --- Energy exchange of the prescribed drift -------------------------------

struct DriftEnergyContext{ET,VAT,SXT,SVT,GT,T}
    Ey::ET
    vaxes::VAT
    sizes_x::SXT
    sizes_v::SVT
    geometry::GT
    dt::T
end

Adapt.@adapt_structure DriftEnergyContext

@kernel function drift_energy_kernel!(fdata, ctx)
    I = @index(Global, Linear)
    ixs, ivs = index_1d_to_combined(I, ctx.sizes_x, ctx.sizes_v)
    g = ctx.geometry
    u1, u2, u3 = ctx.vaxes[1][ivs[1]], ctx.vaxes[2][ivs[2]], ctx.vaxes[3][ivs[3]]
    fm = exp(-(u1 * u1 + u2 * u2 + u3 * u3) / 2) / (2 * pi)^(3 / 2)
    vMy = -g.drift * (u3 * u3 + (u1 * u1 + u2 * u2) / 2) * _curvature_profile(g.z, g, ixs)
    @inbounds fdata[I] += ctx.dt * ctx.Ey[ixs...] * vMy * fm
end

"""
    add_drift_energy!(sp, grid, dt, geometry::ShearedSlab, E)

Linearised work of `E` on the prescribed magnetic drift, `f += dt E_y v_My F_M`
(same normalisation as [`add_kappaT!`](@ref)). A drift that enters only the
y-characteristic moves particles across `φ` without changing their energy;
this term restores the `v_D·∇φ` coupling, turning the local dispersion relation
from `(ω - ω_D - ω*)/(ω - ω_D)` into the gyrokinetic `(ω - ω*)/(ω - ω_D)`.
"""
function add_drift_energy!(
    sp::Species,
    grid::CartGrid,
    dt,
    g::ShearedSlab,
    E::VectorField;
    exec = bslLD.backend(),
)
    DT = eltype(sp.dist.data)
    sizes_x, sizes_v = _cartesian_axis_sizes(grid)
    ctx = DriftEnergyContext(E[2].data, map(backend_vector, grid.vaxes), sizes_x, sizes_v,
        _spatial_geometry(g, sp, grid, DT), DT(dt))
    drift_energy_kernel!(exec)(sp.dist.data, ctx; ndrange = length(sp.dist.data))
    KernelAbstractions.synchronize(exec)
    return sp
end
