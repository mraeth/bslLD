# Magnetic geometry on top of the rotating-grid slab (see Notes, "Implementation
# Plan: From the Slab Model to a Local Tokamak Flux Tube").
#
# Axis convention for every geometry here: spatial axis 1 is radial (x), axis 2
# binormal/poloidal (y), axis 3 (if present) along the reference field (z), with
# `grid.Bdir == 3` and three velocity components. The rotating logical grid
# v = Q(t) u about z stays untouched: v_par = u_z and |v_perp|^2 = u_x^2 + u_y^2
# are invariant under Q, so the geometry terms below never need the phase except
# for the curvature force, which acts along a fixed physical direction.

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
    CurvedPatch(; Rc)

Step 3 of the plan, in the large-aspect-ratio reduction (`r0 -> ∞`, pitch
absorbed into the field-aligned reference direction): a frozen outboard patch
with

    B_z(x) = Rc / (Rc + x),     a_g = (v_z^2, 0, -v_x v_z) / (Rc + x),
    dz/dt  = Rc / (Rc + x) v_z.

`advectX!` applies the metric factor on axis 3; [`advect_geometry!`](@ref)
applies the residual field `B_z(x) - b0` and the curvature force in velocity
space. Characteristics conserve `f`; mass is conserved in the measure
`J dx dv` with `J = (Rc + x) / Rc`.
"""
Base.@kwdef struct CurvedPatch{T<:AbstractFloat} <: AbstractGeometry
    Rc::T
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

struct CurvedPatchShift{T,XT}
    Rc::T
    x::XT
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
    return CurvedPatchShift(DT(g.Rc), backend_vector(grid.xaxes[1]))
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
    ctx.dir == 3 || return zero(g.Rc)
    x = g.x[ixs[1]]
    return -ctx.vaxes[3][ivs[3]] * x / (g.Rc + x)
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

# Curvature kick along u_a: departure u_a - coef(x) * u_z^2.
struct CurvatureKick{CT,VT}
    coef::CT
    vz::VT
end

@inline function (s::CurvatureKick)(ixs, ivs)
    u = s.vz[ivs[3]]
    return s.coef[ixs[1]] * u * u
end

# du_z/dt = -w u_z with w = vth v_x / (Rc + x), v_x = c u_x + s u_y: departure
# u_z exp(w h), i.e. a shift that depends on u_z itself.
struct CurvatureScaling{CT,VT,T}
    wh::CT        # h * vth / (Rc + x) per radial index
    v1::VT
    v2::VT
    v3::VT
    c::T
    s::T
    inv_dv::T
end

@inline function (s::CurvatureScaling)(ixs, ivs)
    vx = s.c * s.v1[ivs[1]] + s.s * s.v2[ivs[2]]
    u = s.v3[ivs[3]]
    return u * (1 - exp(s.wh[ixs[1]] * vx)) * s.inv_dv
end

@inline _constant_along_line(::CurvatureScaling) = false

Adapt.@adapt_structure VelocityShear
Adapt.@adapt_structure CurvatureKick
Adapt.@adapt_structure CurvatureScaling

"""
    advect_geometry!(sp, grid, simTime, geometry; method = Lagrange(8), orbit_window = 0)

Velocity-space part of the geometry over `dt * fraction_dt`, at the logical-grid
phase `simTime.phase` (the curvature kick averaged over `orbit_window`, as in
`advectX!`). A no-op for `Slab` and `ShearedSlab`. For `CurvedPatch`:

1. residual field `δB_z(x) = b0 (Rc/(Rc+x) - 1)`: an x-dependent rotation of
   `(u_x, u_y)`, done exactly as three shears `Sx(tan θ/2) Sy(-sin θ) Sx(tan θ/2)`;
2. curvature force, Strang split into the translation `u_perp += Q^T x̂ h v_z^2/(Rc+x)`
   (half steps, depends on `u_z` only) around the exact scaling of `u_z`.
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
    DT = eltype(sp.dist.data)
    NX = length(grid.xaxes)
    h = DT(_effective_dt(simTime))
    x = collect(DT, grid.xaxes[1])
    dv = ntuple(d -> DT(grid.delta[NX+d]), Val(3))
    vaxes = map(backend_vector, grid.vaxes)
    vth = DT(thermal_velocity(sp))
    Rc = DT(g.Rc)

    # 1. residual rotation, θ(x) = sign(q) δΩ(x) h
    omega = DT(sign(sp.q) * gyro_frequency(sp, grid))
    theta = @. omega * (Rc / (Rc + x) - 1) * h
    a = backend_vector(@. tan(theta / 2) / dv[1])
    b = backend_vector(@. -sin(theta) / dv[2])
    sx = VelocityShear(a, vaxes[2], 2)
    _line_shift!(sp, grid, Val(:v), 1, sx, method)
    _line_shift!(sp, grid, Val(:v), 2, VelocityShear(b, vaxes[1], 1), method)
    _line_shift!(sp, grid, Val(:v), 1, sx, method)

    # 2. curvature force at the physical phase: Q^T x̂ = (cos φ, sin φ)
    phi = DT(electric_acceleration_scale(sp) * simTime.phase)
    c, s = (cos(phi), sin(phi)) .* DT(_orbit_factor(sp, grid, orbit_window))
    kick = @. h / 2 * vth / (Rc + x)
    kx = CurvatureKick(backend_vector(kick .* (c / dv[1])), vaxes[3])
    ky = CurvatureKick(backend_vector(kick .* (s / dv[2])), vaxes[3])
    _line_shift!(sp, grid, Val(:v), 1, kx, method)
    _line_shift!(sp, grid, Val(:v), 2, ky, method)
    scaling = CurvatureScaling(
        backend_vector(@. h * vth / (Rc + x)), vaxes[1], vaxes[2], vaxes[3], c, s,
        inv(dv[3]))
    _line_shift!(sp, grid, Val(:v), 3, scaling, method)
    _line_shift!(sp, grid, Val(:v), 1, kx, method)
    _line_shift!(sp, grid, Val(:v), 2, ky, method)
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
