struct KappaTContext{ET,VAT,SXT,SVT,DT,KT}
    E::ET
    vaxes::VAT
    sizes_x::SXT
    sizes_v::SVT
    Bdir::Int
    dt::DT
    kappa_T::KT
end

Adapt.@adapt_structure KappaTContext

@inline _Ecomp(Ei, ixs) = Ei[ixs...]

@inline function _xhat_dot_ExB(ctx, ixs)
    ctx.Bdir == 1 && return zero(_Ecomp(ctx.E[1], ixs))       # B = (1,0,0)
    ctx.Bdir == 2 && return -_Ecomp(ctx.E[3], ixs)            # B = (0,1,0)
    return _Ecomp(ctx.E[2], ixs)                              # B = (0,0,1)
end

@kernel function kappaT_kernel!(fdata, ctx)
    I = @index(Global, Linear)
    ixs, ivs = index_1d_to_combined(I, ctx.sizes_x, ctx.sizes_v)

    v2 = zero(eltype(fdata))
    fac = one(eltype(fdata))
    sub = zero(eltype(fdata))

    @inbounds for d = 1:length(ivs)
        v = ctx.vaxes[d][ivs[d]]
        v2 += v*v
        fac *= exp(-v*v/2)/(2*pi)^(1/2)
        sub += 0.5
    end

    @inbounds fdata[I] +=
        ctx.kappa_T * ctx.dt * _xhat_dot_ExB(ctx, ixs) * (v2/2 - sub) * fac
end

function add_kappaT!(
    f::DistributionGrid{DT,NX,NV,NXNV,Cart},
    grid::CartGrid,
    dt,
    kappa_T,
    E::VectorField;
    exec = bslLD.backend(),
) where {DT,NX,NV,NXNV}

    sizes_x, sizes_v = _cartesian_axis_sizes(grid)

    ctx = KappaTContext(
        ntuple(i -> E[i].data, Val(3)),
        map(backend_vector, grid.vaxes),
        sizes_x,
        sizes_v,
        grid.Bdir,
        dt,
        kappa_T,
    )

    k! = kappaT_kernel!(exec)
    k!(f.data, ctx; ndrange = length(f.data))
    KernelAbstractions.synchronize(exec)

    return f
end

add_kappaT!(sp::DistSpecies, args...; kwargs...) = add_kappaT!(sp.dist, args...; kwargs...)

# --- Linear drift-kinetic kick ---------------------------------------------------------------
#
# Field kick of the linear DK model (dk_electrons.jl) on δF(x, v∥), velocities in units of
# 1/√m, Maxwellian F0(v) = c0 exp(-v²/2T)/√(2πT) and F0'(v) = -c1 (v/T) exp(-v²/2T)/√(2πT):
#
#   δF -= dt [F0 C / b0 + (q/√m) E∥ F0'],   C = b̂·∇×E,
#
# the E×B compression and the parallel acceleration of the background. One velocity axis, along B.

struct LinearDKKickContext{ET,CT,VT,SXT,SVT,DT}
    Epar::ET
    C::CT
    v::VT
    sizes_x::SXT
    sizes_v::SVT
    aC::DT   # dt c0 / b0
    aE::DT   # dt (q/√m) c1 / T
    T::DT
end

Adapt.@adapt_structure LinearDKKickContext

@kernel function linear_dk_kick_kernel!(fdata, ctx)
    I = @index(Global, Linear)
    ixs, ivs = index_1d_to_combined(I, ctx.sizes_x, ctx.sizes_v)
    @inbounds v = ctx.v[ivs[1]]
    g = exp(-v * v / (2 * ctx.T)) / sqrt(2 * oftype(v, pi) * ctx.T)
    @inbounds fdata[I] -= (ctx.aC * ctx.C[ixs...] - ctx.aE * ctx.Epar[ixs...] * v) * g
end

function add_linear_dk_kick!(
    f::DistributionGrid{DT,NX,NV,NXNV,Cart},
    grid::CartGrid,
    dt,
    Epar::ScalarField,
    curlE_par::ScalarField;
    q,
    m,
    T_e,
    c0 = 1,
    c1 = 1,
    exec = bslLD.backend(),
) where {DT,NX,NV,NXNV}
    NV == 1 || throw(ArgumentError("the linear DK kick needs one (parallel) velocity axis, got $NV"))
    sizes_x, sizes_v = _cartesian_axis_sizes(grid)
    ctx = LinearDKKickContext(
        Epar.data,
        curlE_par.data,
        backend_vector(grid.vaxes[1]),
        sizes_x,
        sizes_v,
        DT(dt * c0 / grid.b0),
        DT(dt * q / sqrt(m) * c1 / T_e),
        DT(T_e),
    )
    k! = linear_dk_kick_kernel!(exec)
    k!(f.data, ctx; ndrange = length(f.data))
    KernelAbstractions.synchronize(exec)
    return f
end

add_linear_dk_kick!(sp::DistSpecies, args...; kwargs...) = add_linear_dk_kick!(sp.dist, args...; kwargs...)

# --- Boussinesq density/temperature gradient drive -------------------------------------------

struct GradientDriveContext{ET,VAT,PT,SXT,SVT,DT}
    E::ET
    vaxes::VAT
    profile::PT
    sizes_x::SXT
    sizes_v::SVT
    Bdir::Int
    scale::DT
    kappa_n::DT
    kappa_T::DT
end

Adapt.@adapt_structure GradientDriveContext

@kernel function gradient_drive_kernel!(fdata, ctx)
    I = @index(Global, Linear)
    ixs, ivs = index_1d_to_combined(I, ctx.sizes_x, ctx.sizes_v)

    v2 = zero(eltype(fdata))
    fac = one(eltype(fdata))
    sub = zero(eltype(fdata))
    @inbounds for d = 1:length(ivs)
        v = ctx.vaxes[d][ivs[d]]
        v2 += v * v
        fac *= exp(-v * v / 2) / (2 * pi)^(1 / 2)
        sub += 0.5
    end

    ctx.profile === nothing || (@inbounds fac = ctx.profile[ivs...])
    @inbounds fdata[I] -=
        ctx.scale * _xhat_dot_ExB(ctx, ixs) * (ctx.kappa_n + ctx.kappa_T * (v2 / 2 - sub)) * fac
end

"""
    add_gradient_drive!(f, grid, dt, E; kappa_n, kappa_T = 0, n0 = 1)

Boussinesq drive of a background F₀(X) = n0 (1 + κn X)(1 + κT X (v²/2 − NV/2)) F_M(v) in the guiding-centre
position X along x: δf −= dt v_E,x n0 [κn + κT (v²/2 − NV/2)] F_M(v), with v_E,x = x̂·(E×b̂)/b0 and the unit
Maxwellian F_M in the species' thermal-velocity units, or with `profile = :mean` the x–y mean of f normalised to
unit density (a drifting or heated uniform equilibrium; κn only). κ = ∂x ln (KineticDR sign); the drive is exact for
an F₀ of X because dX/dt = v_E at the particle position. The κT part has zero density moment.
`add_kappaT!(f, grid, dt, κ, E)` equals `add_gradient_drive!(f, grid, dt, E; kappa_n = 0, kappa_T = -κ)` at b0 = 1.
"""
function add_gradient_drive!(
    f::DistributionGrid{DT,NX,NV,NXNV,Cart},
    grid::CartGrid,
    dt,
    E::VectorField;
    kappa_n,
    kappa_T = 0,
    n0 = 1,
    profile::Symbol = :maxwellian,
    exec = bslLD.backend(),
) where {DT,NX,NV,NXNV}
    sizes_x, sizes_v = _cartesian_axis_sizes(grid)
    P = if profile === :maxwellian
        nothing
    elseif profile === :mean
        iszero(kappa_T) || throw(ArgumentError("profile = :mean supports kappa_n only"))
        m = dropdims(sum(f.data; dims = ntuple(identity, Val(NX))); dims = ntuple(identity, Val(NX)))
        m ./ (sum(m) * DT(prod(grid.delta[(NX+1):(NX+NV)])))
    else
        throw(ArgumentError("profile must be :maxwellian or :mean"))
    end
    ctx = GradientDriveContext(
        ntuple(i -> E[i].data, Val(3)),
        map(backend_vector, grid.vaxes),
        P,
        sizes_x,
        sizes_v,
        grid.Bdir,
        DT(dt * n0 / grid.b0),
        DT(kappa_n),
        DT(kappa_T),
    )
    k! = gradient_drive_kernel!(exec)
    k!(f.data, ctx; ndrange = length(f.data))
    KernelAbstractions.synchronize(exec)
    return f
end

add_gradient_drive!(sp::DistSpecies, args...; kwargs...) = add_gradient_drive!(sp.dist, args...; kwargs...)

# --- BGK (Krook) relaxation toward a Maxwellian at rest ----------------------------------------

struct BGKContext{NT,VAT,SXT,SVT,DT}
    n::NT
    vaxes::VAT
    sizes_x::SXT
    sizes_v::SVT
    decay::DT
    inv2T::DT
    norm::DT
end

Adapt.@adapt_structure BGKContext

@kernel function bgk_kernel!(fdata, ctx)
    I = @index(Global, Linear)
    ixs, ivs = index_1d_to_combined(I, ctx.sizes_x, ctx.sizes_v)
    v2 = zero(eltype(fdata))
    @inbounds for d = 1:length(ivs)
        v = ctx.vaxes[d][ivs[d]]
        v2 += v * v
    end
    @inbounds target = ctx.n[ixs...] * exp(-v2 * ctx.inv2T) * ctx.norm
    @inbounds fdata[I] = target + (fdata[I] - target) * ctx.decay
end

"""
    add_bgk!(f, grid, ν, dt; T = 1)

Exact BGK (Krook) step over dt toward a Maxwellian at rest with the local density:
f ← n F̂_T + (f − n F̂_T) e^{−ν dt}, n = ∫f d^NV v, F̂_T ∝ exp(−v²/2T) normalised to 1 on the velocity grid (so n is
conserved to round-off). Velocities and T in the species' thermal units (T = 1: the species' own temperature;
charge exchange with neutrals: T = T_n/T_s). A Maxwellian at rest is invariant under the gyrophase rotation of the
storage frame, so no phase is needed. In Boussinesq form f = F_M + δf (T = 1) this is −ν(δf − δn F_M), the
number-conserving operator of KineticDR `BGK`.
"""
function add_bgk!(
    f::DistributionGrid{DT,NX,NV,NXNV,Cart},
    grid::CartGrid,
    ν,
    dt;
    T = 1,
    exec = bslLD.backend(),
) where {DT,NX,NV,NXNV}
    ν >= 0 && T > 0 || throw(ArgumentError("need ν ≥ 0 and T > 0"))
    iszero(ν) && return f
    sizes_x, sizes_v = _cartesian_axis_sizes(grid)
    n = compute_density(f, grid).data
    dv = prod(grid.delta[(NX+1):(NX+NV)])
    norm = inv(prod(sum(exp.(-Array(ax) .^ 2 ./ (2T))) for ax in grid.vaxes) * dv)
    ctx = BGKContext(n, map(backend_vector, grid.vaxes), sizes_x, sizes_v, DT(exp(-ν * dt)), DT(1 / (2T)), DT(norm))
    k! = bgk_kernel!(exec)
    k!(f.data, ctx; ndrange = length(f.data))
    KernelAbstractions.synchronize(exec)
    return f
end

add_bgk!(sp::DistSpecies, args...; kwargs...) = add_bgk!(sp.dist, args...; kwargs...)
