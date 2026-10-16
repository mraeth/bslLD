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

add_kappaT!(sp::Species, args...; kwargs...) = add_kappaT!(sp.dist, args...; kwargs...)

"""
    deposit!(f, grid, target, t; weight, dt)

Add `weight * dt * target(grid, t)` to the distribution function `f` (a `Species` or a
distribution). `target(grid, t)` returns an array of the size of `f.data`.

In the time loop, call it with a time-dependent weight to build `f` from a source, e.g.
`deposit!(f, grid, target, t; weight = ramp(t), dt = dt/2)`. Once before the loop it sets the
initial condition: `deposit!(f, grid, target, 0.0; weight = 1, dt = 1)` on an `f` that starts at
zero. `target` is not evaluated if `weight * dt` is zero.
"""
function deposit!(f::DistributionGrid, grid::Grid, target, t; weight, dt)
    w = weight * dt
    iszero(w) && return f
    add = target(grid, t)
    size(add) == size(f.data) || throw(
        DimensionMismatch(
            "target(grid, t) has size $(size(add)), the distribution function has $(size(f.data))",
        ),
    )
    f.data .+= w .* add
    return f
end

function deposit!(sp::Species, args...; kwargs...)
    deposit!(sp.dist, args...; kwargs...)
    return sp
end

"""
    gaussian_ramp(t_mid, σ) -> (t -> weight)

Gaussian weight centred at `t_mid` with standard deviation `σ`, restricted to the window
`0 ≤ t < 2 t_mid` and normalised so that its integral over the window is 1 (bsl6d's
`rampTime/2`, `rampSTD`). Zero outside the window.
"""
function gaussian_ramp(t_mid, σ)
    t_mid > 0 && σ > 0 || throw(ArgumentError("t_mid and σ must be positive"))
    norm = erf(t_mid / (σ * sqrt(2)))          # ∫ over the window of the unit Gaussian pdf
    return t ->
        (0 <= t < 2 * t_mid) ? exp(-(t - t_mid)^2 / (2 * σ^2)) / (σ * sqrt(2pi) * norm) :
        zero(float(t))
end
