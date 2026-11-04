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
