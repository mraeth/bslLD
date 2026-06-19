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

@inline _Ecomp(Ei, ixs) = Ei[ixs[1]]

@inline function _xhat_dot_ExB(ctx, ixs)
    ctx.Bdir == 1 && return zero(_Ecomp(ctx.E[1], ixs))       # B = (1,0,0)
    ctx.Bdir == 2 && return -_Ecomp(ctx.E[3], ixs)            # B = (0,1,0)
    return _Ecomp(ctx.E[2], ixs)                              # B = (0,0,1)
end

@kernel function kappaT_kernel!(fdata, ctx)
    I = @index(Global, Linear)
    ixs, ivs = index_1d_to_combined(I, ctx.sizes_x, ctx.sizes_v)

    v2 = zero(eltype(fdata))
    @inbounds for d in 1:length(ivs)
        v = ctx.vaxes[d][ivs[d]]
        v2 += v*v
    end

    @inbounds fdata[I] += ctx.kappa_T * ctx.dt *
                          _xhat_dot_ExB(ctx, ixs) *
                          (v2/2 - 3/2)
end

function add_kappaT!(
    f::DistributionGrid{Float64,NX,NV,NXNV,Cart},
    grid::CartGrid,
    dt,
    kappa_T,
    E::VectorField;
    exec = bslLD.backend(),
) where {NX,NV,NXNV}

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