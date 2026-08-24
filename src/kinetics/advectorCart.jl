using StaticArrays


function R(id::Int, phi::Real)
    c, s = cos(phi), sin(phi)
    z, o = zero(c), one(c)
    if id == 1
        # Rotation around X-axis
        return SMatrix{3,3}(o, z, z, z, c, s, z, -s, c)
    elseif id == 2
        # Rotation around Y-axis
        return SMatrix{3,3}(c, z, -s, z, o, z, s, z, c)
    else
        # Rotation around Z-axis
        return SMatrix{3,3}(c, s, z, -s, c, z, z, z, o)
    end
end

backend_vector(values) = bslLD.backend_array(collect(values))

@inline _effective_dt(simTime::SimulationTime) = simTime.dt * simTime.fraction_dt

@inline _cartesian_axis_sizes(grid::CartGrid) =
    (Tuple(length.(grid.xaxes)), Tuple(length.(grid.vaxes)))

struct XShiftContext{GT,KT,VAT,SXT,SVT,PT,VTT,EST}
    grid::GT
    k::KT
    vaxes::VAT
    sizes_x::SXT
    sizes_v::SVT
    dir::Int
    phi::PT
    dt::PT
    vth::VTT
    electric_scale::EST
end

struct VShiftContext{GT,ET,KT,SXT,SVT,PT,EST}
    grid::GT
    e_components::ET
    k::KT
    sizes_x::SXT
    sizes_v::SVT
    dir::Int
    phi::PT
    dt::PT
    electric_scale::EST
end

Adapt.@adapt_structure XShiftContext
Adapt.@adapt_structure VShiftContext

@inline function (ctx::XShiftContext)(index::Int)
    return compute_x_multiplier(ctx, index)
end

@inline function (ctx::VShiftContext)(index::Int)
    return compute_v_multiplier(ctx, index)
end

@inline function compute_x_multiplier(ctx::XShiftContext, index::Int)
    ixs, ivs = index_1d_to_combined(index, ctx.sizes_x, ctx.sizes_v)

    xdisp = zero(eltype(ctx.k))
    rotation = R(ctx.grid.Bdir, -ctx.electric_scale*ctx.phi)
    for dv = 1:length(ivs)
        xdisp += ctx.vaxes[dv][ivs[dv]] * rotation[ctx.dir, dv]
    end

    return cis(-ctx.dt * ctx.k[ixs[ctx.dir]] * ctx.vth * xdisp)
end

@inline function compute_v_multiplier(ctx::VShiftContext, index::Int)
    ixs, ivs = index_1d_to_combined(index, ctx.sizes_x, ctx.sizes_v)

    delta_v = zero(eltype(ctx.k))
    rotation = R(ctx.grid.Bdir, ctx.electric_scale*ctx.phi)
    for field_dir = 1:length(ctx.e_components)
        delta_v += ctx.e_components[field_dir][ixs...] * rotation[ctx.dir, field_dir]
    end

    return cis(-ctx.dt * ctx.k[ivs[ctx.dir]] * ctx.electric_scale * delta_v)
end

# --- AdvectionPlan: pre-allocated buffers and cached data for zero-allocation advection ---

struct AdvectionPlan{FB,PXF,PXI,PVF,PVI,KXT,KVT,VAT,BK}
    ff_buf::FB   # Complex{Float64} work buffer, same shape as f.data
    fwd_x::PXF  # ntuple of in-place forward FFT plans, one per x-dim
    inv_x::PXI  # ntuple of in-place inverse FFT plans, one per x-dim
    fwd_v::PVF  # ntuple of in-place forward FFT plans, one per v-dim
    inv_v::PVI  # ntuple of in-place inverse FFT plans, one per v-dim
    kx::KXT  # ntuple of wavenumber arrays for x-dims (on backend)
    kv::KVT  # ntuple of wavenumber arrays for v-dims (on backend)
    vaxes::VAT  # ntuple of velocity axes already copied to backend
    backend::BK
end

function AdvectionPlan(
    f::DistributionGrid{DT,NX,NV,NXNV,Cart},
    grid::CartGrid,
) where {DT,NX,NV,NXNV}
    ff_buf = similar(f.data, Complex{DT})
    fwd_x = ntuple(d -> plan_fft!(ff_buf, d), Val(NX))
    inv_x = ntuple(d -> plan_ifft!(ff_buf, d), Val(NX))
    fwd_v = ntuple(d -> plan_fft!(ff_buf, NX + d), Val(NV))
    inv_v = ntuple(d -> plan_ifft!(ff_buf, NX + d), Val(NV))
    kx = ntuple(Val(NX)) do d
        n = size(f.data, d)
        k = similar(f.data, DT, n)
        copyto!(k, DT.((2pi / (n * grid.delta[d])) .* fftfreq(n, n)))
        k
    end
    kv = ntuple(Val(NV)) do d
        n = size(f.data, NX + d)
        k = similar(f.data, DT, n)
        copyto!(k, DT.((2pi / (n * grid.delta[NX+d])) .* fftfreq(n, n)))
        k
    end
    vaxes = map(backend_vector, grid.vaxes)
    return AdvectionPlan(ff_buf, fwd_x, inv_x, fwd_v, inv_v, kx, kv, vaxes, bslLD.backend())
end

const _plan_cache = IdDict{Any,Dict{Any,AdvectionPlan}}()
const _plan_cache_lock = ReentrantLock()

@inline _plan_cache_key(grid::CartGrid) = (grid.xaxes, grid.vaxes, grid.delta, grid.Bdir)

function _get_plan(
    f::DistributionGrid{DT,NX,NV,NXNV,Cart},
    grid::CartGrid,
) where {DT,NX,NV,NXNV}
    lock(_plan_cache_lock) do
        plans_for_data = get!(() -> Dict{Any,AdvectionPlan}(), _plan_cache, f.data)
        get!(() -> AdvectionPlan(f, grid), plans_for_data, _plan_cache_key(grid))
    end
end

function _apply_phase_shift!(f, ff_buf, fwd_plan, inv_plan, kernel!, ctx, exec)
    @. ff_buf = f.data
    fwd_plan * ff_buf
    kernel!(ff_buf, ctx; ndrange = length(ff_buf))
    KernelAbstractions.synchronize(exec)
    inv_plan * ff_buf
    @. f.data = real(ff_buf)
    return nothing
end

function _advect_x_dir!(
    f::DistributionGrid{DT,NX,NV,NXNV,Cart},
    grid::CartGrid,
    simTime::SimulationTime,
    dir::Int,
    plan::AdvectionPlan,
) where {DT,NX,NV,NXNV}
    1 <= dir <= NX || throw(ArgumentError("advectX! direction $dir out of 1:$NX"))
    exec = plan.backend
    kernel! = spectral_multiply_kernel!(exec)
    sizes_x, sizes_v = _cartesian_axis_sizes(grid)
    ctx = XShiftContext(
        grid,
        plan.kx[dir],
        plan.vaxes,
        sizes_x,
        sizes_v,
        dir,
        DT(simTime.phase),
        DT(_effective_dt(simTime)),
        DT(thermal_velocity(f)),
        DT(electric_acceleration_scale(f)),
    )
    _apply_phase_shift!(
        f,
        plan.ff_buf,
        plan.fwd_x[dir],
        plan.inv_x[dir],
        kernel!,
        ctx,
        exec,
    )
    return nothing
end

function _advect_v_dir!(
    f::DistributionGrid{DT,NX,NV,NXNV,Cart},
    grid::CartGrid,
    simTime::SimulationTime,
    e::VectorField,
    dir::Int,
    plan::AdvectionPlan,
) where {DT,NX,NV,NXNV}
    1 <= dir <= NV || throw(ArgumentError("advectV! direction $dir out of 1:$NV"))
    exec = plan.backend
    kernel! = spectral_multiply_kernel!(exec)
    sizes_x, sizes_v = _cartesian_axis_sizes(grid)
    e_components = ntuple(i -> e[i].data, Val(NV))
    ctx = VShiftContext(
        grid,
        e_components,
        plan.kv[dir],
        sizes_x,
        sizes_v,
        dir,
        DT(simTime.phase),
        DT(_effective_dt(simTime)),
        DT(electric_acceleration_scale(f)),
    )
    _apply_phase_shift!(
        f,
        plan.ff_buf,
        plan.fwd_v[dir],
        plan.inv_v[dir],
        kernel!,
        ctx,
        exec,
    )
    return nothing
end

function advectX!(
    f::DistributionGrid{DT,NX,NV,NXNV,Cart},
    grid::CartGrid,
    simTime::SimulationTime,
) where {DT,NX,NV,NXNV}
    plan = _get_plan(f, grid)
    for dir = 1:NX
        _advect_x_dir!(f, grid, simTime, dir, plan)
    end
end

function advectX!(
    f::DistributionGrid{DT,NX,NV,NXNV,Cart},
    grid::CartGrid,
    simTime::SimulationTime,
    dir::Int,
) where {DT,NX,NV,NXNV}
    _advect_x_dir!(f, grid, simTime, dir, _get_plan(f, grid))
end

function advectV!(
    f::DistributionGrid{DT,NX,NV,NXNV,Cart},
    grid::CartGrid,
    simTime::SimulationTime,
    e::VectorField,
) where {DT,NX,NV,NXNV}
    plan = _get_plan(f, grid)
    for dir = 1:NV
        _advect_v_dir!(f, grid, simTime, e, dir, plan)
    end
end

function advectV!(
    f::DistributionGrid{DT,NX,NV,NXNV,Cart},
    grid::CartGrid,
    simTime::SimulationTime,
    e::VectorField,
    dir::Int,
) where {DT,NX,NV,NXNV}
    _advect_v_dir!(f, grid, simTime, e, dir, _get_plan(f, grid))
end
