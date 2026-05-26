using StaticArrays


function R(id::Int, phi::Real)
    c, s = cos(phi), sin(phi)
    if id == 1
        # Rotation around X-axis
        return SMatrix{3,3}(1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c)
    elseif id == 2
        # Rotation around Y-axis
        return SMatrix{3,3}(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c)
    else
        # Rotation around Z-axis
        return SMatrix{3,3}(c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0)
    end
end

# Convert N-dimensional indices (1-based, column-major) to 1D index
function index_nd_to_1d(indices::NTuple{N, Int}, sizes::NTuple{N, Int}) where {N}
    i0 = 0
    stride = 1
    for k in 1:N
        i0 += (indices[k] - 1) * stride
        stride *= sizes[k]
    end
    return i0 + 1
end

@inline function _index_1d_to_nd(i0::Int, sizes::NTuple{0, Int})
    return ()
end

@inline function _index_1d_to_nd(i0::Int, sizes::NTuple{N, Int}) where {N}
    idx = i0 % sizes[1] + 1
    return (idx, _index_1d_to_nd(i0 ÷ sizes[1], Base.tail(sizes))...)
end

# Convert 1D index (1-based) to N-dimensional indices (column-major)
@inline function index_1d_to_nd(i::Int, sizes::NTuple{N, Int}) where {N}
    return _index_1d_to_nd(i - 1, sizes)
end

# Convert combined indices (X + Y) to 1D index
function index_combined_to_1d(
    indicesX::NTuple{M, Int},
    indicesY::NTuple{N, Int},
    sizesX::NTuple{M, Int},
    sizesY::NTuple{N, Int},
) where {M, N}
    iX = index_nd_to_1d(indicesX, sizesX)
    strideX = prod(sizesX)
    iY = index_nd_to_1d(indicesY, sizesY)
    return iX + (iY - 1) * strideX
end

# Convert 1D index back to (X, Y) indices
@inline function index_1d_to_combined(
    i::Int,
    sizesX::NTuple{M, Int},
    sizesY::NTuple{N, Int},
) where {M, N}
    strideX = prod(sizesX)
    iX = (i - 1) % strideX + 1
    iY = (i - 1) ÷ strideX + 1
    return (index_1d_to_nd(iX, sizesX), index_1d_to_nd(iY, sizesY))
end

backend_vector(values) = bslLD.backend_array(collect(values))

struct XShiftContext{GT, KT, VAT, SXT, SVT, PT}
    grid::GT
    k::KT
    vaxes::VAT
    sizes_x::SXT
    sizes_v::SVT
    dir::Int
    phi::PT
    dt::PT

end

struct VShiftContext{GT, ET, KT, SXT, SVT, PT}
    grid::GT
    e_components::ET
    k::KT
    sizes_x::SXT
    sizes_v::SVT
    dir::Int
    phi::PT
    dt::PT
end

Adapt.@adapt_structure XShiftContext
Adapt.@adapt_structure VShiftContext

@inline function (ctx::XShiftContext)(index::Int)
    return compute_x_phase(ctx, index)
end

@inline function (ctx::VShiftContext)(index::Int)
    return compute_v_phase(ctx, index)
end

@inline function compute_x_phase(ctx::XShiftContext, index::Int)
    ixs, ivs = index_1d_to_combined(index, ctx.sizes_x, ctx.sizes_v)

    xdisp = zero(eltype(ctx.k))
    rotation = R(ctx.grid.Bdir, ctx.phi)
    for dv in 1:length(ivs)
        xdisp += ctx.vaxes[dv][ivs[dv]] * rotation[ctx.dir, dv]
    end

    return (ctx.dt / ctx.grid.delta[ctx.dir]) * ctx.k[ixs[ctx.dir]] * xdisp
end

@inline function compute_v_phase(ctx::VShiftContext, index::Int)
    ixs, ivs = index_1d_to_combined(index, ctx.sizes_x, ctx.sizes_v)

    delta_v = zero(eltype(ctx.k))
    rotation = R(ctx.grid.Bdir, -ctx.phi)
    for field_dir in 1:length(ctx.e_components)
        delta_v += ctx.e_components[field_dir][ixs[1]] * rotation[ctx.dir, field_dir]
    end

    return (ctx.dt / ctx.grid.delta[length(ctx.sizes_x) + ctx.dir]) * ctx.k[ivs[ctx.dir]] * delta_v
end

@kernel function distribution_kernel!(ff, phase_context)
    i = @index(Global)

    if i <= length(ff)
        phase = phase_context(i)
        @inbounds ff[i] *= cis(-phase)
    end
end

function x_shift_context(grid::CartGrid, simTime::SimulationTime, k, dir::Int)
    sizes_x = Tuple(length.(grid.xaxes))
    sizes_v = Tuple(length.(grid.vaxes))
    vaxes = map(backend_vector, grid.vaxes)
    phi = simTime.phase
    dt = simTime.dt
    return XShiftContext(grid, k, vaxes, sizes_x, sizes_v, dir, phi, dt)
end

function v_shift_context(grid::CartGrid, simTime::SimulationTime, e::VectorField, k, dir::Int)
    sizes_x = Tuple(length.(grid.xaxes))
    sizes_v = Tuple(length.(grid.vaxes))
    e_components = Tuple(component.data for component in e)
    phi = simTime.phase
    dt = simTime.dt
    return VShiftContext(grid, e_components, k, sizes_x, sizes_v, dir, phi, dt)
end

# --- AdvectionPlan: pre-allocated buffers and cached data for zero-allocation advection ---

struct AdvectionPlan{FB, PXF, PXI, PVF, PVI, KXT, KVT, VAT, BK}
    ff_buf  :: FB   # Complex{Float64} work buffer, same shape as f.data
    fwd_x   :: PXF  # ntuple of in-place forward FFT plans, one per x-dim
    inv_x   :: PXI  # ntuple of in-place inverse FFT plans, one per x-dim
    fwd_v   :: PVF  # ntuple of in-place forward FFT plans, one per v-dim
    inv_v   :: PVI  # ntuple of in-place inverse FFT plans, one per v-dim
    kx      :: KXT  # ntuple of wavenumber arrays for x-dims (on backend)
    kv      :: KVT  # ntuple of wavenumber arrays for v-dims (on backend)
    vaxes   :: VAT  # ntuple of velocity axes already copied to backend
    backend :: BK
end

function AdvectionPlan(f::DistributionGrid{Float64,NX,NV,NXNV,Cart}, grid::CartGrid) where {NX,NV,NXNV}
    ff_buf = similar(f.data, Complex{Float64})
    fwd_x  = ntuple(d -> plan_fft!(ff_buf, d), Val(NX))
    inv_x  = ntuple(d -> plan_ifft!(ff_buf, d), Val(NX))
    fwd_v  = ntuple(d -> plan_fft!(ff_buf, NX + d), Val(NV))
    inv_v  = ntuple(d -> plan_ifft!(ff_buf, NX + d), Val(NV))
    kx     = ntuple(Val(NX)) do d
        k = similar(f.data, Float64, size(f.data, d))
        copyto!(k, collect(2pi .* fftfreq(size(f.data, d))))
        k
    end
    kv     = ntuple(Val(NV)) do d
        k = similar(f.data, Float64, size(f.data, NX + d))
        copyto!(k, collect(2pi .* fftfreq(size(f.data, NX + d))))
        k
    end
    vaxes  = map(backend_vector, grid.vaxes)
    return AdvectionPlan(ff_buf, fwd_x, inv_x, fwd_v, inv_v, kx, kv, vaxes, bslLD.backend())
end

const _plan_cache = WeakKeyDict{Any, AdvectionPlan}()

function _get_plan(f::DistributionGrid{Float64,NX,NV,NXNV,Cart}, grid::CartGrid) where {NX,NV,NXNV}
    get!(() -> AdvectionPlan(f, grid), _plan_cache, f)
end

function advectX!(f::DistributionGrid{Float64,NX,NV,NXNV,Cart}, grid::CartGrid, simTime::SimulationTime, plan::AdvectionPlan) where {NX,NV,NXNV}
    return _advect_x_planned!(f, grid, simTime, plan, plan.backend)
end

function _advect_x_planned!(f::DistributionGrid{Float64,NX,NV,NXNV,Cart}, grid::CartGrid, simTime::SimulationTime, plan::AdvectionPlan, exec) where {NX,NV,NXNV}
    kernel! = distribution_kernel!(exec)
    sizes_x = Tuple(length.(grid.xaxes))
    sizes_v = Tuple(length.(grid.vaxes))
    for dir in 1:NX
        @. plan.ff_buf = f.data
        plan.fwd_x[dir] * plan.ff_buf
        phi = simTime.phase
        dt = simTime.dt
        ctx = XShiftContext(grid, plan.kx[dir], plan.vaxes, sizes_x, sizes_v, dir, phi, dt)
        kernel!(plan.ff_buf, ctx; ndrange=length(plan.ff_buf))
        KernelAbstractions.synchronize(exec)
        plan.inv_x[dir] * plan.ff_buf
        @. f.data = real(plan.ff_buf)
    end
    return nothing
end

function advectV!(f::DistributionGrid{Float64,NX,NV,NXNV,Cart}, grid::CartGrid, simTime::SimulationTime,  e::VectorField, plan::AdvectionPlan) where {NX,NV,NXNV}
    return _advect_v_planned!(f, grid, simTime, e, plan, plan.backend)
end

function _advect_v_planned!(f::DistributionGrid{Float64,NX,NV,NXNV,Cart}, grid::CartGrid,simTime::SimulationTime, e::VectorField, plan::AdvectionPlan, exec) where {NX,NV,NXNV}
    kernel! = distribution_kernel!(exec)
    sizes_x = Tuple(length.(grid.xaxes))
    sizes_v = Tuple(length.(grid.vaxes))
    e_components = ntuple(i -> e[i].data, Val(NV))
    for dir in 1:NV
        @. plan.ff_buf = f.data
        plan.fwd_v[dir] * plan.ff_buf
        phi = simTime.phase
        dt = simTime.dt
        ctx = VShiftContext(grid, e_components, plan.kv[dir], sizes_x, sizes_v, dir, phi, dt)
        kernel!(plan.ff_buf, ctx; ndrange=length(plan.ff_buf))
        KernelAbstractions.synchronize(exec)
        plan.inv_v[dir] * plan.ff_buf
        @. f.data = real(plan.ff_buf)
    end
    return nothing
end

function advectX!(f::DistributionGrid{Float64,NX,NV,NXNV,Cart}, grid::CartGrid, simTime::SimulationTime) where {NX,NV,NXNV}
    advectX!(f, grid, simTime, _get_plan(f, grid))
end

function advectV!(f::DistributionGrid{Float64,NX,NV,NXNV,Cart}, grid::CartGrid, simTime::SimulationTime, e::VectorField) where {NX,NV,NXNV}
    advectV!(f, grid, simTime, e, _get_plan(f, grid))
end
