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

struct XShiftContext{GT,KT,VAT,SXT,SVT,PT,VTT,EST,FT}
    grid::GT
    k::KT
    vaxes::VAT
    sizes_x::SXT
    sizes_v::SVT
    dir::Int
    phi::PT
    dt::PT
    delta::PT
    vth::VTT
    electric_scale::EST
    vflip::FT  # which velocity components a mirror on this axis reverses
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
    delta::PT
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

# Departure-point characteristics, shared by the Fourier and Lagrange paths.
@inline function _x_displacement(ctx::XShiftContext, ivs)
    xdisp = zero(eltype(ctx.k))
    rotation = R(ctx.grid.Bdir, -ctx.electric_scale*ctx.phi)
    for dv = 1:length(ivs)
        xdisp += ctx.vaxes[dv][ivs[dv]] * rotation[ctx.dir, dv]
    end
    return xdisp
end

@inline function compute_x_multiplier(ctx::XShiftContext, index::Int)
    ixs, ivs = index_1d_to_combined(index, ctx.sizes_x, ctx.sizes_v)
    return cis(-ctx.dt * ctx.k[ixs[ctx.dir]] * ctx.vth * _x_displacement(ctx, ivs))
end

@inline function _v_displacement(ctx::VShiftContext, ixs)
    delta_v = zero(eltype(ctx.k))
    rotation = R(ctx.grid.Bdir, ctx.electric_scale*ctx.phi)
    for field_dir = 1:length(ctx.e_components)
        delta_v += ctx.e_components[field_dir][ixs...] * rotation[ctx.dir, field_dir]
    end
    return delta_v
end

@inline function compute_v_multiplier(ctx::VShiftContext, index::Int)
    ixs, ivs = index_1d_to_combined(index, ctx.sizes_x, ctx.sizes_v)
    return cis(-ctx.dt * ctx.k[ivs[ctx.dir]] * ctx.electric_scale * _v_displacement(ctx, ixs))
end

# --- AdvectionPlan: pre-allocated buffers and cached data for zero-allocation advection ---

struct AdvectionPlan{FB,RB,PXF,PXI,PVF,PVI,KXT,KVT,VAT,BK}
    ff_buf::FB   # Complex{Float64} work buffer, same shape as f.data
    rbuf::RB  # real work buffer for out-of-place gathers (Lagrange path)
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
    rbuf = similar(f.data)
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
    return AdvectionPlan(
        ff_buf, rbuf, fwd_x, inv_x, fwd_v, inv_v, kx, kv, vaxes, bslLD.backend())
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

# --- Interpolation methods ------------------------------------------------

"""
    Fourier()

Spectral shift. Exact for periodic data (`|g| == 1` at every wavenumber), and
the default for every direction.
"""
struct Fourier end

"""
    Lagrange(W)  ==  Lagrange{W}()

Backward-semi-Lagrangian shift by centred Lagrange interpolation of even
stencil width `W`. The width is a dispatch parameter, so each `W` compiles to
its own fully unrolled kernel. All directions are treated as periodic.
"""
struct Lagrange{W} end
Lagrange(W::Integer) = Lagrange{Int(W)}()
@inline stencil_order(::Lagrange{W}) where {W} = Val(W)

# Departure-point displacement in cells along the advected axis. It is constant
# along that axis -- it depends only on the *other* indices -- so the line and
# cached sweeps below compute the weights once per line instead of once per point.
@inline _shift_cells(ctx::XShiftContext, ixs, ivs) =
    ctx.dt * ctx.vth * _x_displacement(ctx, ivs) / ctx.delta
@inline _shift_cells(ctx::VShiftContext, ixs, ivs) =
    ctx.dt * ctx.electric_scale * _v_displacement(ctx, ixs) / ctx.delta

@inline _line_length(ctx::XShiftContext) = ctx.sizes_x[ctx.dir]
@inline _line_length(ctx::VShiftContext) = ctx.sizes_v[ctx.dir]

@inline _line_index(ctx::XShiftContext, ixs, ivs) = ixs[ctx.dir]
@inline _line_index(ctx::VShiftContext, ixs, ivs) = ivs[ctx.dir]

@inline _line_linear_index(ctx::XShiftContext, ixs, ivs, i) =
    index_combined_to_1d(Base.setindex(ixs, i, ctx.dir), ivs, ctx.sizes_x, ctx.sizes_v)
@inline _line_linear_index(ctx::VShiftContext, ixs, ivs, i) =
    index_combined_to_1d(ixs, Base.setindex(ivs, i, ctx.dir), ctx.sizes_x, ctx.sizes_v)

# Linear-index stride between neighbours along the advected axis. The product of the
# sizes before `dir` is written as a loop: slicing the tuple with a runtime length
# allocates and does not compile for GPU kernels.
@inline function _prod_before(sizes::Tuple, dir)
    p = one(eltype(sizes))
    for d = 1:length(sizes)
        d < dir && (p *= sizes[d])
    end
    return p
end
@inline _line_stride(ctx::XShiftContext) = _prod_before(ctx.sizes_x, ctx.dir)
@inline _line_stride(ctx::VShiftContext) =
    prod(ctx.sizes_x) * _prod_before(ctx.sizes_v, ctx.dir)

# The multi-index of line `L`, with the advected coordinate set to 1.
@inline _line_sizes(ctx::XShiftContext) = (Base.setindex(ctx.sizes_x, 1, ctx.dir), ctx.sizes_v)
@inline _line_sizes(ctx::VShiftContext) = (ctx.sizes_x, Base.setindex(ctx.sizes_v, 1, ctx.dir))

# Whether the shift is constant along the advected line, so that the weights can be
# computed once per line. True for every context defined here.
@inline _constant_along_line(ctx) = true

# --- Boundary conditions --------------------------------------------------

"""
    Periodic()

Wrap the stencil around the advected axis. The default, and the only condition
the spectral path can represent.
"""
struct Periodic end

"""
    Mirror()

Specular reflection at the two end nodes of a *spatial* axis, following bsl6d
(`105-implement-mirror-boundary-conditions`): the halo is filled by reflecting
the interior about the boundary node while reversing the velocity components
that the reflection flips, and the boundary node itself is symmetrised so that
`f(x_b, v) == f(x_b, R v)`.

With `B` along `grid.Bdir`, a mirror on an axis perpendicular to `B` reverses
both gyration-plane velocities; a mirror along `B` reverses the field-aligned
one. Requires a velocity axis symmetric about zero (`v[end+1-j] == -v[j]`),
which is checked when the advection is set up.

The mirror planes sit at the first and last grid node, `x[1]` and `x[end]`.
bslLD's Cartesian x-axis excludes its upper endpoint, so the reflected domain
spans `(n-1)*delta`, one cell short of `L`; construct the axis accordingly.
"""
struct Mirror end

# Velocity components reversed by a mirror on spatial axis `dir`.
@inline _mirror_flip(Bdir::Int, dir::Int, ::Val{NV}) where {NV} =
    ntuple(d -> dir == Bdir ? d == Bdir : d != Bdir, Val(NV))

@inline function _flip_v(ctx, ivs::NTuple{NV,Int}) where {NV}
    return ntuple(d -> ctx.vflip[d] ? ctx.sizes_v[d] + 1 - ivs[d] : ivs[d], Val(NV))
end

# Linear index of the stencil node `i` along the advected axis, under `bc`.
@inline _sample_index(::Periodic, ctx, ixs, ivs, i, n) =
    _line_linear_index(ctx, ixs, ivs, mod1(i, n))

@inline function _sample_index(::Mirror, ctx::XShiftContext, ixs, ivs, i, n)
    if i < 1
        j = min(2 - i, n)                       # reflect about node 1
        return index_combined_to_1d(
            Base.setindex(ixs, j, ctx.dir), _flip_v(ctx, ivs), ctx.sizes_x, ctx.sizes_v)
    elseif i > n
        j = max(2n - i, 1)                      # reflect about node n
        return index_combined_to_1d(
            Base.setindex(ixs, j, ctx.dir), _flip_v(ctx, ivs), ctx.sizes_x, ctx.sizes_v)
    else
        return _line_linear_index(ctx, ixs, ivs, i)
    end
end

# f(x_b, v) <- (f(x_b, v) + f(x_b, R v)) / 2 on the two boundary nodes. Each
# velocity pair is handled once, by the thread holding its lower linear index,
# so the in-place update needs no scratch buffer.
@kernel function mirror_symmetrize_kernel!(fdata, ctx)
    I = @index(Global, Linear)
    ixs, ivs = index_1d_to_combined(I, ctx.sizes_x, ctx.sizes_v)
    n = _line_length(ctx)
    i = ixs[ctx.dir]
    if i == 1 || i == n
        J = index_combined_to_1d(ixs, _flip_v(ctx, ivs), ctx.sizes_x, ctx.sizes_v)
        if I <= J
            @inbounds mean = (fdata[I] + fdata[J]) / 2
            @inbounds fdata[I] = mean
            @inbounds fdata[J] = mean
        end
    end
end

@kernel function lagrange_shift_kernel!(dst, @Const(src), ctx, order::Val, bc)
    I = @index(Global, Linear)
    ixs, ivs = index_1d_to_combined(I, ctx.sizes_x, ctx.sizes_v)

    cells, alpha = shift_split(_shift_cells(ctx, ixs, ivs))
    w = lagrange_weights(order, alpha, eltype(dst))

    n = _line_length(ctx)
    base = _line_index(ctx, ixs, ivs) - cells - length(w) ÷ 2 - 1

    acc = zero(eltype(dst))
    @inbounds for m = 1:length(w)
        acc += w[m] * src[_sample_index(bc, ctx, ixs, ivs, base + m, n)]
    end
    @inbounds dst[I] = acc
end

# Stencil node `j` of a line under `bc`, for the line and cached sweeps. `I0 + j*st` is the
# linear index of node `j`; nodes outside 1:n go through `_sample_index` (wrap or reflect).
@inline _sample_line(::Periodic, src, ctx, ixs, ivs, I0, st, j, n) =
    @inbounds src[I0+mod1(j, n)*st]
@inline _sample_line(bc, src, ctx, ixs, ivs, I0, st, j, n) =
    1 <= j <= n ? (@inbounds src[I0+j*st]) : @inbounds src[_sample_index(bc, ctx, ixs, ivs, j, n)]

# One thread per line: weights once, then a strided sweep along the line. Interior
# stencil nodes are read by stride arithmetic; only halo nodes go through `_sample_line`.
# Fast when there are many lines; for few lines (e.g. 1D1V) it leaves the device idle.
@kernel function lagrange_line_kernel!(dst, @Const(src), ctx, order::Val{W}, bc) where {W}
    L = @index(Global, Linear)
    lsx, lsv = _line_sizes(ctx)
    ixs, ivs = index_1d_to_combined(L, lsx, lsv)

    cells, alpha = shift_split(_shift_cells(ctx, ixs, ivs))
    w = lagrange_weights(order, alpha, eltype(dst))

    n = _line_length(ctx)
    st = _line_stride(ctx)
    I0 = _line_linear_index(ctx, ixs, ivs, 1) - st          # I0 + j*st is node j
    off = -cells - W ÷ 2 - 1
    @inbounds for i = 1:n
        base = i + off
        acc = zero(eltype(dst))
        if base >= 0 && base + W <= n
            for m = 1:W
                acc += w[m] * src[I0+(base+m)*st]
            end
        else
            for m = 1:W
                acc += w[m] * _sample_line(bc, src, ctx, ixs, ivs, I0, st, base + m, n)
            end
        end
        dst[I0+i*st] = acc
    end
end

# Per-line tables (integer cell shift, W stencil weights) for the cached sweep, reused
# between calls of the same size.
_order_width(::Val{W}) where {W} = W
const _LINE_TABLES = Dict{Any,Any}()
function _line_tables(dst, W, nl, exec)
    return get!(_LINE_TABLES, (typeof(dst), W, nl)) do
        (
            KernelAbstractions.allocate(exec, Int, nl),
            KernelAbstractions.allocate(exec, eltype(dst), W, nl),
        )
    end
end

# Setup of the cached sweep: one thread per line computes its cell shift and weights.
@kernel function lagrange_line_setup_kernel!(cells, wts, ctx, order::Val{W}) where {W}
    L = @index(Global, Linear)
    lsx, lsv = _line_sizes(ctx)
    ixs, ivs = index_1d_to_combined(L, lsx, lsv)
    c, alpha = shift_split(_shift_cells(ctx, ixs, ivs))
    w = lagrange_weights(order, alpha, eltype(wts))
    cells[L] = c
    for m = 1:W
        wts[m, L] = w[m]
    end
end

# Cached sweep: one thread per point (coalesced access, full parallelism), weights read
# from the per-line table. Same arithmetic, in the same order, as the other kernels.
@kernel function lagrange_cached_kernel!(
    dst,
    @Const(src),
    @Const(cells),
    @Const(wts),
    ctx,
    order::Val{W},
    bc,
    st::Int,
    n::Int,
) where {W}
    p = @index(Global, Linear)
    q = p - 1
    j = (q ÷ st) % n + 1
    L = q % st + st * (q ÷ (st * n)) + 1
    I0 = p - j * st
    base = j - @inbounds(cells[L]) - W ÷ 2 - 1
    acc = zero(eltype(dst))
    if base >= 0 && base + W <= n
        @inbounds for m = 1:W
            acc += wts[m, L] * src[I0+(base+m)*st]
        end
    else
        lsx, lsv = _line_sizes(ctx)
        ixs, ivs = index_1d_to_combined(L, lsx, lsv)
        @inbounds for m = 1:W
            acc += wts[m, L] * _sample_line(bc, src, ctx, ixs, ivs, I0, st, base + m, n)
        end
    end
    @inbounds dst[p] = acc
end

"""
Kernel used by the Lagrange sweeps, `bslLD._SWEEP_MODE[]`:

  * `:auto` (default): `:line` if the sweep has at least `_AUTO_MIN_LINES[]` lines, `:cached` otherwise;
  * `:cached`: per-line weight table, one thread per point;
  * `:line`: one thread per line;
  * `:point`: weights recomputed for every point.

All modes give bit-identical results.
"""
const _SWEEP_MODE = Ref(:auto)
const _AUTO_MIN_LINES = Ref(100_000)

function _lagrange_sweep!(dst, src, ctx, method, bc, exec)
    order = stencil_order(method)
    n = _line_length(ctx)
    nl = length(src) ÷ n
    mode = _SWEEP_MODE[]
    if mode === :auto
        mode = nl >= _AUTO_MIN_LINES[] ? :line : :cached
    end
    if mode === :cached && _constant_along_line(ctx)
        cells, wts = _line_tables(dst, _order_width(order), nl, exec)
        lagrange_line_setup_kernel!(exec)(cells, wts, ctx, order; ndrange = nl)
        lagrange_cached_kernel!(exec)(
            dst, src, cells, wts, ctx, order, bc, _line_stride(ctx), n; ndrange = length(src))
    elseif mode === :line && _constant_along_line(ctx)
        lagrange_line_kernel!(exec)(dst, src, ctx, order, bc; ndrange = nl)
    else
        lagrange_shift_kernel!(exec)(dst, src, ctx, order, bc; ndrange = length(src))
    end
    KernelAbstractions.synchronize(exec)
    return dst
end

# --- Direction drivers ----------------------------------------------------

function _x_context(sp::Species, grid::CartGrid, simTime::SimulationTime, dir, plan)
    DT = eltype(sp.dist.data)
    sizes_x, sizes_v = _cartesian_axis_sizes(grid)
    NV = length(grid.vaxes)
    return XShiftContext(
        grid,
        plan.kx[dir],
        plan.vaxes,
        sizes_x,
        sizes_v,
        dir,
        DT(simTime.phase),
        DT(_effective_dt(simTime)),
        DT(grid.delta[dir]),
        DT(thermal_velocity(sp)),
        DT(electric_acceleration_scale(sp)),
        _mirror_flip(grid.Bdir, dir, Val(NV)),
    )
end

function _v_context(
    sp::Species,
    grid::CartGrid,
    simTime::SimulationTime,
    e::VectorField,
    dir,
    plan,
)
    DT = eltype(sp.dist.data)
    NX = length(grid.xaxes)
    sizes_x, sizes_v = _cartesian_axis_sizes(grid)
    return VShiftContext(
        grid,
        ntuple(i -> e[i].data, Val(length(grid.vaxes))),
        plan.kv[dir],
        sizes_x,
        sizes_v,
        dir,
        DT(simTime.phase),
        DT(_effective_dt(simTime)),
        DT(grid.delta[NX+dir]),
        DT(electric_acceleration_scale(sp)),
    )
end

function _advect_dir!(f, ctx, plan, ::Fourier, ::Periodic, fwd_plan, inv_plan)
    exec = plan.backend
    _apply_phase_shift!(
        f,
        plan.ff_buf,
        fwd_plan,
        inv_plan,
        spectral_multiply_kernel!(exec),
        ctx,
        exec,
    )
    return nothing
end

function _advect_dir!(f, ctx, plan, method::Lagrange, bc, _fwd_plan, _inv_plan)
    exec = plan.backend
    if bc isa Mirror
        sym! = mirror_symmetrize_kernel!(exec)
        sym!(f.data, ctx; ndrange = length(f.data))
        KernelAbstractions.synchronize(exec)
    end
    _lagrange_sweep!(plan.rbuf, f.data, ctx, method, bc, exec)
    copyto!(f.data, plan.rbuf)
    return nothing
end

_advect_dir!(f, ctx, plan, ::Fourier, bc, fwd, inv) = throw(
    ArgumentError("$(typeof(bc)) boundaries need a Lagrange method; Fourier is periodic"))

# A mirror pairs v with -v by index, which is only the physical reflection when
# the velocity axis is symmetric about zero.
function _check_mirror_axes(grid::CartGrid, dir::Int, ::Val{W}) where {W}
    n = length(grid.xaxes[dir])
    W ÷ 2 < n || throw(ArgumentError(
        "Lagrange half-stencil $(W ÷ 2) does not fit in axis $dir of length $n"))
    for (d, ax) in enumerate(grid.vaxes)
        isapprox(first(ax) + last(ax), 0, atol = 1e-12 * max(abs(first(ax)), 1)) ||
            throw(ArgumentError(
                "mirror boundaries need velocity axis $d symmetric about zero, " *
                "got [$(first(ax)), $(last(ax))]"))
    end
    return nothing
end

function _advect_x_dir!(
    sp::Species,
    grid::CartGrid,
    simTime::SimulationTime,
    dir::Int,
    plan::AdvectionPlan,
    method,
    bc,
)
    NX = length(grid.xaxes)
    1 <= dir <= NX || throw(ArgumentError("advectX! direction $dir out of 1:$NX"))
    bc isa Mirror &&
        method isa Lagrange &&
        _check_mirror_axes(grid, dir, stencil_order(method))
    _advect_dir!(
        sp.dist,
        _x_context(sp, grid, simTime, dir, plan),
        plan,
        method,
        bc,
        plan.fwd_x[dir],
        plan.inv_x[dir],
    )
    return nothing
end

function _advect_v_dir!(
    sp::Species,
    grid::CartGrid,
    simTime::SimulationTime,
    e::VectorField,
    dir::Int,
    plan::AdvectionPlan,
    method,
    bc,
)
    NV = length(grid.vaxes)
    1 <= dir <= NV || throw(ArgumentError("advectV! direction $dir out of 1:$NV"))
    bc isa Periodic || throw(ArgumentError(
        "mirror boundaries apply to spatial axes only, as in bsl6d"))
    _advect_dir!(
        sp.dist,
        _v_context(sp, grid, simTime, e, dir, plan),
        plan,
        method,
        bc,
        plan.fwd_v[dir],
        plan.inv_v[dir],
    )
    return nothing
end

function advectX!(
    sp::Species,
    grid::CartGrid,
    simTime::SimulationTime;
    method = Fourier(),
    boundary = Periodic(),
)
    plan = _get_plan(sp.dist, grid)
    for dir = 1:length(grid.xaxes)
        _advect_x_dir!(sp, grid, simTime, dir, plan, method, boundary)
    end
end

function advectX!(
    sp::Species,
    grid::CartGrid,
    simTime::SimulationTime,
    dir::Int;
    method = Fourier(),
    boundary = Periodic(),
)
    _advect_x_dir!(sp, grid, simTime, dir, _get_plan(sp.dist, grid), method, boundary)
end

function advectV!(
    sp::Species,
    grid::CartGrid,
    simTime::SimulationTime,
    e::VectorField;
    method = Fourier(),
    boundary = Periodic(),
)
    plan = _get_plan(sp.dist, grid)
    for dir = 1:length(grid.vaxes)
        _advect_v_dir!(sp, grid, simTime, e, dir, plan, method, boundary)
    end
end

function advectV!(
    sp::Species,
    grid::CartGrid,
    simTime::SimulationTime,
    e::VectorField,
    dir::Int;
    method = Fourier(),
    boundary = Periodic(),
)
    _advect_v_dir!(sp, grid, simTime, e, dir, _get_plan(sp.dist, grid), method, boundary)
end
