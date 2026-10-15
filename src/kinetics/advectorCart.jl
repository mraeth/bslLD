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
    return cis(
        -ctx.dt * ctx.k[ivs[ctx.dir]] * ctx.electric_scale * _v_displacement(ctx, ixs),
    )
end

# --- AdvectionPlan: cached data and lazily allocated buffers for advection ---
#
# `buffers` holds everything allocated on first use: the FFT work buffer and plans (only
# `Fourier()`), the real work buffer (only Lagrange sweeps that cannot run in place), the
# shift/weight tables and the mirror halos. A Lagrange run with the tiled sweep therefore
# holds no copy of f, and `release_advection_buffers!` frees all of it.

struct AdvectionPlan{KXT,KVT,VAT,BK}
    kx::KXT  # ntuple of wavenumber arrays for x-dims (on backend)
    kv::KVT  # ntuple of wavenumber arrays for v-dims (on backend)
    vaxes::VAT  # ntuple of velocity axes already copied to backend
    backend::BK
    buffers::Dict{Any,Any}
end

function AdvectionPlan(
    f::DistributionGrid{DT,NX,NV,NXNV,Cart},
    grid::CartGrid,
) where {DT,NX,NV,NXNV}
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
    return AdvectionPlan(kx, kv, vaxes, bslLD.backend(), Dict{Any,Any}())
end

# Buffer `key` of `plan`, created by `make()` on first use.
_buffer!(make, plan::AdvectionPlan, key) =
    lock(() -> get!(make, plan.buffers, key), _plan_cache_lock)

# Complex work buffer and in-place FFT plans along every axis (x axes first, then v).
_spectral_buffers(
    plan::AdvectionPlan,
    f::DistributionGrid{DT,NX,NV,NXNV},
) where {DT,NX,NV,NXNV} = _buffer!(plan, :spectral) do
    ff_buf = similar(f.data, Complex{DT})
    (
        ff_buf,
        ntuple(d -> plan_fft!(ff_buf, d), Val(NXNV)),
        ntuple(d -> plan_ifft!(ff_buf, d), Val(NXNV)),
    )
end

_work_buffer(plan::AdvectionPlan, f) =
    _buffer!(() -> similar(f.data), plan, :work)::typeof(f.data)

"""
    bslLD.release_advection_buffers!(f)

Drop the cached advection plans of distribution `f` (wavenumbers and all buffers), so
that their device memory can be reclaimed once `f` is no longer used. They are rebuilt on the
next advection of `f`.
"""
function release_advection_buffers!(f)
    lock(_plan_cache_lock) do
        delete!(_plan_cache, f.data)
    end
    return nothing
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
its own fully unrolled kernel. `W` must be even and at least 2. Boundaries are
chosen with the `boundary` keyword of `advectX!` (`Periodic()` or `Mirror()`).
"""
struct Lagrange{W}
    function Lagrange{W}() where {W}
        W isa Int && W >= 2 && iseven(W) ||
            throw(ArgumentError("Lagrange stencil width must be an even Int >= 2, got $W"))
        return new{W}()
    end
end
Lagrange(W::Integer) = Lagrange{Int(W)}()
@inline stencil_order(::Lagrange{W}) where {W} = Val(W)

# Departure-point displacement in cells along the advected axis. It is constant
# along that axis -- it depends only on the *other* indices -- so the sweeps below
# compute the weights once per key (see `_shift_key`) instead of once per point.
@inline _shift_cells(ctx::XShiftContext, ixs, ivs) =
    ctx.dt * ctx.vth * _x_displacement(ctx, ivs) / ctx.delta
@inline _shift_cells(ctx::VShiftContext, ixs, ivs) =
    ctx.dt * ctx.electric_scale * _v_displacement(ctx, ixs) / ctx.delta

@inline _line_length(ctx::XShiftContext) = ctx.sizes_x[ctx.dir]
@inline _line_length(ctx::VShiftContext) = ctx.sizes_v[ctx.dir]

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
@inline _line_sizes(ctx::XShiftContext) =
    (Base.setindex(ctx.sizes_x, 1, ctx.dir), ctx.sizes_v)
@inline _line_sizes(ctx::VShiftContext) =
    (ctx.sizes_x, Base.setindex(ctx.sizes_v, 1, ctx.dir))

# --- Boundary conditions --------------------------------------------------

"""
    Periodic()

Wrap the stencil around the advected axis. The default, and the only condition
the spectral path can represent. (Defined in PlasmaCore.)
"""
Periodic

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
`Grid` excludes the upper endpoint `etaMax` of x-axes, so for walls at `0` and
`Lx` with `N` nodes `0:Δx:Lx`, pass `etaMax = Lx + Δx` with `Δx = Lx/(N-1)`
(README, "Grid for a mirror domain"). (Defined in PlasmaCore.)
"""
Mirror

# Velocity components reversed by a mirror on spatial axis `dir`.
@inline _mirror_flip(Bdir::Int, dir::Int, ::Val{NV}) where {NV} =
    ntuple(d -> dir == Bdir ? d == Bdir : d != Bdir, Val(NV))

@inline function _flip_v(ctx, ivs::NTuple{NV,Int}) where {NV}
    return ntuple(d -> ctx.vflip[d] ? ctx.sizes_v[d] + 1 - ivs[d] : ivs[d], Val(NV))
end

# Stencil node `i` of line `L` under `bc`, for the general sweep; `I0 + i*st` is the linear
# index of node `i` for 1 <= i <= n. Periodic wraps. Mirror reads nodes outside 1:n from the
# halo buffer `hb[L, h]` filled by `mirror_halo_kernel!` (node 1-h at h, node n+h at H+h).
@inline _sample_line(::Periodic, src, hb, L, I0, st, i, n) = @inbounds src[I0+mod1(i, n)*st]
@inline function _sample_line(::Mirror, src, hb, L, I0, st, i, n)
    i < 1 && return @inbounds hb[L, 1-i]
    i > n && return @inbounds hb[L, size(hb, 2)÷2+i-n]
    return @inbounds src[I0+i*st]
end

# General sweep, for any axis and boundary: one thread per point (coalesced access, full
# parallelism), cell shift and weights read from the key table (see `_shift_key`), mirror
# halo nodes from `hb` (`nothing` for periodic). Same arithmetic, in the same order, as the
# tiled kernels, so results are bit-identical.
@kernel function lagrange_cached_kernel!(
    dst,
    @Const(src),
    @Const(cells),
    @Const(wts),
    @Const(hb),
    ctx,
    order::Val{W},
    bc,
    st::Int,
    n::Int,
    nxtot::Int,
) where {W}
    p = @index(Global, Linear)
    q = p - 1
    j = (q ÷ st) % n + 1                        # node along the line
    I0 = p - j * st                             # I0 + i*st is node i of this line
    k = _shift_key(ctx, q, nxtot) + 1
    base = j - @inbounds(cells[k]) - W ÷ 2 - 1
    acc = zero(eltype(dst))
    if base >= 0 && base + W <= n
        @inbounds for m = 1:W
            acc += wts[m, k] * src[I0+(base+m)*st]
        end
    else
        L = q % st + st * (q ÷ (st * n)) + 1    # line index, as enumerated by `_line_sizes`
        @inbounds for m = 1:W
            acc += wts[m, k] * _sample_line(bc, src, hb, L, I0, st, base + m, n)
        end
    end
    @inbounds dst[p] = acc
end

# --- Tiled sweeps (periodic): shared-memory staging ------------------------------------
# One workgroup loads a tile of whole lines into shared memory with coalesced reads, stages
# the lines' cell shifts and weights from the key table next to it, and writes the output
# coalesced. Same arithmetic, in the same order, as the general kernel.
const _TILE_WG = 256                        # threads per workgroup
const _TILE_LINES = 32                      # lines per workgroup

# The shift of an x-sweep depends only on the velocity indices, that of a v-sweep only on
# the position indices (see `_shift_cells`). The tiled sweep therefore computes the cell
# shift and weights once per distinct key -- prod(sizes_v) or prod(sizes_x) values instead
# of one per line -- and looks them up by the linear index `q` (0-based) of any node.
@inline _shift_nkeys(ctx::XShiftContext) = prod(ctx.sizes_v)
@inline _shift_nkeys(ctx::VShiftContext) = prod(ctx.sizes_x)
@inline _shift_key(ctx::XShiftContext, q, nxtot) = q ÷ nxtot
@inline _shift_key(ctx::VShiftContext, q, nxtot) = q % nxtot
@inline _key_node(ctx::XShiftContext, k, nxtot) = k * nxtot + 1    # 1-based node of key k (0-based)
@inline _key_node(ctx::VShiftContext, k, nxtot) = k + 1

@kernel function lagrange_key_setup_kernel!(
    cells,
    wts,
    ctx,
    order::Val{W},
    nxtot::Int,
) where {W}
    k = @index(Global, Linear)
    ixs, ivs = index_1d_to_combined(_key_node(ctx, k - 1, nxtot), ctx.sizes_x, ctx.sizes_v)
    c, alpha = shift_split(_shift_cells(ctx, ixs, ivs))
    w = lagrange_weights(order, alpha, eltype(wts))
    @inbounds cells[k] = c
    @inbounds for m = 1:W
        wts[m, k] = w[m]
    end
end

# Copy the cell shifts and weights of the tile's lines from the key table to shared memory;
# `node0(c)` is the 0-based linear index of node 0 of tile line `c` (0-based), or -1 if absent.
@inline function _tile_stage!(
    wsh,
    csh,
    cells,
    wts,
    ctx,
    lid,
    nxtot,
    node0,
    ::Val{W},
) where {W}
    idx = lid
    while idx < _TILE_LINES * W
        c = idx % _TILE_LINES
        m = idx ÷ _TILE_LINES + 1
        q = node0(c)
        if q >= 0
            key = _shift_key(ctx, q, nxtot) + 1
            @inbounds wsh[c+1, m] = wts[m, key]
            m == 1 && (@inbounds csh[c+1] = cells[key])
        end
        idx += _TILE_WG
    end
    return nothing
end

# Stencil sum for node `j` (0-based) of a line held in shared memory; `tile_at(k)` reads node `k`.
@inline function _tile_stencil(
    wsh,
    c,
    cells,
    j,
    order::Val{W},
    ::Val{N},
    tile_at,
    ::Type{T},
) where {W,N,T}
    k0 = mod(j - cells - W ÷ 2, N)                        # node of the stencil start, 0-based
    acc = zero(T)
    @inbounds for m = 1:W
        k = k0 + m
        k = k > N ? k - N : k
        acc += wsh[c, m] * tile_at(k)
    end
    return acc
end

# Every workgroup reads all nodes of its lines into shared memory before the barrier and
# writes only those nodes after it, so the tiled sweep may run in place (`dst === src`).

# Axis with stride 1 (lines contiguous): the tile is `_TILE_LINES` consecutive lines.
# (KernelAbstractions rule: `@index` only as `x = @index(...)`, and nothing computed
# before `@synchronize` is visible after it, so the indices are rebuilt in each phase.)
@kernel function lagrange_tile_contig_kernel!(
    dst,
    src,
    @Const(cells),
    @Const(wts),
    ctx,
    order::Val{W},
    nodes::Val{N},
    nl::Int,
    nxtot::Int,
) where {W,N}
    tile = @localmem eltype(dst) (_TILE_LINES * N)
    wsh = @localmem eltype(dst) (_TILE_LINES, W)
    csh = @localmem Int (_TILE_LINES)
    I1 = @index(Global, Linear)
    let lid = (I1 - 1) % _TILE_WG, L0 = (I1 - 1) ÷ _TILE_WG * _TILE_LINES
        _tile_stage!(
            wsh,
            csh,
            cells,
            wts,
            ctx,
            lid,
            nxtot,
            c -> L0 + c < nl ? (L0 + c) * N : -1,
            order,
        )
        e = lid
        while e < _TILE_LINES * N
            if L0 + e ÷ N < nl
                @inbounds tile[e+1] = src[L0*N+e+1]
            end
            e += _TILE_WG
        end
    end
    @synchronize
    I2 = @index(Global, Linear)
    let lid = (I2 - 1) % _TILE_WG, L0 = (I2 - 1) ÷ _TILE_WG * _TILE_LINES
        e = lid
        while e < _TILE_LINES * N
            l = e ÷ N
            if L0 + l < nl
                j = e - l * N
                acc = _tile_stencil(
                    wsh,
                    l + 1,
                    @inbounds(csh[l+1]),
                    j,
                    order,
                    nodes,
                    k -> @inbounds(tile[l*N+k]),
                    eltype(dst),
                )
                @inbounds dst[L0*N+e+1] = acc
            end
            e += _TILE_WG
        end
    end
end

# Axis with stride `A > 1`: the tile is `_TILE_LINES` neighbouring lines (consecutive in
# memory at every node of the line), all `N` nodes; rows are walked by the thread rows.
@kernel function lagrange_tile_strided_kernel!(
    dst,
    src,
    @Const(cells),
    @Const(wts),
    ctx,
    order::Val{W},
    nodes::Val{N},
    A::Int,
    tiles_per_b::Int,
    nxtot::Int,
) where {W,N}
    tile = @localmem eltype(dst) (_TILE_LINES, N)
    wsh = @localmem eltype(dst) (_TILE_LINES, W)
    csh = @localmem Int (_TILE_LINES)
    I1 = @index(Global, Linear)
    let lid = (I1 - 1) % _TILE_WG, g = (I1 - 1) ÷ _TILE_WG
        b = g ÷ tiles_per_b
        a0 = (g - b * tiles_per_b) * _TILE_LINES           # first line of the tile in the fast block
        _tile_stage!(
            wsh,
            csh,
            cells,
            wts,
            ctx,
            lid,
            nxtot,
            c -> a0 + c < A ? a0 + c + A * N * b : -1,
            order,
        )
        ta = lid % _TILE_LINES
        a = a0 + ta
        if a < A
            p0 = a + A * N * b + 1                       # linear index of node 0 of the line
            j = lid ÷ _TILE_LINES
            while j < N
                @inbounds tile[ta+1, j+1] = src[p0+j*A]
                j += _TILE_WG ÷ _TILE_LINES
            end
        end
    end
    @synchronize
    I2 = @index(Global, Linear)
    let lid = (I2 - 1) % _TILE_WG, g = (I2 - 1) ÷ _TILE_WG
        ta = lid % _TILE_LINES
        b = g ÷ tiles_per_b
        a = (g - b * tiles_per_b) * _TILE_LINES + ta
        if a < A
            p0 = a + A * N * b + 1
            j = lid ÷ _TILE_LINES
            while j < N
                acc = _tile_stencil(
                    wsh,
                    ta + 1,
                    @inbounds(csh[ta+1]),
                    j,
                    order,
                    nodes,
                    k -> @inbounds(tile[ta+1, k]),
                    eltype(dst),
                )
                @inbounds dst[p0+j*A] = acc
                j += _TILE_WG ÷ _TILE_LINES
            end
        end
    end
end

# --- Mirror: reflected halos; tiles on a contiguous spatial axis -----------------------
# A mirror halo node of a line is a node of a *different* line (velocity components
# reversed), so the periodic wrap does not apply, and an in-place sweep must read the
# partner line before any workgroup writes. Before every mirror sweep, a halo kernel
# therefore symmetrises the wall nodes and copies, for every line and any stride, the `H`
# reflected nodes on each side into a small buffer `hb[nl, 2H]`. The tile kernel then holds
# `H + N + H` nodes per line and needs no wrap and no per-tap boundary test; the general
# kernel reads its out-of-range taps from `hb` instead of locating the partner line per tap
# (which made it 9-19x slower than periodic on GPU). `H = W/2 + ceil(smax)` halo nodes per
# side cover every stencil for a cell shift of at most `smax` (a bound from the host
# velocity grid and dt). Same arithmetic, in the same order, as the other kernels, so
# results are bit-identical.

# Upper bound of the cell shift of an x-sweep, from the extent of the velocity grid.
_mirror_shift_bound(ctx::XShiftContext) =
    abs(ctx.dt) * ctx.vth * sqrt(sum(ax -> maximum(abs, ax)^2, ctx.grid.vaxes)) / ctx.delta
_mirror_halo(ctx, ::Val{W}) where {W} =
    W ÷ 2 + max(1, ceil(Int, _mirror_shift_bound(ctx) + 1e-9))

# One thread per pair of mirror-partner lines: line `L` and its velocity-flipped partner `P`,
# handled by the thread with `L <= P`. A line's halo reads only its partner, so no other
# thread touches either line. The thread first symmetrises the wall nodes of both lines,
# f(x_b, v) <- (f(x_b, v) + f(x_b, R v)) / 2, in place, then copies the reflected nodes of
# each line into the other's halo: left node 1-h is node min(1+h, N), right node N+h is
# node max(N-h, 1) of the partner.
@kernel function mirror_halo_kernel!(hb, src, ctx, ::Val{H}, ::Val{N}, st::Int) where {H,N}
    L = @index(Global, Linear)
    lsx, lsv = _line_sizes(ctx)
    ixs, ivs = index_1d_to_combined(L, lsx, lsv)
    fv = _flip_v(ctx, ivs)
    P = index_combined_to_1d(ixs, fv, lsx, lsv)
    if L <= P
        A0 = _line_linear_index(ctx, ixs, ivs, 1) - st    # node j of line L at A0 + j*st
        B0 = _line_linear_index(ctx, ixs, fv, 1) - st     # node j of line P at B0 + j*st
        @inbounds for i in (1, N)
            mean = (src[A0+i*st] + src[B0+i*st]) / 2
            src[A0+i*st] = mean
            src[B0+i*st] = mean
        end
        @inbounds for h = 1:H
            hb[L, h] = src[B0+min(1+h, N)*st]
            hb[L, H+h] = src[B0+max(N-h, 1)*st]
            hb[P, h] = src[A0+min(1+h, N)*st]
            hb[P, H+h] = src[A0+max(N-h, 1)*st]
        end
    end
end

# Symmetrise the wall nodes of `src` in place and fill the reflected halo of every line, for a
# mirror sweep (tiled or general).
function _mirror_halo!(plan, src, ctx, order::Val{W}) where {W}
    n = _line_length(ctx)
    nl = length(src) ÷ n
    H = _mirror_halo(ctx, order)
    hb = _buffer!(plan, (:halo, H, nl)) do
        KernelAbstractions.allocate(plan.backend, eltype(src), nl, 2H)
    end
    mirror_halo_kernel!(plan.backend)(
        hb,
        src,
        ctx,
        Val(H),
        Val(n),
        _line_stride(ctx);
        ndrange = nl,
    )
    return hb
end

@kernel function lagrange_tile_mirror_kernel!(
    dst,
    src,
    @Const(hb),
    @Const(cells),
    @Const(wts),
    ctx,
    order::Val{W},
    nodes::Val{N},
    halo::Val{H0},
    nl::Int,
    nxtot::Int,
) where {W,N,H0}
    tile = @localmem eltype(dst) (_TILE_LINES * (N + 2 * H0))
    wsh = @localmem eltype(dst) (_TILE_LINES, W)
    csh = @localmem Int (_TILE_LINES)
    I1 = @index(Global, Linear)
    let lid = (I1 - 1) % _TILE_WG,
        L0 = (I1 - 1) ÷ _TILE_WG * _TILE_LINES,
        H = H0,
        NT = N + 2 * H0

        _tile_stage!(
            wsh,
            csh,
            cells,
            wts,
            ctx,
            lid,
            nxtot,
            c -> L0 + c < nl ? (L0 + c) * N : -1,
            order,
        )
        e = lid
        while e < _TILE_LINES * NT
            l = e ÷ NT
            if L0 + l < nl
                p = e - l * NT                       # 0-based tile position; node p - H + 1
                if p < H
                    @inbounds tile[e+1] = hb[L0+l+1, H-p]          # node 1-h, h = H-p
                elseif p < H + N
                    @inbounds tile[e+1] = src[(L0+l)*N+(p-H)+1]
                else
                    @inbounds tile[e+1] = hb[L0+l+1, H+(p-H-N)+1]  # node N+h, h = p-H-N+1
                end
            end
            e += _TILE_WG
        end
    end
    @synchronize
    I2 = @index(Global, Linear)
    let lid = (I2 - 1) % _TILE_WG,
        L0 = (I2 - 1) ÷ _TILE_WG * _TILE_LINES,
        H = H0,
        NT = N + 2 * H0                      # (locals do not survive the barrier)

        e = lid
        while e < _TILE_LINES * N
            l = e ÷ N
            if L0 + l < nl
                j = e - l * N                        # 0-based node
                c = @inbounds csh[l+1]
                acc = zero(eltype(dst))
                @inbounds for m = 1:W
                    acc += wsh[l+1, m] * tile[l*NT+H+(j-c-W÷2+m)]
                end
                @inbounds dst[L0*N+e+1] = acc
            end
            e += _TILE_WG
        end
    end
end

# --- Sweep selection -------------------------------------------------------

# Static shared memory per workgroup (CUDA's limit without opt-in).
const _SHMEM_BYTES = 48 * 1024

# Whether a tile of `nodes` values per line, plus the W weights and the cell shift of each
# line, fits in shared memory.
_tile_fits(::Type{T}, nodes, W) where {T} =
    _TILE_LINES * ((nodes + W) * sizeof(T) + sizeof(Int)) <= _SHMEM_BYTES

# Whether the sweep of `ctx` runs tiled (in place): periodic axes, or mirror x-sweeps along
# the contiguous axis, if the tile fits.
function _tiled(ctx, order::Val{W}, bc, ::Type{T}) where {W,T}
    n = _line_length(ctx)
    W <= n || return false
    bc isa Periodic && return _tile_fits(T, n, W)
    return bc isa Mirror &&
           ctx isa XShiftContext &&
           _line_stride(ctx) == 1 &&
           _tile_fits(T, n + 2 * _mirror_halo(ctx, order), W)
end

# Shift tables: integer cell shift and W weights per key (see `_shift_key`).
_shift_tables(plan, src, ctx, ::Val{W}) where {W} =
    _buffer!(plan, (:shifts, W, _shift_nkeys(ctx))) do
        nk = _shift_nkeys(ctx)
        (
            KernelAbstractions.allocate(plan.backend, Int, nk),
            KernelAbstractions.allocate(plan.backend, eltype(src), W, nk),
        )
    end

# One Lagrange sweep of `src` into `dst`; `dst === src` is allowed only if `tiled`.
function _lagrange_sweep!(dst, src, ctx, order::Val{W}, bc, plan, tiled::Bool) where {W}
    exec = plan.backend
    n = _line_length(ctx)
    nl = length(src) ÷ n
    st = _line_stride(ctx)
    nxtot = prod(ctx.sizes_x)
    nkeys = _shift_nkeys(ctx)
    cells, wts = _shift_tables(plan, src, ctx, order)
    lagrange_key_setup_kernel!(exec)(cells, wts, ctx, order, nxtot; ndrange = nkeys)
    hb = bc isa Mirror ? _mirror_halo!(plan, src, ctx, order) : nothing
    if !tiled
        lagrange_cached_kernel!(exec)(
            dst,
            src,
            cells,
            wts,
            hb,
            ctx,
            order,
            bc,
            st,
            n,
            nxtot;
            ndrange = length(src),
        )
    elseif bc isa Mirror
        H = size(hb, 2) ÷ 2
        lagrange_tile_mirror_kernel!(exec, _TILE_WG)(
            dst,
            src,
            hb,
            cells,
            wts,
            ctx,
            order,
            Val(n),
            Val(H),
            nl,
            nxtot;
            ndrange = cld(nl, _TILE_LINES) * _TILE_WG,
        )
    elseif st == 1
        lagrange_tile_contig_kernel!(exec, _TILE_WG)(
            dst,
            src,
            cells,
            wts,
            ctx,
            order,
            Val(n),
            nl,
            nxtot;
            ndrange = cld(nl, _TILE_LINES) * _TILE_WG,
        )
    else
        tpb = cld(st, _TILE_LINES)
        lagrange_tile_strided_kernel!(exec, _TILE_WG)(
            dst,
            src,
            cells,
            wts,
            ctx,
            order,
            Val(n),
            st,
            tpb,
            nxtot;
            ndrange = tpb * (nl ÷ st) * _TILE_WG,
        )
    end
    # No synchronize: every kernel and copy of the advection runs on the same stream, so
    # they are ordered already. A synchronize per sweep cost several ms of host wake-up
    # latency (CUDA.jl's default synchronize is non-blocking) on top of a ~6 ms sweep.
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

# --- Advection over a set of directions ----------------------------------

# `contexts` and `axes` (array dimension of each context) run in sequence.
function _advect!(f, plan, ::Fourier, ::Periodic, contexts, axes)
    exec = plan.backend
    ff_buf, fwd, inv = _spectral_buffers(plan, f)
    for (ctx, axis) in zip(contexts, axes)
        _apply_phase_shift!(
            f,
            ff_buf,
            fwd[axis],
            inv[axis],
            spectral_multiply_kernel!(exec),
            ctx,
            exec,
        )
    end
    return nothing
end

_advect!(f, plan, ::Fourier, bc, contexts, axes) = throw(
    ArgumentError("$(typeof(bc)) boundaries need a Lagrange method; Fourier is periodic"),
)

# Tiled sweeps run in place; the others ping-pong between `f.data` and the plan's work
# buffer, so at most one copy back is made at the end. `tiled = false` forces the general
# kernel (used by the tests to check that both give bit-identical results).
function _advect!(f, plan, method::Lagrange, bc, contexts, _axes; tiled::Bool = true)
    order = stencil_order(method)
    src = f.data
    for ctx in contexts
        if tiled && _tiled(ctx, order, bc, eltype(src))
            _lagrange_sweep!(src, src, ctx, order, bc, plan, true)
        else
            dst = src === f.data ? _work_buffer(plan, f) : f.data
            _lagrange_sweep!(dst, src, ctx, order, bc, plan, false)
            src = dst
        end
    end
    src === f.data || copyto!(f.data, src)
    return nothing
end

# A mirror pairs v with -v by index, which is only the physical reflection when
# the velocity axis is symmetric about zero.
function _check_mirror_axes(grid::CartGrid, dir::Int, ::Val{W}) where {W}
    n = length(grid.xaxes[dir])
    W ÷ 2 < n || throw(
        ArgumentError(
            "Lagrange half-stencil $(W ÷ 2) does not fit in axis $dir of length $n",
        ),
    )
    for (d, ax) in enumerate(grid.vaxes)
        isapprox(first(ax) + last(ax), 0, atol = 1e-12 * max(abs(first(ax)), 1)) || throw(
            ArgumentError(
                "mirror boundaries need velocity axis $d symmetric about zero, " *
                "got [$(first(ax)), $(last(ax))]",
            ),
        )
    end
    return nothing
end

function _advect_x!(
    sp::Species,
    grid::CartGrid,
    simTime::SimulationTime,
    dirs,
    method,
    boundary,
)
    NX = length(grid.xaxes)
    all(in(1:NX), dirs) || throw(ArgumentError("advectX! direction $dirs out of 1:$NX"))
    if boundary isa Mirror && method isa Lagrange
        foreach(dir -> _check_mirror_axes(grid, dir, stencil_order(method)), dirs)
    end
    plan = _get_plan(sp.dist, grid)
    contexts = (_x_context(sp, grid, simTime, dir, plan) for dir in dirs)
    _advect!(sp.dist, plan, method, boundary, contexts, dirs)
    return nothing
end

function _advect_v!(
    sp::Species,
    grid::CartGrid,
    simTime::SimulationTime,
    e::VectorField,
    dirs,
    method,
    boundary,
)
    NV = length(grid.vaxes)
    all(in(1:NV), dirs) || throw(ArgumentError("advectV! direction $dirs out of 1:$NV"))
    boundary isa Periodic ||
        throw(ArgumentError("mirror boundaries apply to spatial axes only, as in bsl6d"))
    plan = _get_plan(sp.dist, grid)
    contexts = (_v_context(sp, grid, simTime, e, dir, plan) for dir in dirs)
    _advect!(sp.dist, plan, method, boundary, contexts, length(grid.xaxes) .+ dirs)
    return nothing
end

advectX!(
    sp::Species,
    grid::CartGrid,
    simTime::SimulationTime;
    method = Fourier(),
    boundary = Periodic(),
) = _advect_x!(sp, grid, simTime, 1:length(grid.xaxes), method, boundary)

advectX!(
    sp::Species,
    grid::CartGrid,
    simTime::SimulationTime,
    dir::Int;
    method = Fourier(),
    boundary = Periodic(),
) = _advect_x!(sp, grid, simTime, dir:dir, method, boundary)

advectV!(
    sp::Species,
    grid::CartGrid,
    simTime::SimulationTime,
    e::VectorField;
    method = Fourier(),
    boundary = Periodic(),
) = _advect_v!(sp, grid, simTime, e, 1:length(grid.vaxes), method, boundary)

advectV!(
    sp::Species,
    grid::CartGrid,
    simTime::SimulationTime,
    e::VectorField,
    dir::Int;
    method = Fourier(),
    boundary = Periodic(),
) = _advect_v!(sp, grid, simTime, e, dir:dir, method, boundary)
