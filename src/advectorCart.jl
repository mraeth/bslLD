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
end

struct VShiftContext{GT, ET, KT, SXT, SVT, PT}
    grid::GT
    e_components::ET
    k::KT
    sizes_x::SXT
    sizes_v::SVT
    dir::Int
    phi::PT
end

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

    return (ctx.grid.dt / ctx.grid.delta[ctx.dir]) * ctx.k[ixs[ctx.dir]] * xdisp
end

@inline function compute_v_phase(ctx::VShiftContext, index::Int)
    ixs, ivs = index_1d_to_combined(index, ctx.sizes_x, ctx.sizes_v)

    delta_v = zero(eltype(ctx.k))
    rotation = R(ctx.grid.Bdir, -ctx.phi)
    for field_dir in 1:length(ctx.e_components)
        delta_v += ctx.e_components[field_dir][ixs[1]] * rotation[ctx.dir, field_dir]
    end

    return (ctx.grid.dt / ctx.grid.delta[length(ctx.sizes_x) + ctx.dir]) * ctx.k[ivs[ctx.dir]] * delta_v
end

@kernel function distribution_kernel!(ff, phase_context)
    i = @index(Global)

    if i <= length(ff)
        phase = phase_context(i)
        @inbounds ff[i] *= cis(-phase)
    end
end

function x_shift_context(grid::CartGrid, k, dir::Int)
    sizes_x = Tuple(length.(grid.xaxes))
    sizes_v = Tuple(length.(grid.vaxes))
    vaxes = Tuple(backend_vector(axis) for axis in grid.vaxes)
    phi = grid.b0 * grid.time[grid.index[1]]
    return XShiftContext(grid, k, vaxes, sizes_x, sizes_v, dir, phi)
end

function v_shift_context(grid::CartGrid, e::VectorField, k, dir::Int)
    sizes_x = Tuple(length.(grid.xaxes))
    sizes_v = Tuple(length.(grid.vaxes))
    e_components = Tuple(component.data for component in e)
    phi = grid.b0 * grid.time[grid.index[1]]
    return VShiftContext(grid, e_components, k, sizes_x, sizes_v, dir, phi)
end

function fourier_wavenumbers(ff, n::Int)
    k = similar(ff, Float64, n)
    copyto!(k, collect(2pi .* fftfreq(n)))
    return k
end

function advect_x_generic!(f::DistributionGrid{Float64,NX,NV,NXNV,Cart}, grid::CartGrid) where {NX, NV, NXNV}
    exec = bslLD.backend()
    kernel! = distribution_kernel!(exec)

    for dir in 1:NX
        fft_dim = dir

        ff = fft(f.data, fft_dim)
        kx = fourier_wavenumbers(ff, size(f.data, fft_dim))

        kernel!(ff, x_shift_context(grid, kx, dir); ndrange=length(ff))
        KernelAbstractions.synchronize(exec)

        f.data .= real(ifft(ff, fft_dim))
    end
    return nothing
end

function advect_v_generic!(f::DistributionGrid{Float64,NX,NV,NXNV,Cart}, grid::CartGrid, e::VectorField) where {NX, NV, NXNV}
    exec = bslLD.backend()
    kernel! = distribution_kernel!(exec)

    for dir in 1:NV
        fft_dim = length(grid.xaxes) + dir
        ff = fft(f.data, fft_dim)
        kv = fourier_wavenumbers(ff, size(f.data, fft_dim))

        kernel!(ff, v_shift_context(grid, e, kv, dir); ndrange=length(ff))
        KernelAbstractions.synchronize(exec)

        f.data .= real(ifft(ff, fft_dim))
    end

    return nothing
end

function advectX!(f::DistributionGrid{Float64,NX,NV,NXNV,Cart}, grid::CartGrid) where {NX, NV, NXNV}
    return advect_x_generic!(f, grid)
end

function advectV!(f::DistributionGrid{Float64,NX,NV,NXNV,Cart}, grid::CartGrid, e::VectorField) where {NX, NV, NXNV}
    return advect_v_generic!(f, grid, e)
end
