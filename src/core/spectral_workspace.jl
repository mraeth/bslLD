struct SpectralWorkspace{FB,FP,IP,KT,BK}
    fft_buf::FB
    fwd_plans::FP
    inv_plans::IP
    k::KT
    backend::BK
end

function SpectralWorkspace(arr::AbstractArray, grid::Grid)
    ndims_x = spatial_ndims(grid)
    fft_buf = similar(arr, Complex{Float64})
    fwd_plans = ntuple(d -> plan_fft!(fft_buf, d), ndims_x)
    inv_plans = ntuple(d -> plan_ifft!(fft_buf, d), ndims_x)
    k = ntuple(d -> spectral_wavenumbers(arr, grid, d), ndims_x)
    return SpectralWorkspace(fft_buf, fwd_plans, inv_plans, k, bslLD.backend())
end

# Unified cache for all solver workspaces.
# Key: (solver_tag::Symbol, size, eltype, ndims_x, ncomp, bdir, delta_tuple)
const _solver_workspace_cache = Dict{Any,Any}()
const _solver_workspace_cache_lock = ReentrantLock()

# Shared cache for SpectralWorkspace objects keyed by grid shape.
# Key: (size, eltype, ndims_x, delta_tuple)
const _spectral_ws_cache = Dict{Any,Any}()
const _spectral_ws_cache_lock = ReentrantLock()

function _spectral_ws_key(arr::AbstractArray, grid::Grid)
    ndims_x = spatial_ndims(grid)
    return (size(arr), eltype(arr), ndims_x, Tuple(grid.delta[1:ndims_x]))
end

function _differentiate_impl!(
    out::AbstractArray,
    field::ScalarField,
    ws::SpectralWorkspace,
    dir::Int,
    negate::Bool = false,
)
    _differentiate_impl!(
        out, field, ws.fft_buf, ws.fwd_plans[dir], ws.inv_plans[dir],
        ws.k[dir], dir, ws.backend, negate,
    )
end

function _get_spectral_workspace(arr::AbstractArray, grid::Grid)
    key = _spectral_ws_key(arr, grid)
    lock(_spectral_ws_cache_lock) do
        get!(() -> SpectralWorkspace(arr, grid), _spectral_ws_cache, key)
    end
end

# Copy src → fft_buf, apply all fwd_plans in-place, copy result → dst.
function _fwd_fft_to!(sw::SpectralWorkspace, src::AbstractArray, dst::AbstractArray, ndims_x::Int)
    @. sw.fft_buf = src
    for d in 1:ndims_x
        sw.fwd_plans[d] * sw.fft_buf
    end
    dst .= sw.fft_buf
    return dst
end

# Copy src → fft_buf, apply all inv_plans in-place, write real part → dst.
function _inv_fft_from!(sw::SpectralWorkspace, src::AbstractArray, dst::AbstractArray, ndims_x::Int)
    @. sw.fft_buf = src
    for d in 1:ndims_x
        sw.inv_plans[d] * sw.fft_buf
    end
    @. dst = real(sw.fft_buf)
    return dst
end

# In-place real-space divergence using SpectralWorkspace.
# Writes sum_d d/dx_d field[d] into `out`. Uses `temp` as a scratch Float buffer.
function _apply_div!(
    out::AbstractArray,
    field::VectorField,
    temp::AbstractArray,
    ws::SpectralWorkspace,
    grid::Grid,
)
    ndims_x = spatial_ndims(grid)
    _differentiate_impl!(out, field[1], ws, 1, false)
    for d in 2:ndims_x
        _differentiate_impl!(temp, field[d], ws, d, false)
        out .+= temp
    end
    return out
end

# In-place real-space curl using SpectralWorkspace.
# `out` is a NTuple{3} of Float64 field-sized arrays; `temp` is a single scratch Float array.
function _apply_curl!(
    out::NTuple{3},
    field::VectorField,
    temp::AbstractArray,
    ws::SpectralWorkspace,
    grid::Grid,
)
    ndims_x = spatial_ndims(grid)
    ncomp = ncomponents(field)

    if ndims_x == 1
        if ncomp == 2
            _differentiate_impl!(out[1], field[2], ws, 1, true)    # -d/dx f[2]
            _differentiate_impl!(out[2], field[1], ws, 1, false)   # +d/dx f[1]
            fill!(out[3], 0)
        elseif ncomp == 3
            fill!(out[1], 0)
            _differentiate_impl!(out[2], field[3], ws, 1, true)    # -d/dx f[3]
            _differentiate_impl!(out[3], field[2], ws, 1, false)   # +d/dx f[2]
        else
            throw(ArgumentError("1D curl requires 2 or 3 field components"))
        end
    elseif ndims_x == 2
        ncomp == 3 || throw(ArgumentError("2D curl requires 3 field components"))
        _differentiate_impl!(out[1], field[3], ws, 2, false)   # d/dy f[3]
        _differentiate_impl!(out[2], field[3], ws, 1, true)    # -d/dx f[3]
        _differentiate_impl!(out[3], field[2], ws, 1, false)   # d/dx f[2]
        _differentiate_impl!(temp, field[1], ws, 2, false)     # d/dy f[1]
        out[3] .-= temp
    elseif ndims_x == 3
        ncomp == 3 || throw(ArgumentError("3D curl requires 3 field components"))
        _differentiate_impl!(out[1], field[3], ws, 2, false)   # d/dy f[3]
        _differentiate_impl!(temp, field[2], ws, 3, false)     # d/dz f[2]
        out[1] .-= temp
        _differentiate_impl!(out[2], field[1], ws, 3, false)   # d/dz f[1]
        _differentiate_impl!(temp, field[3], ws, 1, false)     # d/dx f[3]
        out[2] .-= temp
        _differentiate_impl!(out[3], field[2], ws, 1, false)   # d/dx f[2]
        _differentiate_impl!(temp, field[1], ws, 2, false)     # d/dy f[1]
        out[3] .-= temp
    else
        throw(ArgumentError("_apply_curl! supports 1D, 2D, and 3D grids"))
    end
    return out
end
