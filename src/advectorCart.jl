using StaticArrays

function R(id::Int, φ::Real)
    c, s = cos(φ), sin(φ)
    if id == 1
        return SMatrix{2,2}(1.0, 0.0, 0.0, c)
    elseif id == 2
        return SMatrix{2,2}(c, 0.0, 0.0, 1.0)
    else  # id == 3
        return SMatrix{2,2}(c, s, -s, c)
    end
end

function R(φ::Real)
    R(3, φ)
end


# Convert N-dimensional indices (1-based, column-major) to 1D index
function index_nd_to_1d(indices::NTuple{N, Int}, sizes::NTuple{N, Int}) where N
    i0 = 0
    stride = 1
    for k in 1:N
        i0 += (indices[k] - 1) * stride
        stride *= sizes[k]
    end
    return i0 + 1
end

# Convert 1D index (1-based) to N-dimensional indices (column-major)
function index_1d_to_nd(i::Int, sizes::NTuple{N, Int}) where N
    i0 = i - 1  # 0-based
    indices = zeros(Int, N)
    for k in 1:N
        indices[k] = i0 % sizes[k] + 1
        i0 = i0 ÷ sizes[k]
    end
    return Tuple(indices)
end

# Convert combined indices (X + Y) to 1D index
function index_combined_to_1d(
    indicesX::NTuple{M, Int},
    indicesY::NTuple{N, Int},
    sizesX::NTuple{M, Int},
    sizesY::NTuple{N, Int}
) where {M, N}
    iX = index_nd_to_1d(indicesX, sizesX)
    strideX = prod(sizesX)
    iY = index_nd_to_1d(indicesY, sizesY)
    return iX + (iY - 1) * strideX
end

# Convert 1D index back to (X, Y) indices
function index_1d_to_combined(
    i::Int,
    sizesX::NTuple{M, Int},
    sizesY::NTuple{N, Int}
) where {M, N}
    strideX = prod(sizesX)
    iX = (i - 1) % strideX + 1
    iY = (i - 1) ÷ strideX + 1
    return (index_1d_to_nd(iX, sizesX), index_1d_to_nd(iY, sizesY))
end


@inline function compute_x_shift(grid::Grid, index::Int, dir::Int)
    sx = Tuple(map(length, grid.xaxes))
    sv = Tuple(map(length, grid.vaxes))

    (ixs,ivs) = index_1d_to_combined(index,sx,sv)
    
    xdisp = 0.0
    for dv in 1:length(ivs)
            xdisp += grid.vaxes[dv][ivs[dv]] * R(1, grid.b0 * grid.time[grid.index[1]])[dir, dv]
    end

    return grid.dt * xdisp
end



@kernel function ka_advect_x_1d1v_phase!(ff, kx, vaxis, dt, dx)
    ix, iv = @index(Global, NTuple)
    if ix <= size(ff, 1) && iv <= size(ff, 2)
        phase = -(dt / dx) * kx[ix] * vaxis[iv]
        @inbounds ff[ix, iv] *= cis(phase)
    end
end

@kernel function ka_advect_v_1d1v_phase!(ff, ex, kv, dt, dv)
    ix, ik = @index(Global, NTuple)
    if ix <= size(ff, 1) && ik <= size(ff, 2)
        phase = -(dt / dv) * ex[ix] * kv[ik]
        @inbounds ff[ix, ik] *= cis(phase)
    end
end

function advectX!(f::DistributionGrid{Float64,1,1,2}, grid::Grid)
    ff = fft(f.data, 1)
    kx = similar(ff, Float64, size(f.data, 1))
    copyto!(kx, collect(2pi .* fftfreq(size(f.data, 1))))
    vaxis = similar(ff, Float64, length(grid.vaxes[1]))
    copyto!(vaxis, grid.vaxes[1])
    exec = bslLD.backend()
    kernel! = ka_advect_x_1d1v_phase!(exec)
    kernel!(ff, kx, vaxis, grid.dt, grid.delta[1]; ndrange=size(ff))
    KernelAbstractions.synchronize(exec)
    f.data .= real(ifft(ff, 1))
    return nothing
end

function advectV!(f::DistributionGrid{Float64,1,1,2}, grid::Grid, e::VectorField)
    ff = fft(f.data, 2)
    kv = similar(ff, Float64, size(f.data, 2))
    copyto!(kv, collect(2pi .* fftfreq(size(f.data, 2))))
    exec = bslLD.backend()
    kernel! = ka_advect_v_1d1v_phase!(exec)
    kernel!(ff, e[1].data, kv, grid.dt, grid.delta[2]; ndrange=size(ff))
    KernelAbstractions.synchronize(exec)
    f.data .= real(ifft(ff, 2))
    return nothing
end


@kernel function ka_advect_x_1d2v_phase!(ff, kx, vaxis1, vaxis2, phi, dt, dx)
    ix, iv1, iv2 = @index(Global, NTuple)
    if ix <= size(ff, 1) && iv1 <= size(ff, 2) && iv2 <= size(ff, 3)
        # Rotation matrix R(phi)[1,1]*v1 + R(phi)[1,2]*v2
        xdisp = cos(phi) * vaxis1[iv1] + (-sin(phi)) * vaxis2[iv2]
        phase = -(dt / dx) * kx[ix] * xdisp
        @inbounds ff[ix, iv1, iv2] *= cis(phase)
    end
end

@kernel function ka_advect_v1_1d2v_phase!(ff, ex1_rotated, kv1, dt, dv1)
    ix, ik, iv2 = @index(Global, NTuple)
    if ix <= size(ff, 1) && ik <= size(ff, 2) && iv2 <= size(ff, 3)
        phase = -(dt / dv1) * ex1_rotated[ix] * kv1[ik]
        @inbounds ff[ix, ik, iv2] *= cis(phase)
    end
end

@kernel function ka_advect_v2_1d2v_phase!(ff, ex2_rotated, kv2, dt, dv2)
    ix, iv1, ik = @index(Global, NTuple)
    if ix <= size(ff, 1) && iv1 <= size(ff, 2) && ik <= size(ff, 3)
        phase = -(dt / dv2) * ex2_rotated[ix] * kv2[ik]
        @inbounds ff[ix, iv1, ik] *= cis(phase)
    end
end

function advectX!(f::DistributionGrid{Float64,1,2,3}, grid::Grid) where DT
    # FFT along x dimension (dim 1)
    ff = fft(f.data, 1)

    # Build kx frequencies on the appropriate backend array
    kx = similar(ff, Float64, size(f.data, 1))
    copyto!(kx, collect(2pi .* fftfreq(size(f.data, 1))))

    # Build velocity axes on the appropriate backend array
    vaxis1 = similar(ff, Float64, length(grid.vaxes[1]))
    copyto!(vaxis1, grid.vaxes[1])

    vaxis2 = similar(ff, Float64, length(grid.vaxes[2]))
    copyto!(vaxis2, grid.vaxes[2])

    # Current rotation angle
    phi = grid.b0 * grid.time[grid.index[1]]

    exec = bslLD.backend()
    kernel! = ka_advect_x_1d2v_phase!(exec)
    kernel!(ff, kx, vaxis1, vaxis2, phi, grid.dt, grid.delta[1]; ndrange=size(ff))
    KernelAbstractions.synchronize(exec)

    f.data .= real(ifft(ff, 1))
    return nothing
end

function advectV!(f::DistributionGrid{Float64,1,2,3}, grid::Grid, e::VectorField) where DT
    dt  = grid.dt
    phi = -grid.b0 * grid.time[grid.index[1]]
    Rphi = R(phi)   # 2×2 rotation matrix

    exec = bslLD.backend()

    # --- Step 1: advect along v1 dimension (dim 2) ---
    # FFT along v1
    ff = fft(f.data, 2)

    # Rotated E-field component along v1: ex1_rot[ix] = R(-phi)[1,1]*e1[ix] + R(-phi)[1,2]*e2[ix]
    ex1_rotated = similar(ff, Float64, size(f.data, 1))
    copyto!(ex1_rotated,
        Rphi[1,1] .* e[1].data)

    kv1 = similar(ff, Float64, size(f.data, 2))
    copyto!(kv1, collect(2pi .* fftfreq(size(f.data, 2))))

    kernel! = ka_advect_v1_1d2v_phase!(exec)
    kernel!(ff, ex1_rotated, kv1, dt, grid.delta[2]; ndrange=size(ff))
    KernelAbstractions.synchronize(exec)

    f.data .= real(ifft(ff, 2))

    # --- Step 2: advect along v2 dimension (dim 3) ---
    # FFT along v2
    ff = fft(f.data, 3)

    # Rotated E-field component along v2: ex2_rot[ix] = R(-phi)[2,1]*e1[ix] + R(-phi)[2,2]*e2[ix]
    ex2_rotated = similar(ff, Float64, size(f.data, 1))
    copyto!(ex2_rotated,
        Rphi[2,1] .* e[1].data )

    kv2 = similar(ff, Float64, size(f.data, 3))
    copyto!(kv2, collect(2pi .* fftfreq(size(f.data, 3))))

    kernel! = ka_advect_v2_1d2v_phase!(exec)
    kernel!(ff, ex2_rotated, kv2, dt, grid.delta[3]; ndrange=size(ff))
    KernelAbstractions.synchronize(exec)

    f.data .= real(ifft(ff, 3))
    return nothing
end
