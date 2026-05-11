function R(omega::Float64)
    return [[cos(omega) ,sin(omega)] [-sin(omega), cos(omega)]]
end


function advect1DFourier!(data::AbstractArray{Float64,1}, shift::Float64, grid::Grid)
    sshift = 2pi * shift .* fftfreq(size(data)[1])
    data .= real(ifft(fft(data) .* exp.(-sshift .* im)))
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