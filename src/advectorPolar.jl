
function advect1DFourier!(data::AbstractArray{Float64,1}, shift::Float64, grid::Grid)
    sshift = 2pi * shift .* fftfreq(size(data)[1])
    data .= real(ifft(fft(data) .* exp.(-sshift .* im)))
end




function advectX!(f::DistributionGrid1d2v{DT,Polar}, grid::PolarGrid, advector=advect1DFourier!) where DT
    dt = grid.dt
    vth = thermal_velocity(f)
    for iv1 = 2 :size(f.data)[2]
        for iv2 = 1:size(f.data)[3]
            xdisp = vth * grid.vaxes[1][iv1] * cos(grid.vaxes[2][iv2] + grid.b0 * grid.time[grid.index[1]])
            fshift =  dt * xdisp/ grid.delta[1]
            advector(view(f.data, :,iv1, iv2), fshift, grid)
        end
    end
end




@inline function displace_polar_velocity!(
    vrp::Base.RefValue{<:Real},
    phip::Base.RefValue{<:Real},
    vr::Real,
    phi::Real,
    delta_vx::Real,
    delta_vy::Real
)
    vx = vr * cos(phi)
    vy = vr * sin(phi)

    vx_prime = vx + delta_vx
    vy_prime = vy + delta_vy

    vrp[] = sqrt(vx_prime^2 + vy_prime^2)
    phip[] = atan(vy_prime, vx_prime)

    return nothing
end

function advectV!(f::DistributionGrid1d2v{DT,Polar}, grid::PolarGrid, e::VectorField{DT,1}) where DT
    F = bslLD.fft(f.data[:, :, :], [3]) / length(grid.vaxes[2])
    rgrid = grid.xaxes[1]
    vgrid = grid.vaxes[1]
    phigrid = grid.vaxes[2]
    nx = length(rgrid)
    nv = length(vgrid)
    nphi = length(phigrid)
    imax = min(round(Int, (nphi / 2 - 1)), fld(nphi, 2))
    dt = grid.dt
    electric_scale = electric_acceleration_scale(f)

    @threads for ix in 1:nx
        delta_vx = dt * electric_scale * e[1].data[ix]
        delta_vy = dt * electric_scale * e[2].data[ix]
        
        # Transformation for advection in the rotating frame
        delta_mux = delta_vx * cos(grid.b0 * grid.time[grid.index[1]]) + delta_vy * sin(grid.b0 * grid.time[grid.index[1]])
        delta_muy = -delta_vx * sin(grid.b0 * grid.time[grid.index[1]]) + delta_vy * cos(grid.b0 * grid.time[grid.index[1]])

        # 1. Build Splines
        itp_re = Vector{Dierckx.Spline1D}(undef, imax + 1)
        itp_im = Vector{Dierckx.Spline1D}(undef, imax + 1)
        @inbounds for m in 1:imax + 1
            vals_re = real.(F[ix, :, m])
            vals_im = imag.(F[ix, :, m])
            itp_re[m] = Spline1D(vgrid, vals_re, k=3)
            itp_im[m] = Spline1D(vgrid, vals_im, k=3)
        end

        # 2. Pre-calculate displaced polar velocity coordinates for the slice
        vr_prime_slice = Array{Float64}(undef, nv, nphi)
        phi_prime_slice = Array{Float64}(undef, nv, nphi)
        
        # Use RefValue for output of inplace function
        vrp_ref = Ref(0.0)
        phip_ref = Ref(0.0)
        
        @inbounds for iv in 1:nv
            vr_grid = vgrid[iv]
            @inbounds for ip in 1:nphi
                phi_grid = phigrid[ip]
                displace_polar_velocity!(vrp_ref, phip_ref, vr_grid, phi_grid, delta_mux, delta_muy)
                vr_prime_slice[iv, ip] = vrp_ref[]
                phi_prime_slice[iv, ip] = phip_ref[]
            end
        end

        # 3. Reconstruction (optimized)
        FvalsRe_at_vr_prime = MVector{imax + 1, Float64}(undef)
        FvalsIm_at_vr_prime = MVector{imax + 1, Float64}(undef)
        
        @inbounds for iv in 1:nv
            @inbounds for ip in 1:nphi
                vr_prime = vr_prime_slice[iv, ip]
                phi_prime = phi_prime_slice[iv, ip]

                @simd for m in 1:imax + 1
                    FvalsRe_at_vr_prime[m] = itp_re[m](vr_prime)
                    FvalsIm_at_vr_prime[m] = itp_im[m](vr_prime)
                end

                # Perform reconstruction sum (m=0 is the first element)
                r_val = FvalsRe_at_vr_prime[1]
                @inbounds for m in 1:imax
                    re = FvalsRe_at_vr_prime[m + 1]
                    im = FvalsIm_at_vr_prime[m + 1]
                    m_phi_prime = m * phi_prime
                    r_val += 2.0 * (re * cos(m_phi_prime) - im * sin(m_phi_prime))
                end
                f.data[ix, iv, ip] = r_val
            end
        end
    end
end
