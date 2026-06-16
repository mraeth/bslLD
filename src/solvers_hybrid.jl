struct EMSolverDKPol{T<:AbstractFloat} <: AbstractFieldSolver
    beta_i      :: T
    mu          :: T
    initialized :: Base.RefValue{Bool}
end

function EMSolverDKPol(beta_i::Real, mu::Real)
    T = promote_type(typeof(float(beta_i)), typeof(float(mu)))
    return EMSolverDKPol{T}(T(beta_i), T(mu), Ref(false))
end

function _step_dk_pol!(
    E        :: VectorField,
    B        :: VectorField,
    J_i_perp :: VectorField,
    Pi_diff  :: VectorField,
    grid     :: Grid,
    solver   :: EMSolverDKPol,
    dt       :: Real,
)
    ndims_x     = spatial_ndims(grid)
    ncomp_field = ncomponents(E)
    perp_dirs = [d for d in 1:ncomp_field if d != grid.Bdir]

    # Explicit Faraday: B^{n-1/2} -> B^{n+1/2}
    curlE = curl(E, grid)
    for d in 1:ncomponents(B)
        B[d].data .-= dt .* curlE[d].data
    end

    # Implicit E_⊥ update: [I - (dt/μ)R] E_⊥^{n+1} = RHS_⊥
    curlB = curl(B, grid)
    α = dt / solver.mu
    c = 2 / solver.beta_i

    d1, d2 = perp_dirs[1], perp_dirs[2]
    # J_i_perp is velocity-space indexed ([1,2,...]) not field-space indexed ([d1,d2,...])
    rhs1  = E[d1].data .+ α .* (c .* curlB[d1].data .- J_i_perp[1].data)
    rhs2  = E[d2].data .+ α .* (c .* curlB[d2].data .- J_i_perp[2].data)
    denom = 1 + α^2
    # sign of the Levi-Civita symbol ε_{d1,d2,Bdir}: +1 for cyclic, -1 for anti-cyclic
    parity = (d2 % 3 + 1 == grid.Bdir) ? 1 : -1
    E[d1].data .= (rhs1 .+ parity * α .* rhs2) ./ denom
    E[d2].data .= (rhs2 .- parity * α .* rhs1) ./ denom

    # Helmholtz solve for E_z
    div_Pi = div(Pi_diff, grid)

    spatial_perp_dirs = [d for d in perp_dirs if d <= ndims_x]
    if isempty(spatial_perp_dirs)
        div_E_perp = ScalarField(zero.(E[1].data))
    else
        div_E_perp = differentiate(E[spatial_perp_dirs[1]], grid, spatial_perp_dirs[1])
        for d in spatial_perp_dirs[2:end]
            div_E_perp = div_E_perp + differentiate(E[d], grid, d)
        end
    end

    if grid.Bdir <= ndims_x
        dz_div_E_perp = differentiate(div_E_perp, grid, grid.Bdir)
    else
        dz_div_E_perp = ScalarField(zero.(div_E_perp.data))
    end

    rhs_data = (solver.beta_i / 2) .* div_Pi.data .+ dz_div_E_perp.data
    rhs_hat  = fft_spatial(rhs_data, grid)

    kviews = spectral_wavenumber_views(E[1], grid)
    kperp2 = bslLD.backend_array(zeros(Float64, size(E[1].data)))
    for d in spatial_perp_dirs
        kperp2 = kperp2 .+ kviews[d] .^ 2
    end

    helmholtz_op = .-kperp2 .- (solver.beta_i / 2) * (1 + 1 / solver.mu)
    E[grid.Bdir].data .= real(ifft_spatial(rhs_hat ./ helmholtz_op, grid))

    return nothing
end

function solve_fields!(
    sol     :: FieldSolution,
    moments :: Moments,
    grid    :: Grid,
    solver  :: EMSolverDKPol,
    dt      :: Real,
)
    moments.J       !== nothing || throw(ArgumentError("moments.J is required for EMSolverDKPol"))
    moments.Pi_diff !== nothing || throw(ArgumentError("moments.Pi_diff is required for EMSolverDKPol"))
    if !solver.initialized[]
        initialize_staggered_B!(sol, grid, dt)
        solver.initialized[] = true
    end
    copyto!(sol.Enew, sol.E)
    _step_dk_pol!(sol.Enew, sol.B, moments.J, moments.Pi_diff, grid, solver, dt)
    return sol
end


struct EMSolverDKNoPol{T<:AbstractFloat} <: AbstractFieldSolver
    beta_i :: T
    mu     :: T
end

function EMSolverDKNoPol(beta_i::Real, mu::Real)
    T = promote_type(typeof(float(beta_i)), typeof(float(mu)))
    return EMSolverDKNoPol{T}(T(beta_i), T(mu))
end

function _step_dk_no_pol!(
    E        :: VectorField,
    B        :: VectorField,
    J_i_perp :: VectorField,
    Pi_diff  :: VectorField,
    grid     :: Grid,
    solver   :: EMSolverDKNoPol,
    dt       :: Real,
)
    ndims_x     = spatial_ndims(grid)
    ncomp_field = ncomponents(E)
    perp_dirs   = [d for d in 1:ncomp_field if d != grid.Bdir]
    d1, d2, pz  = perp_dirs[1], perp_dirs[2], grid.Bdir

    c = 2 / solver.beta_i
    λ = (solver.beta_i / 2) * (1 + 1 / solver.mu)
    α = c * dt / 2   # CN: half-step coupling

    Bhat   = [fft_spatial(B[d].data, grid) for d in 1:3]
    Ehat_n = [fft_spatial(E[d].data, grid) for d in 1:3]
    J_hat  = [fft_spatial(J_i_perp[d].data, grid) for d in 1:2]
    Pi_hat = [fft_spatial(Pi_diff[d].data, grid) for d in 1:ndims_x]

    kviews = spectral_wavenumber_views(B[1], grid)
    k1 = d1 <= ndims_x ? kviews[d1] : 0.0
    k2 = d2 <= ndims_x ? kviews[d2] : 0.0
    kz = pz <= ndims_x ? kviews[pz] : 0.0

    k2_perp = k1 .^ 2 .+ k2 .^ 2
    k2_tot  = k2_perp .+ kz .^ 2

    curlB_hat  = spectral_curl_hat(Bhat, kviews, ndims_x)
    div_Pi_hat = sum(im .* kviews[d] .* Pi_hat[d] for d in 1:ndims_x)

    # RHS via rotation R(w1, w2) = (w2, −w1)
    w1 = c .* curlB_hat[d1] .- J_hat[1]
    w2 = c .* curlB_hat[d2] .- J_hat[2]
    b1 =  w2
    b2 = .-w1
    b3 = (solver.beta_i / 2) .* div_Pi_hat

    # 3×3 matrix entries (all real, broadcast over all Fourier modes)
    a11 = 1 .- α .* k1 .* k2
    a12 = α .* (k2_tot .- k2 .^ 2)
    a13 = .-α .* k2 .* kz
    a21 = .-α .* (k2_tot .- k1 .^ 2)
    a22 = 1 .+ α .* k1 .* k2
    a23 = α .* k1 .* kz
    a31 = kz .* k1
    a32 = kz .* k2
    a33 = .-k2_perp .- λ   # always < 0 since λ > 0

    # Cramer's rule: first-row cofactors of A
    m11 = a22 .* a33 .- a23 .* a32
    m12 = a21 .* a33 .- a23 .* a31
    m13 = a21 .* a32 .- a22 .* a31
    detA = a11 .* m11 .- a12 .* m12 .+ a13 .* m13

    Ehat_d1 = (b1 .* m11 .- a12 .* (b2 .* a33 .- a23 .* b3) .+ a13 .* (b2 .* a32 .- a22 .* b3)) ./ detA
    Ehat_d2 = (a11 .* (b2 .* a33 .- a23 .* b3) .- b1 .* m12 .+ a13 .* (a21 .* b3 .- b2 .* a31)) ./ detA
    Ehat_pz = (a11 .* (a22 .* b3 .- b2 .* a32) .- a12 .* (a21 .* b3 .- b2 .* a31) .+ b1 .* m13) ./ detA

    # Faraday: B^{n+1} = B^n − dt (ik × E^{n+1/2})
    Ehalf_hat = similar(Bhat)
    Ehalf_hat[d1] = Ehat_d1
    Ehalf_hat[d2] = Ehat_d2
    Ehalf_hat[pz] = Ehat_pz
    curlEhalf_hat = spectral_curl_hat(Ehalf_hat, kviews, ndims_x)
    for d in 1:3
        Bhat[d] .= Bhat[d] .- dt .* curlEhalf_hat[d]
    end

    # Recover E^{n+1} = 2 E^{n+1/2} − E^n
    E[d1].data .= real(ifft_spatial(2 .* Ehat_d1 .- Ehat_n[d1], grid))
    E[d2].data .= real(ifft_spatial(2 .* Ehat_d2 .- Ehat_n[d2], grid))
    E[pz].data .= real(ifft_spatial(2 .* Ehat_pz .- Ehat_n[pz], grid))
    for d in 1:3
        B[d].data .= real(ifft_spatial(Bhat[d], grid))
    end

    return nothing
end

function solve_fields!(
    sol     :: FieldSolution,
    moments :: Moments,
    grid    :: Grid,
    solver  :: EMSolverDKNoPol,
    dt      :: Real,
)
    moments.J       !== nothing || throw(ArgumentError("moments.J is required for EMSolverDKNoPol"))
    moments.Pi_diff !== nothing || throw(ArgumentError("moments.Pi_diff is required for EMSolverDKNoPol"))
    copyto!(sol.Enew, sol.E)
    _step_dk_no_pol!(sol.Enew, sol.B, moments.J, moments.Pi_diff, grid, solver, dt)
    return sol
end


# Initialise B to the staggered B^{-1/2} from B^0 and E^0 before the time loop (EMSolverDKPol only).
function initialize_staggered_B!(sol::FieldSolution, grid::Grid, dt::Real)
    curlE = curl(sol.E, grid)
    for d in 1:ncomponents(sol.B)
        sol.B[d].data .+= (dt / 2) .* curlE[d].data
    end
    return sol
end
