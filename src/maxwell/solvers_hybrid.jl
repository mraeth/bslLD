struct EMSolverDKPol{T<:AbstractFloat} <: AbstractFieldSolver
    beta_i::T
    mu::T
    initialized::Base.RefValue{Bool}
end

function EMSolverDKPol(beta_i::Real, mu::Real)
    T = promote_type(typeof(float(beta_i)), typeof(float(mu)))
    return EMSolverDKPol{T}(T(beta_i), T(mu), Ref(false))
end

# ── EMDKPolWorkspace ─────────────────────────────────────────────────────────

struct EMDKPolWorkspace{SW,FA}
    sw::SW
    curl_buf::NTuple{3,FA}  # holds curlE then curlB (reused)
    temp1::FA               # scratch: rhs1, div_Pi, rhs_data
    temp2::FA               # scratch: rhs2, div_E_perp
    temp3::FA               # scratch: dz_div_E_perp; 2nd term in curl/div
    helmholtz::FA           # pre-computed: -kperp2 - (beta_i/2)*(1+1/mu)
    ndims_x::Int
    d1::Int                 # first perp direction index
    d2::Int                 # second perp direction index
    pz::Int                 # parallel (Bdir) index
    have_z_curl::Bool       # grid.Bdir <= ndims_x
    spatial_perp_dirs::Vector{Int}
end

function _make_em_dk_pol_workspace(
    arr::AbstractArray,
    grid::Grid,
    ncomp::Int,
    solver::EMSolverDKPol,
)
    ndims_x = spatial_ndims(grid)
    ndims_data = ndims(arr)
    perp_dirs = [d for d = 1:ncomp if d != grid.Bdir]
    d1, d2, pz = perp_dirs[1], perp_dirs[2], grid.Bdir
    spatial_perp_dirs = [d for d in perp_dirs if d <= ndims_x]

    sw = _get_spectral_workspace(arr, grid)

    DT = eltype(arr)
    curl_buf = ntuple(_ -> fill!(similar(arr), zero(DT)), 3)
    temp1 = fill!(similar(arr), zero(DT))
    temp2 = fill!(similar(arr), zero(DT))
    temp3 = fill!(similar(arr), zero(DT))

    # Pre-compute Helmholtz operator: -kperp2 - (beta_i/2)*(1+1/mu)
    kperp2 = fill!(similar(arr), zero(DT))
    for d in spatial_perp_dirs
        kv = reshape(sw.k[d], ntuple(i -> i == d ? length(sw.k[d]) : 1, ndims_data))
        kperp2 .+= kv .^ 2
    end
    lambda = DT((solver.beta_i / 2) * (1 + 1 / solver.mu))
    helmholtz = similar(arr)
    @. helmholtz = -kperp2 - lambda

    return EMDKPolWorkspace(
        sw,
        curl_buf,
        temp1,
        temp2,
        temp3,
        helmholtz,
        ndims_x,
        d1,
        d2,
        pz,
        grid.Bdir <= ndims_x,
        spatial_perp_dirs,
    )
end

function _get_em_dk_pol_workspace(sol::FieldSolution, grid::Grid, solver::EMSolverDKPol)
    arr = sol.E[1].data
    ndims_x = spatial_ndims(grid)
    ncomp = ncomponents(sol.E)
    key = (
        :em_dk_pol,
        size(arr),
        eltype(arr),
        ndims_x,
        ncomp,
        grid.Bdir,
        Tuple(grid.delta[1:ndims_x]),
        solver.beta_i,
        solver.mu,
    )
    lock(_solver_workspace_cache_lock) do
        get!(
            () -> _make_em_dk_pol_workspace(arr, grid, ncomp, solver),
            _solver_workspace_cache,
            key,
        )
    end
end

# ── _step_dk_pol! with workspace ─────────────────────────────────────────────

function _step_dk_pol!(
    E::VectorField,
    B::VectorField,
    J_i_perp::VectorField,
    Pi_diff::VectorField,
    grid::Grid,
    solver::EMSolverDKPol,
    dt::Real,
    ws::EMDKPolWorkspace,
)
    d1, d2, pz = ws.d1, ws.d2, ws.pz
    DT = eltype(ws.temp1)
    _dt = DT(dt)

    # Explicit Faraday: B^{n-1/2} -> B^{n+1/2}
    _apply_curl!(ws.curl_buf, E, ws.temp3, ws.sw, grid)
    for d = 1:ncomponents(B)
        B[d].data .-= _dt .* ws.curl_buf[d]
    end

    # Implicit E_⊥ update: [I - (dt/μ)R] E_⊥^{n+1} = RHS_⊥
    _apply_curl!(ws.curl_buf, B, ws.temp3, ws.sw, grid)
    α = DT(dt / solver.mu)
    c = DT(2 / solver.beta_i)
    @. ws.temp1 = E[d1].data + α * (c * ws.curl_buf[d1] - J_i_perp[1].data)  # rhs1
    @. ws.temp2 = E[d2].data + α * (c * ws.curl_buf[d2] - J_i_perp[2].data)  # rhs2
    denom = DT(1) + α^2
    parity = DT((d2 % 3 + 1 == pz) ? 1 : -1)
    @. E[d1].data = (ws.temp1 + parity * α * ws.temp2) / denom
    @. E[d2].data = (ws.temp2 - parity * α * ws.temp1) / denom

    # Helmholtz solve for E_z
    # div_Pi → temp1
    _apply_div!(ws.temp1, Pi_diff, ws.temp3, ws.sw, grid)

    # div_E_perp → temp2
    if isempty(ws.spatial_perp_dirs)
        fill!(ws.temp2, 0)
    else
        _differentiate_impl!(
            ws.temp2,
            E[ws.spatial_perp_dirs[1]],
            ws.sw,
            ws.spatial_perp_dirs[1],
            false,
        )
        for d in ws.spatial_perp_dirs[2:end]
            _differentiate_impl!(ws.temp3, E[d], ws.sw, d, false)
            ws.temp2 .+= ws.temp3
        end
    end

    # dz_div_E_perp → temp3
    if ws.have_z_curl
        _differentiate_impl!(ws.temp3, ScalarField(ws.temp2), ws.sw, pz, false)
    else
        fill!(ws.temp3, 0)
    end

    # rhs_data = (beta_i/2)*div_Pi + dz_div_E_perp → temp1 (in-place)
    _half_beta = DT(solver.beta_i / 2)
    @. ws.temp1 = _half_beta * ws.temp1 + ws.temp3

    # Forward FFT rhs_data into fft_buf, divide by helmholtz, inverse FFT
    @. ws.sw.fft_buf = ws.temp1
    for d = 1:ws.ndims_x
        ws.sw.fwd_plans[d] * ws.sw.fft_buf
    end
    @. ws.sw.fft_buf /= ws.helmholtz
    for d = 1:ws.ndims_x
        ws.sw.inv_plans[d] * ws.sw.fft_buf
    end
    @. E[pz].data = real(ws.sw.fft_buf)

    return nothing
end

function solve_fields!(
    sol::FieldSolution,
    moments::Moments,
    grid::Grid,
    solver::EMSolverDKPol,
    dt::Real,
)
    moments.J !== nothing || throw(ArgumentError("moments.J is required for EMSolverDKPol"))
    moments.Pi_diff !== nothing ||
        throw(ArgumentError("moments.Pi_diff is required for EMSolverDKPol"))
    if !solver.initialized[]
        initialize_staggered_B!(sol, grid, dt)
        solver.initialized[] = true
    end
    copyto!(sol.Enew, sol.E)
    ws = _get_em_dk_pol_workspace(sol, grid, solver)
    _step_dk_pol!(sol.Enew, sol.B, moments.J, moments.Pi_diff, grid, solver, dt, ws)
    return sol
end


struct EMSolverDKNoPol{T<:AbstractFloat} <: AbstractFieldSolver
    beta_i::T
    mu::T
end

function EMSolverDKNoPol(beta_i::Real, mu::Real)
    T = promote_type(typeof(float(beta_i)), typeof(float(mu)))
    return EMSolverDKNoPol{T}(T(beta_i), T(mu))
end

# ── EMDKNoPolWorkspace ────────────────────────────────────────────────────────

struct EMDKNoPolWorkspace{SW,CA,FA,K1V,K2V,KZV,KV}
    sw::SW
    kviews::KV       # NTuple{ndims_x} reshaped k views for _spectral_curl_hat!
    k1_view::K1V     # reshaped k for d1 direction (or zero-filled)
    k2_view::K2V     # reshaped k for d2 direction (or zero-filled)
    kz_view::KZV     # reshaped k for pz direction (or zero-filled)
    # complex per-call buffers:
    Bhat::NTuple{3,CA}
    Jhat::NTuple{2,CA}
    Pihat_buf::NTuple{3,CA}
    curlBhat::NTuple{3,CA}
    div_Pi_hat::CA
    Ehat::NTuple{3,CA}
    # pre-computed float constants:
    k2_perp::FA
    k2_tot::FA
    a31_pre::FA
    a32_pre::FA
    a33_pre::FA
    # per-call float scratch (for a-matrix and cofactors):
    a11::FA
    a12::FA
    a13::FA
    a21::FA
    a22::FA
    a23::FA
    m11::FA
    m12::FA
    m13::FA
    detA::FA
    ndims_x::Int
    d1::Int
    d2::Int
    pz::Int
end

function _make_kview_or_zero(
    sw::SpectralWorkspace,
    d::Int,
    ndims_x::Int,
    ndims_data::Int,
    arr::AbstractArray,
)
    if d <= ndims_x
        return reshape(sw.k[d], ntuple(i -> i == d ? length(sw.k[d]) : 1, ndims_data))
    else
        return fill!(similar(arr, eltype(arr), ntuple(_ -> 1, ndims_data)), zero(eltype(arr)))
    end
end

function _make_em_dk_no_pol_workspace(
    arr::AbstractArray,
    grid::Grid,
    ncomp::Int,
    solver::EMSolverDKNoPol,
)
    ndims_x = spatial_ndims(grid)
    ndims_data = ndims(arr)
    perp_dirs = [d for d = 1:ncomp if d != grid.Bdir]
    d1, d2, pz = perp_dirs[1], perp_dirs[2], grid.Bdir

    sw = _get_spectral_workspace(arr, grid)

    kviews = ntuple(
        d -> reshape(sw.k[d], ntuple(i -> i == d ? length(sw.k[d]) : 1, ndims_data)),
        ndims_x,
    )
    k1_view = _make_kview_or_zero(sw, d1, ndims_x, ndims_data, arr)
    k2_view = _make_kview_or_zero(sw, d2, ndims_x, ndims_data, arr)
    kz_view = _make_kview_or_zero(sw, pz, ndims_x, ndims_data, arr)

    DT = eltype(arr)
    c_arr() = similar(arr, Complex{DT})
    f_arr() = fill!(similar(arr), zero(DT))

    Bhat = ntuple(_ -> c_arr(), 3)
    Jhat = ntuple(_ -> c_arr(), 2)
    Pihat_buf = ntuple(_ -> c_arr(), 3)
    curlBhat = ntuple(_ -> c_arr(), 3)
    div_Pi_hat = c_arr()
    Ehat = ntuple(_ -> c_arr(), 3)

    k2_perp = f_arr()
    @. k2_perp = k1_view^2 + k2_view^2
    k2_tot = f_arr()
    @. k2_tot = k2_perp + kz_view^2

    a31_pre = f_arr();
    @. a31_pre = kz_view * k1_view
    a32_pre = f_arr();
    @. a32_pre = kz_view * k2_view
    lambda = DT((solver.beta_i / 2) * (1 + 1 / solver.mu))
    a33_pre = f_arr();
    @. a33_pre = -k2_perp - lambda

    return EMDKNoPolWorkspace(
        sw,
        kviews,
        k1_view,
        k2_view,
        kz_view,
        Bhat,
        Jhat,
        Pihat_buf,
        curlBhat,
        div_Pi_hat,
        Ehat,
        k2_perp,
        k2_tot,
        a31_pre,
        a32_pre,
        a33_pre,
        f_arr(),
        f_arr(),
        f_arr(),
        f_arr(),
        f_arr(),
        f_arr(),
        f_arr(),
        f_arr(),
        f_arr(),
        f_arr(),
        ndims_x,
        d1,
        d2,
        pz,
    )
end

function _get_em_dk_no_pol_workspace(
    sol::FieldSolution,
    grid::Grid,
    solver::EMSolverDKNoPol,
)
    arr = sol.E[1].data
    ndims_x = spatial_ndims(grid)
    ncomp = ncomponents(sol.E)
    key = (
        :em_dk_no_pol,
        size(arr),
        eltype(arr),
        ndims_x,
        ncomp,
        grid.Bdir,
        Tuple(grid.delta[1:ndims_x]),
        solver.beta_i,
        solver.mu,
    )
    lock(_solver_workspace_cache_lock) do
        get!(
            () -> _make_em_dk_no_pol_workspace(arr, grid, ncomp, solver),
            _solver_workspace_cache,
            key,
        )
    end
end

# ── _step_dk_no_pol! with workspace ──────────────────────────────────────────

function _step_dk_no_pol!(
    E::VectorField,
    B::VectorField,
    J_i_perp::VectorField,
    Pi_diff::VectorField,
    solver::EMSolverDKNoPol,
    dt::Real,
    ws::EMDKNoPolWorkspace,
)
    ndims_x = ws.ndims_x
    d1, d2, pz = ws.d1, ws.d2, ws.pz
    DT = eltype(ws.k2_perp)
    c = DT(2 / solver.beta_i)
    α = DT(c * dt)
    _dt = DT(dt)
    _half_beta = DT(solver.beta_i / 2)
    sw = ws.sw

    # Forward FFT inputs
    for d = 1:3
        _fwd_fft_to!(sw, B[d].data, ws.Bhat[d], ndims_x)
    end
    for d = 1:2
        _fwd_fft_to!(sw, J_i_perp[d].data, ws.Jhat[d], ndims_x)
    end
    for d = 1:ndims_x
        _fwd_fft_to!(sw, Pi_diff[d].data, ws.Pihat_buf[d], ndims_x)
    end

    # curlBhat (in-place into ws.curlBhat)
    _spectral_curl_hat!(ws.curlBhat, ws.Bhat, ws.kviews, ndims_x)

    # div_Pi_hat
    fill!(ws.div_Pi_hat, 0)
    for d = 1:ndims_x
        @. ws.div_Pi_hat += im * ws.kviews[d] * ws.Pihat_buf[d]
    end

    # w1 = c*curlBhat[d1] - J[1], w2 = c*curlBhat[d2] - J[2]  (in-place)
    @. ws.curlBhat[d1] = c * ws.curlBhat[d1] - ws.Jhat[1]   # w1
    @. ws.curlBhat[d2] = c * ws.curlBhat[d2] - ws.Jhat[2]   # w2
    # b3 = (beta_i/2)*div_Pi_hat  (in-place overwrite)
    @. ws.div_Pi_hat *= _half_beta

    # a-matrix entries (all real, per-call due to α = c*dt)
    k1v, k2v, kzv = ws.k1_view, ws.k2_view, ws.kz_view
    _one = one(DT)
    @. ws.a11 = _one - α * k1v * k2v
    @. ws.a12 = α * (ws.k2_tot - k2v^2)
    @. ws.a13 = -α * k2v * kzv
    @. ws.a21 = -α * (ws.k2_tot - k1v^2)
    @. ws.a22 = _one + α * k1v * k2v
    @. ws.a23 = α * k1v * kzv

    # Cofactors and determinant
    @. ws.m11 = ws.a22 * ws.a33_pre - ws.a23 * ws.a32_pre
    @. ws.m12 = ws.a21 * ws.a33_pre - ws.a23 * ws.a31_pre
    @. ws.m13 = ws.a21 * ws.a32_pre - ws.a22 * ws.a31_pre
    @. ws.detA = ws.a11 * ws.m11 - ws.a12 * ws.m12 + ws.a13 * ws.m13

    # Cramer's rule: Ehat = A⁻¹ b  (b1=w2=curlBhat[d2], b2=-w1=-curlBhat[d1], b3=div_Pi_hat)
    w1 = ws.curlBhat[d1]
    w2 = ws.curlBhat[d2]
    b3 = ws.div_Pi_hat
    @. ws.Ehat[d1] =
        (
            w2 * ws.m11 +
            ws.a12 * (w1 * ws.a33_pre + ws.a23 * b3) +
            ws.a13 * (-w1 * ws.a32_pre - ws.a22 * b3)
        ) / ws.detA
    @. ws.Ehat[d2] =
        (
            ws.a11 * (-w1 * ws.a33_pre - ws.a23 * b3) - w2 * ws.m12 +
            ws.a13 * (ws.a21 * b3 + w1 * ws.a31_pre)
        ) / ws.detA
    @. ws.Ehat[pz] =
        (
            ws.a11 * (ws.a22 * b3 + w1 * ws.a32_pre) -
            ws.a12 * (ws.a21 * b3 + w1 * ws.a31_pre) + w2 * ws.m13
        ) / ws.detA

    # Faraday: Bhat -= dt * curl(Ehat)  — reuse curlBhat buffers
    _spectral_curl_hat!(ws.curlBhat, ws.Ehat, ws.kviews, ndims_x)
    for d = 1:3
        @. ws.Bhat[d] -= _dt * ws.curlBhat[d]
    end

    # Inverse FFT
    for d = 1:3
        _inv_fft_from!(sw, ws.Ehat[d], E[d].data, ndims_x)
        _inv_fft_from!(sw, ws.Bhat[d], B[d].data, ndims_x)
    end

    return nothing
end

function solve_fields!(
    sol::FieldSolution,
    moments::Moments,
    grid::Grid,
    solver::EMSolverDKNoPol,
    dt::Real,
)
    moments.J !== nothing ||
        throw(ArgumentError("moments.J is required for EMSolverDKNoPol"))
    moments.Pi_diff !== nothing ||
        throw(ArgumentError("moments.Pi_diff is required for EMSolverDKNoPol"))
    copyto!(sol.Enew, sol.E)
    ws = _get_em_dk_no_pol_workspace(sol, grid, solver)
    _step_dk_no_pol!(sol.Enew, sol.B, moments.J, moments.Pi_diff, solver, dt, ws)
    return sol
end


# Initialise B to the staggered B^{-1/2} from B^0 and E^0 (EMSolverDKPol only).
function initialize_staggered_B!(sol::FieldSolution, grid::Grid, dt::Real)
    curlE = curl(sol.E, grid)
    DT = eltype(sol.B[1].data)
    for d = 1:ncomponents(sol.B)
        sol.B[d].data .+= DT(dt / 2) .* curlE[d].data
    end
    return sol
end
