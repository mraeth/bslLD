# --- ExB drift: semi-discrete time derivative ---------------------------------
#
# Evaluates the right-hand side of the ExB drift advection as a Poisson bracket in
# the x1-x2 plane,
#
#   ∂f/∂t = {f, g} = ∂f/∂x1 ∂g/∂x2 − ∂g/∂x1 ∂f/∂x2 ,   g = φ / B_0 ,
#
# for B_0 = grid.b0 along x3. Derivatives are spectral (Nyquist wavenumber zero, as `differentiate!`
# from spectral_operators.jl), so the bracket is skew-symmetric by construction and no
# Arakawa-style averaging of finite-difference forms is needed. The bracket is batched
# over x3 (when NX == 3) and over all velocity dimensions purely by broadcasting. The FFTs run over
# the leading axes (x1, x2) together, which every FFT backend can batch (rocFFT cannot batch a 1D
# transform along x2 over axes on both sides of it).
#
# Time integration is left to the caller; `exb_euler!` provides a single fused explicit
# stage from which multi-stage schemes can be built.

struct ExBBracketPlan{FB,DGX,DG,CF,CX,PF,PX,K}
    df_buf1::FB  # ∂f/∂x1, phase-space sized
    df_buf2::FB  # ∂f/∂x2, phase-space sized
    dg_x::DGX    # (∂g/∂x1, ∂g/∂x2), x-shaped
    dg::DG       # same memory reshaped to full phase-space rank for broadcasting
    fhat::CF     # complex buffer, phase-space sized
    ghat::CX     # complex buffer, x-shaped
    plans_f::PF  # in-place (forward, inverse) FFT over x1, x2 of fhat
    plans_g::PX  # same for ghat
    k::K         # (k1, k2) on the backend, Nyquist zeroed
end

function ExBBracketPlan(
    f::DistributionGrid{Float64,NX,NV,NXNV,Cart},
    grid::CartGrid,
) where {NX,NV,NXNV}
    NX >= 2 || throw(
        ArgumentError("the ExB bracket requires at least two spatial dimensions, got $NX"),
    )
    xshape = ntuple(i -> length(grid.xaxes[i]), Val(NX))
    broadcast_shape = ntuple(i -> i <= NX ? xshape[i] : 1, Val(NXNV))
    dg_x = ntuple(_ -> similar(f.data, Float64, xshape), Val(2))
    dg = map(a -> reshape(a, broadcast_shape), dg_x)
    fhat = similar(f.data, ComplexF64)
    ghat = similar(f.data, ComplexF64, xshape)
    plans(a) = (plan_fft!(a, 1:2), plan_ifft!(a, 1:2))
    return ExBBracketPlan(
        similar(f.data, Float64),
        similar(f.data, Float64),
        dg_x,
        dg,
        fhat,
        ghat,
        plans(fhat),
        plans(ghat),
        ntuple(d -> spectral_wavenumbers(f.data, grid, d), 2),
    )
end

# out = ∂a/∂x_d (d = 1, 2) through the complex buffer `buf` and its FFT plans over x1, x2.
function _exb_derivative!(out, a, buf, (P, iP), k, d)
    buf .= a
    P * buf
    kd = reshape(k[d], ntuple(i -> i == d ? length(k[d]) : 1, ndims(buf)))
    buf .*= im .* kd
    iP * buf
    out .= real.(buf)
    return out
end

const _exb_plan_cache = IdDict{Any,Dict{Any,ExBBracketPlan}}()
const _exb_plan_cache_lock = ReentrantLock()

@inline _exb_plan_cache_key(grid::CartGrid) = (grid.xaxes, grid.vaxes, grid.Bdir)

function _get_exb_plan(
    f::DistributionGrid{Float64,NX,NV,NXNV,Cart},
    grid::CartGrid,
) where {NX,NV,NXNV}
    lock(_exb_plan_cache_lock) do
        plans_for_data = get!(() -> Dict{Any,ExBBracketPlan}(), _exb_plan_cache, f.data)
        get!(() -> ExBBracketPlan(f, grid), plans_for_data, _exb_plan_cache_key(grid))
    end
end

function _similar_distribution(
    f::DistributionGrid{DT,NX,NV,NXNV,ID},
) where {DT,NX,NV,NXNV,ID}
    new_raw = similar(f.data)
    new_sf = ScalarField(new_raw)
    return DistributionGridImpl{NX,NV,ID,typeof(new_sf)}(new_sf)
end

# Fill plan.dg with ∇_{x1,x2}(φ/B_0) and plan.df_buf{1,2} with ∂f/∂x{1,2}. All reads of
# `f` happen here, so callers may safely write their output over `f` afterwards.
function _exb_setup!(
    f::DistributionGrid{Float64,NX,NV,NXNV,Cart},
    phi::ScalarField,
    grid::CartGrid,
    plan::ExBBracketPlan,
) where {NX,NV,NXNV}
    NX >= 2 || throw(
        ArgumentError("the ExB bracket requires at least two spatial dimensions, got $NX"),
    )
    grid.Bdir == 3 || throw(
        ArgumentError("the ExB bracket assumes B along x3, got grid.Bdir = $(grid.Bdir)"),
    )
    size(phi.data) == size(plan.dg_x[1]) || throw(
        ArgumentError(
            "potential of size $(size(phi.data)) does not match the spatial grid $(size(plan.dg_x[1]))",
        ),
    )

    grid.b0 == 0.0 && error("ExB bracket requires nonzero b0")
    inv_b0 = inv(grid.b0)
    for d = 1:2
        _exb_derivative!(plan.dg_x[d], phi.data, plan.ghat, plan.plans_g, plan.k, d)
        plan.dg_x[d] .*= inv_b0
    end

    _exb_derivative!(plan.df_buf1, f.data, plan.fhat, plan.plans_f, plan.k, 1)
    _exb_derivative!(plan.df_buf2, f.data, plan.fhat, plan.plans_f, plan.k, 2)
    return nothing
end

# ∂f/∂t = {f, φ/B_0}, written into `df`.
function exb_bracket!(
    df::DistributionGrid{Float64,NX,NV,NXNV,Cart},
    f::DistributionGrid{Float64,NX,NV,NXNV,Cart},
    phi::ScalarField,
    grid::CartGrid,
    plan::ExBBracketPlan,
) where {NX,NV,NXNV}
    _exb_setup!(f, phi, grid, plan)
    @. df.data = plan.df_buf1 * plan.dg[2] - plan.dg[1] * plan.df_buf2
    return df
end

function exb_bracket!(
    df::DistributionGrid{Float64,NX,NV,NXNV,Cart},
    f::DistributionGrid{Float64,NX,NV,NXNV,Cart},
    phi::ScalarField,
    grid::CartGrid,
) where {NX,NV,NXNV}
    return exb_bracket!(df, f, phi, grid, _get_exb_plan(f, grid))
end

function exb_bracket(
    f::DistributionGrid{Float64,NX,NV,NXNV,Cart},
    phi::ScalarField,
    grid::CartGrid,
) where {NX,NV,NXNV}
    return exb_bracket!(_similar_distribution(f), f, phi, grid)
end

# One explicit Euler stage f_out = f_in + dt {f_in, φ/B_0}. Safe with f_out === f_in.
function exb_euler!(
    f_out::DistributionGrid{Float64,NX,NV,NXNV,Cart},
    f_in::DistributionGrid{Float64,NX,NV,NXNV,Cart},
    phi::ScalarField,
    grid::CartGrid,
    dt::Real,
    plan::ExBBracketPlan,
) where {NX,NV,NXNV}
    _exb_setup!(f_in, phi, grid, plan)
    @. f_out.data = f_in.data + dt * (plan.df_buf1 * plan.dg[2] - plan.dg[1] * plan.df_buf2)
    return f_out
end

function exb_euler!(
    f_out::DistributionGrid{Float64,NX,NV,NXNV,Cart},
    f_in::DistributionGrid{Float64,NX,NV,NXNV,Cart},
    phi::ScalarField,
    grid::CartGrid,
    dt::Real,
) where {NX,NV,NXNV}
    return exb_euler!(f_out, f_in, phi, grid, dt, _get_exb_plan(f_in, grid))
end

# Walls along x1 (`Mirror()` grid: N nodes, walls on nodes 1 and N): the extension to the periodic grid of 2(N−1)
# nodes, even (s = 1) or odd (s = −1; the wall values are then taken as 0), ã_j = a_j (j ≤ N), ã_{2N−j} = s a_j
# (2 ≤ j ≤ N−1), and the restriction back to nodes 1…N.
@kernel function _x1_extend_kernel!(e, a, s, N)
    i, c = @index(Global, NTuple)
    if i <= N
        e[i, c] = (s < 0 && (i == 1 || i == N)) ? zero(eltype(e)) : a[i, c]
    else
        e[i, c] = s * a[2N - i, c]
    end
end

function _x1_extend(a::AbstractArray, s)
    N = size(a, 1)
    A = reshape(a, N, :)
    E = similar(A, 2N - 2, size(A, 2))
    be = get_backend(A)
    _x1_extend_kernel!(be)(E, A, eltype(a)(s), N; ndrange = size(E))
    KernelAbstractions.synchronize(be)
    return reshape(E, 2N - 2, size(a)[2:end]...)
end

_x1_restrict(e::AbstractArray, N::Int) = copy(selectdim(e, 1, 1:N))

# 2/3-rule Galerkin truncation over x1, x2 in place: keeps the modes |k_d| ≤ N_d/3. The mask is
# built on the host and copied to the array's backend.
function exb_dealias!(a::AbstractArray{Float64})
    N1, N2 = size(a, 1), size(a, 2)
    keep(N) = abs.(fftfreq(N, N)) .<= N / 3
    mask = copyto!(similar(a, N1, N2), Float64.(keep(N1) .* keep(N2)'))
    a .= real(ifft(fft(a, 1:2) .* reshape(mask, N1, N2, ntuple(_ -> 1, ndims(a) - 2)...), 1:2))
    return a
end

# Classical RK4 over dt for ∂f/∂t = {f, φ/B_0} with φ frozen; overwrites f. `dealias`: each stage is
# P{Pf, Pφ} with P the 2/3 truncation (exb_dealias!), skew-symmetric to round-off (Σ f P{Pf, Pφ} = 0);
# the modes of f above 2/3 are left unchanged.
function exb_rk4!(
    f::DistributionGrid{Float64,NX,NV,NXNV,Cart},
    phi::ScalarField,
    grid::CartGrid,
    dt::Real;
    dealias::Bool = false,
) where {NX,NV,NXNV}
    plan = _get_exb_plan(f, grid)
    f0 = copy(f.data)
    k = _similar_distribution(f)
    if dealias
        phi = ScalarField(exb_dealias!(copy(phi.data)))
        fP = _similar_distribution(f)
        rhs! = (k, f) -> (fP.data .= f.data; exb_dealias!(fP.data);
                          exb_bracket!(k, fP, phi, grid, plan); exb_dealias!(k.data))
    else
        rhs! = (k, f) -> exb_bracket!(k, f, phi, grid, plan)
    end
    rhs!(k, f)
    acc = f0 .+ dt / 6 .* k.data
    for (c, w) in ((0.5, 1 / 3), (0.5, 1 / 3), (1.0, 1 / 6))
        f.data .= f0 .+ c * dt .* k.data
        rhs!(k, f)
        acc .+= w * dt .* k.data
    end
    f.data .= acc
    return f
end

# Species forwarding overloads
ExBBracketPlan(sp::DistSpecies, grid::CartGrid) = ExBBracketPlan(sp.dist, grid)
_get_exb_plan(sp::DistSpecies, grid::CartGrid) = _get_exb_plan(sp.dist, grid)
exb_bracket(sp::DistSpecies, phi::ScalarField, grid::CartGrid) = exb_bracket(sp.dist, phi, grid)
exb_bracket!(
    df::DistributionGrid,
    sp::DistSpecies,
    phi::ScalarField,
    grid::CartGrid,
    plan::ExBBracketPlan,
) = exb_bracket!(df, sp.dist, phi, grid, plan)
exb_bracket!(df::DistributionGrid, sp::DistSpecies, phi::ScalarField, grid::CartGrid) =
    exb_bracket!(df, sp.dist, phi, grid)
exb_bracket!(
    df::DistSpecies,
    f_in::DistSpecies,
    phi::ScalarField,
    grid::CartGrid,
    plan::ExBBracketPlan,
) = exb_bracket!(df.dist, f_in.dist, phi, grid, plan)
exb_bracket!(df::DistSpecies, f_in::DistSpecies, phi::ScalarField, grid::CartGrid) =
    exb_bracket!(df.dist, f_in.dist, phi, grid)
exb_euler!(
    f_out::DistributionGrid,
    sp::DistSpecies,
    phi::ScalarField,
    grid::CartGrid,
    dt::Real,
    plan::ExBBracketPlan,
) = exb_euler!(f_out, sp.dist, phi, grid, dt, plan)
exb_euler!(
    f_out::DistributionGrid,
    sp::DistSpecies,
    phi::ScalarField,
    grid::CartGrid,
    dt::Real,
) = exb_euler!(f_out, sp.dist, phi, grid, dt)
exb_euler!(
    f_out::DistSpecies,
    f_in::DistSpecies,
    phi::ScalarField,
    grid::CartGrid,
    dt::Real,
    plan::ExBBracketPlan,
) = exb_euler!(f_out.dist, f_in.dist, phi, grid, dt, plan)
exb_euler!(f_out::DistSpecies, f_in::DistSpecies, phi::ScalarField, grid::CartGrid, dt::Real) =
    exb_euler!(f_out.dist, f_in.dist, phi, grid, dt)
exb_rk4!(sp::DistSpecies, phi::ScalarField, grid::CartGrid, dt::Real; kw...) = (exb_rk4!(sp.dist, phi, grid, dt; kw...); sp)
