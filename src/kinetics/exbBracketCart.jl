# --- ExB drift: semi-discrete time derivative ---------------------------------
#
# Evaluates the right-hand side of the ExB drift advection as a Poisson bracket in
# the x1-x2 plane,
#
#   ∂f/∂t = {f, g} = ∂f/∂x1 ∂g/∂x2 − ∂g/∂x1 ∂f/∂x2 ,   g = φ / B_0 ,
#
# for B_0 = grid.b0 along x3. Derivatives are spectral (`_differentiate_impl!` from
# spectral_operators.jl), so the bracket is skew-symmetric by construction and no
# Arakawa-style averaging of finite-difference forms is needed. The bracket is batched
# over x3 (when NX == 3) and over all velocity dimensions purely by broadcasting.
#
# Time integration is left to the caller; `exb_euler!` provides a single fused explicit
# stage from which multi-stage schemes can be built.

struct ExBBracketPlan{FB,DGX,DG,WF,WP}
    df_buf1::FB  # ∂f/∂x1, phase-space sized
    df_buf2::FB  # ∂f/∂x2, phase-space sized
    dg_x::DGX    # (∂g/∂x1, ∂g/∂x2), x-shaped, written by _differentiate_impl!
    dg::DG       # same memory reshaped to full phase-space rank for broadcasting
    ws_f::WF     # spectral workspace for the phase-space array
    ws_phi::WP   # spectral workspace for the potential
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
    return ExBBracketPlan(
        similar(f.data, Float64),
        similar(f.data, Float64),
        dg_x,
        dg,
        _get_spectral_workspace(f.data, grid),
        _get_spectral_workspace(dg_x[1], grid),
    )
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
    data = similar(f.data)
    return DistributionGrid{DT,NX,NV,NXNV,ID,typeof(data)}(data, f.m, f.q)
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

    inv_b0 = inv(grid.b0)
    for d = 1:2
        _differentiate_impl!(plan.dg_x[d], phi.data, plan.ws_phi, d)
        plan.dg_x[d] .*= inv_b0
    end

    _differentiate_impl!(plan.df_buf1, f.data, plan.ws_f, 1)
    _differentiate_impl!(plan.df_buf2, f.data, plan.ws_f, 2)
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
