abstract type AbstractFieldSolver end

struct Moments{SF<:ScalarField,VF<:Union{Nothing,VectorField},PF<:Union{Nothing,VectorField}}
    rho     :: SF
    J       :: VF
    Pi_diff :: PF

    function Moments(
        rho::SF,
        J::VF=nothing,
        Pi_diff::PF=nothing,
    ) where {
        SF<:ScalarField,
        VF<:Union{Nothing,VectorField},
        PF<:Union{Nothing,VectorField},
    }
        if J !== nothing
            axes(rho.data) == axes(J[1].data) ||
                throw(DimensionMismatch("Moments rho and J must share spatial axes"))
        end
        if Pi_diff !== nothing
            axes(rho.data) == axes(Pi_diff[1].data) ||
                throw(DimensionMismatch("Moments rho and Pi_diff must share spatial axes"))
        end
        return new{SF,VF,PF}(rho, J, Pi_diff)
    end
end

struct PoissonFieldSolver{DT<:AbstractFloat} <: AbstractFieldSolver
    factor::DT
end
PoissonFieldSolver() = PoissonFieldSolver(1.0)
PoissonFieldSolver(factor::Real) = PoissonFieldSolver(float(factor))
struct AdiabaticFieldSolver <: AbstractFieldSolver end


struct FieldSolution{VF<:VectorField}
    E       :: VF
    B       :: VF
    B0      :: VF
    Enew    :: VF   # E^{n+1}, written by solve_fields!
    Ecenter :: VF   # E^{n+1/2} = ½(E + Enew), built by the stepper
    Bhalf   :: VF   # B^{n-1/2} / B^{n+1/2}
end

function FieldSolution(E::VF, B::VF, B0::VF) where {VF<:VectorField}
    return FieldSolution{VF}(
        E, B, B0,
        zero_vectorfield_like(E),
        zero_vectorfield_like(E),
        zero_vectorfield_like(B),
    )
end

function zero_scalarfield_like(field::ScalarField)
    return ScalarField(zero.(field.data))
end

function vectorfield_from_spatial_components(components::AbstractVector{<:ScalarField})
    ncomp = length(components)
    1 <= ncomp <= 3 || throw(ArgumentError("field solvers support between 1 and 3 physical components"))

    padded = ScalarField[components...]
    if ncomp < 3
        zero_component = zero_scalarfield_like(first(components))
        append!(padded, [zero_component for _ in 1:(3 - ncomp)])
    end

    return VectorField(padded)
end

function zero_vectorfield3(grid::Grid)
    dims = Tuple(length.(grid.xaxes))
    return VectorField([zeros(Float64, dims) for _ in 1:3])
end

function background_field(grid::Grid)
    1 <= grid.Bdir <= 3 || throw(ArgumentError("grid.Bdir must be between 1 and 3"))

    dims = Tuple(length.(grid.xaxes))
    components = [zeros(Float64, dims) for _ in 1:3]
    components[grid.Bdir] .= grid.b0
    return VectorField(components)
end

function poisson_potential(rho::ScalarField, grid::Grid, coefficient::T) where {T<:AbstractFloat}
    rhohat = fft_spatial(rho.data, grid)
    k2 = spectral_wave_number_squared(rho, grid) .* coefficient
    phihat = -rhohat
    zero_mode = k2 .== 0
    phihat[.!zero_mode] ./= k2[.!zero_mode]
    phihat[zero_mode] .= 0

    return ScalarField(real(ifft_spatial(phihat, grid)))
end

function electric_field_from_potential(phi::ScalarField, grid::Grid)
    e_components = [ScalarField(-differentiate(phi, grid, dir).data) for dir in 1:spatial_ndims(grid)]
    return vectorfield_from_spatial_components(e_components)
end


function fourier_filter(field::ScalarField, grid::Grid, cutoff_fraction::Real)
    fieldhat = fft_spatial(field.data, grid)
    k2 = spectral_wave_number_squared(field, grid)
    kmax2 = maximum(k2) * cutoff_fraction^2
    fieldhat[k2 .> kmax2] .= 0
    return ScalarField(real(ifft_spatial(fieldhat, grid)))
end


function solve_fields(moments::Moments, grid::Grid, solver::PoissonFieldSolver)
    phi = poisson_potential(moments.rho, grid, solver.factor)
    phi = fourier_filter(phi, grid, 0.5)
    return FieldSolution(
        electric_field_from_potential(phi, grid),
        zero_vectorfield3(grid),
        background_field(grid),
    )
end

function solve_fields(moments::Moments, grid::Grid, ::AdiabaticFieldSolver)
    return FieldSolution(
        electric_field_from_potential(moments.rho, grid),
        zero_vectorfield3(grid),
        background_field(grid),
    )
end

struct SemiImplicitEMSolver{T<:AbstractFloat} <: AbstractFieldSolver
    beta_i :: T
    mu     :: T
end

function SemiImplicitEMSolver(beta_i::Real, mu::Real)
    T = promote_type(typeof(float(beta_i)), typeof(float(mu)))
    return SemiImplicitEMSolver{T}(T(beta_i), T(mu))
end

function _step_semi_implicit_em!(
    E::VectorField,
    Bhalf::VectorField,
    J_i_perp::VectorField,
    Pi_diff::VectorField,
    grid::Grid,
    solver::SemiImplicitEMSolver,
    dt::Real,
)
    ndims_x     = spatial_ndims(grid)
    ncomp_field = ncomponents(E)
    perp_dirs = [d for d in 1:ncomp_field if d != grid.Bdir]

    # Explicit Faraday: Bhalf^{n-1/2} -> Bhalf^{n+1/2}
    curlE = curl(E, grid)
    for d in 1:ncomponents(Bhalf)
        Bhalf[d].data .-= dt .* curlE[d].data
    end

    # Implicit E_⊥ update: [I - (dt/μ)R] E_⊥^{n+1} = RHS_⊥
    curlB = curl(Bhalf, grid)
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

    if ndims_x >= 3
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
    sol::FieldSolution,
    moments::Moments,
    grid::Grid,
    solver::SemiImplicitEMSolver,
    dt::Real,
)
    moments.J       !== nothing || throw(ArgumentError("moments.J is required for SemiImplicitEMSolver"))
    moments.Pi_diff !== nothing || throw(ArgumentError("moments.Pi_diff is required for SemiImplicitEMSolver"))
    copyto!(sol.Enew, sol.E)
    _step_semi_implicit_em!(sol.Enew, sol.Bhalf, moments.J, moments.Pi_diff, grid, solver, dt)
    return sol
end


struct FullyImplicitEMSolver{T<:AbstractFloat} <: AbstractFieldSolver
    beta_i :: T
    mu     :: T
end

function FullyImplicitEMSolver(beta_i::Real, mu::Real)
    T = promote_type(typeof(float(beta_i)), typeof(float(mu)))
    return FullyImplicitEMSolver{T}(T(beta_i), T(mu))
end

function _step_fully_implicit_em!(
    E        :: VectorField,
    B        :: VectorField,
    J_i_perp :: VectorField,
    Pi_diff  :: VectorField,
    grid     :: Grid,
    solver   :: FullyImplicitEMSolver,
    dt       :: Real,
)
    ndims_x     = spatial_ndims(grid)
    ncomp_field = ncomponents(E)
    perp_dirs   = [d for d in 1:ncomp_field if d != grid.Bdir]
    d1, d2, pz  = perp_dirs[1], perp_dirs[2], grid.Bdir

    c = 2 / solver.beta_i
    λ = (solver.beta_i / 2) * (1 + 1 / solver.mu)
    α = c * dt

    Bhat   = [fft_spatial(B[d].data, grid) for d in 1:3]
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

    # Faraday: B^{n+1} = B^n − dt (ik × E^{n+1})
    Enew_hat = similar(Bhat)
    Enew_hat[d1] = Ehat_d1
    Enew_hat[d2] = Ehat_d2
    Enew_hat[pz] = Ehat_pz
    curlEnew_hat = spectral_curl_hat(Enew_hat, kviews, ndims_x)
    for d in 1:3
        Bhat[d] .= Bhat[d] .- dt .* curlEnew_hat[d]
    end

    E[d1].data .= real(ifft_spatial(Ehat_d1, grid))
    E[d2].data .= real(ifft_spatial(Ehat_d2, grid))
    E[pz].data .= real(ifft_spatial(Ehat_pz, grid))
    for d in 1:3
        B[d].data .= real(ifft_spatial(Bhat[d], grid))
    end

    return nothing
end


function solve_fields!(
    sol     :: FieldSolution,
    moments :: Moments,
    grid    :: Grid,
    solver  :: FullyImplicitEMSolver,
    dt      :: Real,
)
    moments.J       !== nothing || throw(ArgumentError("moments.J is required for FullyImplicitEMSolver"))
    moments.Pi_diff !== nothing || throw(ArgumentError("moments.Pi_diff is required for FullyImplicitEMSolver"))
    copyto!(sol.Enew, sol.E)
    _step_fully_implicit_em!(sol.Enew, sol.B, moments.J, moments.Pi_diff, grid, solver, dt)
    copyto!(sol.Bhalf, sol.B)
    return sol
end


# Initialise the staggered B^{-1/2} from B^0 and E^0 before the time loop.
function initialize_Bhalf!(sol::FieldSolution, grid::Grid, dt::Real)
    curlE = curl(sol.E, grid)
    for d in 1:ncomponents(sol.Bhalf)
        sol.Bhalf[d].data .= sol.B[d].data .+ (dt / 2) .* curlE[d].data
    end
    return sol
end
