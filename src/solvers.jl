abstract type AbstractFieldSolver end

struct Moments{SF<:ScalarField,VF<:Union{Nothing,VectorField},PF<:Union{Nothing,VectorField}}
    rho       :: SF
    J         :: VF
    Pi_diff_z :: PF

    function Moments(
        rho::SF,
        J::VF=nothing,
        Pi_diff_z::PF=nothing,
    ) where {
        SF<:ScalarField,
        VF<:Union{Nothing,VectorField},
        PF<:Union{Nothing,VectorField},
    }
        if J !== nothing
            axes(rho.data) == axes(J[1].data) || throw(DimensionMismatch("Moments rho and J must share spatial axes"))
        end
        return new{SF,VF,PF}(rho, J, Pi_diff_z)
    end
end

struct PoissonFieldSolver{DT<:AbstractFloat} <: AbstractFieldSolver
    factor::DT
end
PoissonFieldSolver() = PoissonFieldSolver(1.0)
PoissonFieldSolver(factor::Real) = PoissonFieldSolver(float(factor))
struct AdiabaticFieldSolver <: AbstractFieldSolver end




struct FieldSolution{VF<:VectorField}
    E::VF
    B::VF
    B0::VF
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

function solve_fields(moments::Moments, grid::Grid, solver::PoissonFieldSolver)
    phi = poisson_potential(moments.rho, grid, solver.factor)
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

function solve_fields!(
    sol::FieldSolution,
    moments::Moments,
    grid::Grid,
    solver::SemiImplicitEMSolver,
    dt::Real,
)
    moments.J         !== nothing || throw(ArgumentError("moments.J (J_i_⊥) is required for SemiImplicitEMSolver"))
    moments.Pi_diff_z !== nothing || throw(ArgumentError("moments.Pi_diff_z (Π_e,z − Π_i,z) is required for SemiImplicitEMSolver"))
    grid.Bdir == 3 || throw(ArgumentError("SemiImplicitEMSolver requires grid.Bdir == 3"))
    _step_semi_implicit_em!(sol.E, sol.B, moments.J, moments.Pi_diff_z, grid, solver, dt)
    return sol
end
