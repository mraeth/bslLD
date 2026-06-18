struct PoissonSolver{DT<:AbstractFloat} <: AbstractFieldSolver
    factor::DT
end
PoissonSolver() = PoissonSolver(1.0)
PoissonSolver(factor::Real) = PoissonSolver(float(factor))

struct AdiabaticSolver <: AbstractFieldSolver end


function poisson_potential(
    rho::ScalarField,
    grid::Grid,
    coefficient::T,
) where {T<:AbstractFloat}
    rhohat = fft_spatial(rho.data, grid)
    k2 = spectral_wave_number_squared(rho, grid) .* coefficient
    phihat = -rhohat
    zero_mode = k2 .== 0
    phihat[.!zero_mode] ./= k2[.!zero_mode]
    phihat[zero_mode] .= 0

    return ScalarField(real(ifft_spatial(phihat, grid)))
end

function electric_field_from_potential(phi::ScalarField, grid::Grid)
    e_components =
        [ScalarField(-differentiate(phi, grid, dir).data) for dir = 1:spatial_ndims(grid)]
    return vectorfield_from_spatial_components(e_components)
end

function fourier_filter(field::ScalarField, grid::Grid, cutoff_fraction::Real)
    fieldhat = fft_spatial(field.data, grid)
    k2 = spectral_wave_number_squared(field, grid)
    kmax2 = maximum(k2) * cutoff_fraction^2
    fieldhat[k2 .> kmax2] .= 0
    return ScalarField(real(ifft_spatial(fieldhat, grid)))
end


function solve_fields(moments::Moments, grid::Grid, solver::PoissonSolver)
    phi = poisson_potential(moments.rho, grid, solver.factor)
    phi = fourier_filter(phi, grid, 0.5)
    return FieldSolution(electric_field_from_potential(phi, grid), zero_vectorfield3(grid))
end

function solve_fields(moments::Moments, grid::Grid, ::AdiabaticSolver)
    return FieldSolution(
        electric_field_from_potential(moments.rho, grid),
        zero_vectorfield3(grid),
    )
end
