using Test
using AbstractFFTs
using bslLD

@testset "Maxwell CN Vacuum" begin
    bslLD.set_execution_space!(exec=bslLD.backend())

    grid = bslLD.Grid([0.0], [2pi], [64], 1)
    dt = 0.05
    x = grid.xaxes[1]
    params = bslLD.VacuumMaxwellParams(c=1.0, ϵ0=1.0, μ0=1.0)

    E = bslLD.VectorField([
        zeros(length(x)),
        sin.(x),
        zeros(length(x)),
    ])
    B = bslLD.VectorField([
        zeros(length(x)),
        zeros(length(x)),
        sin.(x),
    ])

    energy_before = bslLD.electromagnetic_energy(E, B; params=params)
    divE_before, divB_before = bslLD.maxwell_constraints(E, B, grid)
    E_before = [copy(component.data) for component in E]
    B_before = [copy(component.data) for component in B]

    Ehat_before = [fft(component, (1,)) for component in E_before]
    Bhat_before = [fft(component, (1,)) for component in B_before]
    kx = reshape(bslLD.spectral_wavenumbers(E[1].data, grid, 1), :)
    alpha2 = (params.c * dt / 2)^2 .* (kx .^ 2)
    prefac = 1.0 ./ (1.0 .+ alpha2)
    diagonal = 1.0 .- alpha2
    curlEhat = [
        zero.(Ehat_before[1]),
        -im .* kx .* Ehat_before[3],
        im .* kx .* Ehat_before[2],
    ]
    curlBhat = [
        zero.(Bhat_before[1]),
        -im .* kx .* Bhat_before[3],
        im .* kx .* Bhat_before[2],
    ]
    Ehat_expected = [
        prefac .* (diagonal .* Ehat_before[d] .+ params.c^2 * dt .* curlBhat[d]) for d in 1:3
    ]
    Bhat_expected = [
        prefac .* (diagonal .* Bhat_before[d] .- dt .* curlEhat[d]) for d in 1:3
    ]
    E_expected = [real(ifft(component, (1,))) for component in Ehat_expected]
    B_expected = [real(ifft(component, (1,))) for component in Bhat_expected]

    solver = bslLD.EMSolverVacuum(; c=params.c, ϵ0=params.ϵ0, μ0=params.μ0)
    moments = bslLD.Moments(bslLD.ScalarField(zeros(length(x))))
    sol = bslLD.FieldSolution(E, B)
    bslLD.solve_fields!(sol, moments, grid, solver, dt)
    sol.E .= sol.Enew

    for d in 1:3
        @test maximum(abs.(E[d].data .- E_expected[d])) < 1e-10
        @test maximum(abs.(B[d].data .- B_expected[d])) < 1e-10
    end

    energy_after = bslLD.electromagnetic_energy(E, B; params=params)
    divE_after, divB_after = bslLD.maxwell_constraints(E, B, grid)

    @test abs(energy_after - energy_before) < 1e-10
    @test maximum(abs.(divE_before.data)) < 1e-10
    @test maximum(abs.(divB_before.data)) < 1e-10
    @test maximum(abs.(divE_after.data)) < 1e-10
    @test maximum(abs.(divB_after.data)) < 1e-10
end
