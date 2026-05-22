cartesian_test_cases() = ((1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (3, 1))

all_axes(grid) = (grid.xaxes..., grid.vaxes...)

function cartesian_grid(nx::Int, nv::Int)
    eta_min = vcat(fill(0.0, nx), fill(-1.5, nv))
    eta_max = vcat(fill(2pi, nx), fill(1.5, nv))
    counts = vcat(fill(8, nx), fill(6, nv))
    return bslLD.Grid(eta_min, eta_max, counts, 0.05, 1, nx)
end

function advection_seed_data(grid)
    nx = length(grid.xaxes)
    nv = length(grid.vaxes)
    dims = Tuple(length.(all_axes(grid)))

    return Float64[
        sum(0.1 * dir * sin(2pi * (idx[dir] - 1) / dims[dir]) for dir in 1:nx) +
        sum(0.2 * dir * cos(2pi * (idx[nx + dir] - 1) / dims[nx + dir]) for dir in 1:nv)
        for idx in CartesianIndices(dims)
    ]
end

function expected_after_advect_x(grid)
    nx = length(grid.xaxes)
    nv = length(grid.vaxes)
    dims = Tuple(length.(all_axes(grid)))
    expected = Array{Float64}(undef, dims...)

    for idx in CartesianIndices(expected)
        shift_sum = 0.0
        for dir in 1:nx
            shift = dir <= nv ? grid.dt * grid.vaxes[dir][idx[nx + dir]] / grid.delta[dir] : 0.0
            shift_sum += 0.1 * dir * sin(2pi * ((idx[dir] - 1) - shift) / dims[dir])
        end

        velocity_sum = sum(
            0.2 * dir * cos(2pi * (idx[nx + dir] - 1) / dims[nx + dir]) for dir in 1:nv
        )

        expected[idx] = shift_sum + velocity_sum
    end

    return expected
end

function constant_electric_field(grid)
    dims = Tuple(length.(grid.xaxes))
    nv = length(grid.vaxes)

    return bslLD.VectorField([
        fill(0.15 * dir, dims) for dir in 1:nv
    ])
end

function expected_after_advect_v(grid, e_field)
    nx = length(grid.xaxes)
    nv = length(grid.vaxes)
    dims = Tuple(length.(all_axes(grid)))
    field_strengths = [component.data[begin] for component in e_field]
    expected = Array{Float64}(undef, dims...)

    for idx in CartesianIndices(expected)
        spatial_sum = sum(0.1 * dir * sin(2pi * (idx[dir] - 1) / dims[dir]) for dir in 1:nx)

        shift_sum = 0.0
        for dir in 1:nv
            shift = grid.dt * field_strengths[dir] / grid.delta[nx + dir]
            shift_sum += 0.2 * dir * cos(2pi * ((idx[nx + dir] - 1) - shift) / dims[nx + dir])
        end

        expected[idx] = spatial_sum + shift_sum
    end

    return expected
end

@testset "Advection" begin
    bslLD.set_execution_space!(exec=bslLD.backend())

    @testset "Cartesian ndmv exactness" begin
        for (nx, nv) in cartesian_test_cases()
            @testset "$(nx)d$(nv)v" begin
                grid = cartesian_grid(nx, nv)
                initial = advection_seed_data(grid)

                f_x = bslLD.Distribution(grid, 0.01)
                f_x.data .= initial
                expected_x = expected_after_advect_x(grid)
                initial_mass_x = sum(f_x.data)
                bslLD.advectX!(f_x, grid)

                @test size(f_x.data) == Tuple(length.(all_axes(grid)))
                @test all(isfinite, f_x.data)
                @test maximum(abs.(f_x.data .- expected_x)) < 1e-10
                @test abs(sum(f_x.data) - initial_mass_x) < 1e-10 * max(abs(initial_mass_x), 1.0)

                f_v = bslLD.Distribution(grid, 0.01)
                f_v.data .= initial
                e_field = constant_electric_field(grid)
                expected_v = expected_after_advect_v(grid, e_field)
                initial_mass_v = sum(f_v.data)
                bslLD.advectV!(f_v, grid, e_field)

                @test size(f_v.data) == Tuple(length.(all_axes(grid)))
                @test all(isfinite, f_v.data)
                @test maximum(abs.(f_v.data .- expected_v)) < 1e-10
                @test abs(sum(f_v.data) - initial_mass_v) < 1e-10 * max(abs(initial_mass_v), 1.0)
            end
        end
    end
end
