exb_grid(; nx = 2, nv = 2, nxpts = 16, nvpts = 16, b0 = 2.0, Bdir = 3) = bslLD.Grid(
    vcat(fill(0.0, nx), fill(-6.0, nv)),
    vcat(fill(2pi, nx), fill(6.0, nv)),
    vcat(fill(nxpts, nx), fill(nvpts, nv)),
    nx,
    b0,
    Bdir,
)

# Separable phase-space array: prod of per-axis factors, one factor per axis.
function exb_separable(grid, xfactors, vfactors)
    axes_all = (grid.xaxes..., grid.vaxes...)
    factors = vcat(
        [xfactors[d].(collect(axes_all[d])) for d = 1:length(grid.xaxes)],
        [
            vfactors[d].(collect(axes_all[length(grid.xaxes)+d])) for
            d = 1:length(grid.vaxes)
        ],
    )
    return bslLD.outer_product(factors)
end

exb_maxwellian() = (v -> exp(-v^2 / 2))

# f's x-part is sin(x1)cos(x2)·h(x3), φ = cos(x1)sin(x2)·h3phi(x3), so
#   {f, φ/B0} = h(x3)·h3phi(x3)/B0 · (cos²x1 cos²x2 − sin²x1 sin²x2) · W(v)
function exb_test_fields(grid; h = (x -> 1.0), h3phi = (x -> 1.0))
    nx = length(grid.xaxes)
    nv = length(grid.vaxes)
    xf = nx == 2 ? [sin, cos] : [sin, cos, h]
    fdata = exb_separable(grid, xf, fill(exb_maxwellian(), nv))

    phix = nx == 2 ? [cos, sin] : [cos, sin, h3phi]
    phidata = bslLD.outer_product([phix[d].(collect(grid.xaxes[d])) for d = 1:nx])

    xexp = nx == 2 ? [x -> cos(x)^2, x -> cos(x)^2] : [x -> cos(x)^2, x -> cos(x)^2, h]
    xexp2 = nx == 2 ? [x -> sin(x)^2, x -> sin(x)^2] : [x -> sin(x)^2, x -> sin(x)^2, h]
    if nx == 3
        xexp[3] = (x -> h(x) * h3phi(x))
        xexp2[3] = (x -> h(x) * h3phi(x))
    end
    expected =
        (
            exb_separable(grid, xexp, fill(exb_maxwellian(), nv)) .-
            exb_separable(grid, xexp2, fill(exb_maxwellian(), nv))
        ) ./ grid.b0

    return fdata, bslLD.ScalarField(phidata), expected
end

function exb_distribution(grid, data)
    f = bslLD.Distribution(grid, 0.0)
    f.data .= data
    return f
end

@testset "ExB Poisson bracket" begin
    @testset "analytic bracket (2d2v)" begin
        grid = exb_grid()
        fdata, phi, expected = exb_test_fields(grid)
        f = exb_distribution(grid, fdata)
        df = bslLD.exb_bracket(f, phi, grid)
        @test maximum(abs, df.data .- expected) < 1e-10
    end

    @testset "antisymmetry: {f, f} = 0" begin
        grid = exb_grid()
        phidata = bslLD.outer_product([
            sin.(collect(grid.xaxes[1])),
            cos.(collect(grid.xaxes[2])),
        ])
        f = exb_distribution(
            grid,
            exb_separable(grid, [sin, cos], [v -> 1.0, v -> 1.0]) .* grid.b0,
        )
        df = bslLD.exb_bracket(f, bslLD.ScalarField(phidata), grid)
        @test maximum(abs, df.data) < 1e-11
    end

    @testset "conservation" begin
        grid = exb_grid()
        fdata, phi, _ = exb_test_fields(grid)
        f = exb_distribution(grid, fdata)
        df = bslLD.exb_bracket(f, phi, grid)
        dvol = prod(grid.delta)

        mass = sum(df.data) * dvol
        @test abs(mass) < 1e-10 * sum(abs, f.data) * dvol

        # d/dt ∫f²/2 dx dv = ∫ f {f, g} = 0 for the incompressible ExB flow
        l2 = sum(f.data .* df.data) * dvol
        @test abs(l2) < 1e-10 * sum(abs2, f.data) * dvol
    end

    @testset "exb_euler! consistency" begin
        grid = exb_grid()
        fdata, phi, _ = exb_test_fields(grid)
        f = exb_distribution(grid, fdata)
        dt = 0.01

        df = bslLD.exb_bracket(f, phi, grid)
        reference = f.data .+ dt .* df.data

        fout = exb_distribution(grid, zero(fdata))
        bslLD.exb_euler!(fout, f, phi, grid, dt)
        @test fout.data ≈ reference

        bslLD.exb_euler!(f, f, phi, grid, dt)   # in place
        @test f.data ≈ reference
    end

    # No phase-space temporaries are allocated per call; the small residual is the
    # fixed per-launch overhead of the four KernelAbstractions kernels inside
    # `_differentiate_impl!` (~350 B each, independent of grid size).
    @testset "no phase-space temporaries" begin
        grid = exb_grid()
        fdata, phi, _ = exb_test_fields(grid)
        f = exb_distribution(grid, fdata)
        df = exb_distribution(grid, zero(fdata))
        plan = bslLD._get_exb_plan(f, grid)

        bslLD.exb_bracket!(df, f, phi, grid, plan)
        @test @allocated(bslLD.exb_bracket!(df, f, phi, grid, plan)) < 2048

        bslLD.exb_euler!(df, f, phi, grid, 0.01, plan)
        @test @allocated(bslLD.exb_euler!(df, f, phi, grid, 0.01, plan)) < 2048
    end

    @testset "guards" begin
        grid1d = exb_grid(nx = 1)
        f1d = bslLD.Distribution(grid1d, 0.0)
        phi1d = bslLD.ScalarField(zeros(length(grid1d.xaxes[1])))
        @test_throws ArgumentError bslLD.exb_bracket(f1d, phi1d, grid1d)

        gridB = exb_grid(Bdir = 1)
        fB = bslLD.Distribution(gridB, 0.0)
        phiB = bslLD.ScalarField(zeros(length.(gridB.xaxes)))
        @test_throws ArgumentError bslLD.exb_bracket(fB, phiB, gridB)

        grid = exb_grid()
        f = bslLD.Distribution(grid, 0.0)
        @test_throws ArgumentError bslLD.exb_bracket(
            f,
            bslLD.ScalarField(zeros(4, 4)),
            grid,
        )
    end

    @testset "batching over x3 (3d2v)" begin
        grid = exb_grid(nx = 3, nxpts = 8, nvpts = 8)
        fdata, phi, expected =
            exb_test_fields(grid; h = (x -> 2.0 + sin(x)), h3phi = (x -> 1.0 + 0.5cos(x)))
        f = exb_distribution(grid, fdata)
        df = bslLD.exb_bracket(f, phi, grid)
        @test maximum(abs, df.data .- expected) < 1e-10
    end
end
