using Test
using bslLD

@testset "Differentiate" begin
    bslLD.set_execution_space!(exec=bslLD.backend())

    grid1 = bslLD.Grid([0.0], [2pi], [32], 0.1, 1, 1)
    x1 = grid1.xaxes[1]
    f1 = bslLD.ScalarField(sin.(x1))
    df1 = bslLD.differentiate(f1, grid1, 1)
    @test maximum(abs.(df1.data .- cos.(x1))) < 1e-10

    grid2 = bslLD.Grid([0.0, 0.0], [2pi, 2pi], [24, 20], 0.1, 1, 2)
    x2 = grid2.xaxes[1]
    y2 = grid2.xaxes[2]
    data2 = [sin(x) + cos(2y) for x in x2, y in y2]
    f2 = bslLD.ScalarField(data2)
    dfdx2 = bslLD.differentiate(f2, grid2, 1)
    dfdy2 = bslLD.differentiate(f2, grid2, 2)
    @test maximum(abs.(dfdx2.data .- [cos(x) for x in x2, y in y2])) < 1e-10
    @test maximum(abs.(dfdy2.data .- [-2sin(2y) for x in x2, y in y2])) < 1e-10

    grid3 = bslLD.Grid([0.0, 0.0, 0.0], [2pi, 2pi, 2pi], [16, 12, 10], 0.1, 1, 3)
    x3 = grid3.xaxes[1]
    y3 = grid3.xaxes[2]
    z3 = grid3.xaxes[3]
    data3 = [sin(x) + cos(y) + sin(3z) for x in x3, y in y3, z in z3]
    f3 = bslLD.ScalarField(data3)
    dfdz3 = bslLD.differentiate(f3, grid3, 3)
    @test maximum(abs.(dfdz3.data .- [3cos(3z) for x in x3, y in y3, z in z3])) < 1e-9
end

@testset "Grad Div Curl" begin
    bslLD.set_execution_space!(exec=bslLD.backend())

    grid1 = bslLD.Grid([0.0], [2pi], [32], 0.1, 1, 1)
    x1 = grid1.xaxes[1]
    phi1 = bslLD.ScalarField(sin.(x1))
    grad1 = bslLD.grad(phi1, grid1)
    @test length(grad1) == 1
    @test maximum(abs.(grad1[1].data .- cos.(x1))) < 1e-10

    vec1_2 = bslLD.VectorField([
        sin.(x1),
        cos.(x1),
    ])
    div1_2 = bslLD.div(vec1_2, grid1)
    curl1_2 = bslLD.curl(vec1_2, grid1)
    @test maximum(abs.(div1_2.data .- cos.(x1))) < 1e-10
    @test length(curl1_2) == 2
    @test maximum(abs.(curl1_2[1].data .- sin.(x1))) < 1e-10
    @test maximum(abs.(curl1_2[2].data .- cos.(x1))) < 1e-10

    vec1_3 = bslLD.VectorField([
        zeros(length(x1)),
        sin.(x1),
        cos.(x1),
    ])
    div1_3 = bslLD.div(vec1_3, grid1)
    curl1_3 = bslLD.curl(vec1_3, grid1)
    @test maximum(abs.(div1_3.data)) < 1e-10
    @test length(curl1_3) == 3
    @test maximum(abs.(curl1_3[1].data)) < 1e-10
    @test maximum(abs.(curl1_3[2].data .- sin.(x1))) < 1e-10
    @test maximum(abs.(curl1_3[3].data .- cos.(x1))) < 1e-10

    grid2 = bslLD.Grid([0.0, 0.0], [2pi, 2pi], [24, 20], 0.1, 1, 2)
    x2 = grid2.xaxes[1]
    y2 = grid2.xaxes[2]

    phi2 = bslLD.ScalarField([sin(x) + cos(2y) for x in x2, y in y2])
    grad2 = bslLD.grad(phi2, grid2)
    @test length(grad2) == 2
    @test maximum(abs.(grad2[1].data .- [cos(x) for x in x2, y in y2])) < 1e-10
    @test maximum(abs.(grad2[2].data .- [-2sin(2y) for x in x2, y in y2])) < 1e-10

    vec2 = bslLD.VectorField([
        [sin(x) for x in x2, y in y2],
        [cos(2y) for x in x2, y in y2],
    ])
    div2 = bslLD.div(vec2, grid2)
    curl2 = bslLD.curl(vec2, grid2)
    @test maximum(abs.(div2.data .- [cos(x) - 2sin(2y) for x in x2, y in y2])) < 1e-10
    @test maximum(abs.(curl2.data)) < 1e-10

    vec2_3 = bslLD.VectorField([
        [sin(x) for x in x2, y in y2],
        [cos(2y) for x in x2, y in y2],
        [sin(x + y) for x in x2, y in y2],
    ])
    div2_3 = bslLD.div(vec2_3, grid2)
    curl2_3 = bslLD.curl(vec2_3, grid2)
    @test maximum(abs.(div2_3.data .- [cos(x) - 2sin(2y) for x in x2, y in y2])) < 1e-10
    @test length(curl2_3) == 3
    @test maximum(abs.(curl2_3[1].data .- [cos(x + y) for x in x2, y in y2])) < 1e-10
    @test maximum(abs.(curl2_3[2].data .- [-cos(x + y) for x in x2, y in y2])) < 1e-10
    @test maximum(abs.(curl2_3[3].data)) < 1e-10

    grid3 = bslLD.Grid([0.0, 0.0, 0.0], [2pi, 2pi, 2pi], [16, 12, 10], 0.1, 1, 3)
    x3 = grid3.xaxes[1]
    y3 = grid3.xaxes[2]
    z3 = grid3.xaxes[3]

    vec3 = bslLD.VectorField([
        [sin(y) for x in x3, y in y3, z in z3],
        [sin(z) for x in x3, y in y3, z in z3],
        [sin(x) for x in x3, y in y3, z in z3],
    ])
    div3 = bslLD.div(vec3, grid3)
    curl3 = bslLD.curl(vec3, grid3)
    @test maximum(abs.(div3.data)) < 1e-10
    @test length(curl3) == 3
    @test maximum(abs.(curl3[1].data .- [-cos(z) for x in x3, y in y3, z in z3])) < 1e-10
    @test maximum(abs.(curl3[2].data .- [-cos(x) for x in x3, y in y3, z in z3])) < 1e-10
    @test maximum(abs.(curl3[3].data .- [-cos(y) for x in x3, y in y3, z in z3])) < 1e-10
end
