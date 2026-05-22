using Test

function gpu_cartesian_grid()
    return bslLD.Grid([0.0, -1.5], [2pi, 1.5], [8, 6], 0.05, 1, 1)
end

gpu_all_axes(grid) = (grid.xaxes..., grid.vaxes...)

function gpu_seed_data(grid)
    dims = Tuple(length.(gpu_all_axes(grid)))
    return Float64[
        0.4 * sin(2pi * (idx[1] - 1) / dims[1]) + 0.2 * cos(2pi * (idx[2] - 1) / dims[2])
        for idx in CartesianIndices(dims)
    ]
end

function gpu_constant_field(grid)
    dims = Tuple(length.(grid.xaxes))
    return [fill(0.15, dims) for _ in 1:length(grid.vaxes)]
end

@testset "GPU Regression" begin
    cuda_loaded = false
    try
        @eval using CUDA
        cuda_loaded = true
    catch
        cuda_loaded = false
    end

    if cuda_loaded && bslLD.cuda_available()
        grid = gpu_cartesian_grid()
        initial = gpu_seed_data(grid)
        field_components = gpu_constant_field(grid)

        bslLD.use_cpu!()
        f_cpu_x = bslLD.Distribution(grid, 0.01)
        f_cpu_x.data .= initial
        bslLD.advectX!(f_cpu_x, grid)
        cpu_x = Array(f_cpu_x.data)

        f_cpu_v = bslLD.Distribution(grid, 0.01)
        f_cpu_v.data .= initial
        e_cpu = bslLD.VectorField(field_components)
        bslLD.advectV!(f_cpu_v, grid, e_cpu)
        cpu_v = Array(f_cpu_v.data)

        bslLD.use_cuda!()
        f_gpu_x = bslLD.Distribution(grid, 0.01)
        f_gpu_x.data .= bslLD.backend_array(initial)
        @test_nowarn bslLD.advectX!(f_gpu_x, grid)
        gpu_x = Array(f_gpu_x.data)

        f_gpu_v = bslLD.Distribution(grid, 0.01)
        f_gpu_v.data .= bslLD.backend_array(initial)
        e_gpu = bslLD.VectorField(field_components)
        @test_nowarn bslLD.advectV!(f_gpu_v, grid, e_gpu)
        gpu_v = Array(f_gpu_v.data)

        @test maximum(abs.(gpu_x .- cpu_x)) < 1e-10
        @test maximum(abs.(gpu_v .- cpu_v)) < 1e-10

        bslLD.use_cpu!()
    else
        @test true
    end
end
