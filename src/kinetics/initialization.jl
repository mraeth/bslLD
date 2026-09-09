function Distribution(
    grid::CartGrid,
    epsilon;
    m = 1.0,
    q = 1.0,
    initFuncx = (x -> (1.0 .+ epsilon * sin(2pi / (grid.xaxes[1][end] + grid.delta[1]) * x))),
    initFuncv = (v -> exp(-v^2 / 2) / sqrt(2 * pi)),
    initFuncv1 = initFuncv,
)
    fct_sp(x) = initFuncx(x)
    fct_v(v) = initFuncv(v)
    dx = [fct_sp.(x) for x in grid.xaxes]
    dv = [fct_v.(x) for x in grid.vaxes]
    if length(dv) == 2
        dv[2] .= initFuncv1.(grid.vaxes[2])
    end
    da = vcat(dx, dv)
    m > 0 || throw(ArgumentError("distribution mass m must be positive"))
    raw = backend_array(outer_product(da))
    sf = ScalarField(raw)
    NX = length(grid.xaxes)
    NV = length(grid.vaxes)
    dist = DistributionGridImpl{NX, NV, Cart, typeof(sf)}(sf)
    return Species(m, q, dist)
end

function Distribution(
    grid::PolarGrid,
    epsilon;
    m = 1.0,
    q = 1.0,
    initFuncx = (x -> (1.0 .+ epsilon * sin(2pi / (grid.xaxes[1][end] + grid.delta[1]) * x))),
    initFuncv = (v -> exp(-v^2 / 2) / sqrt(2 * pi)),
)
    fct_sp(x) = initFuncx(x)
    fct_v(v) = initFuncv(v)
    dx = [fct_sp.(x) for x in grid.xaxes]
    dv = [fct_v.(grid.vaxes[1]), sin.(grid.vaxes[2])]
    da = vcat(dx, dv)
    raw = backend_array(outer_product(da))
    sf = ScalarField(raw)
    NX = length(grid.xaxes)
    NV = length(grid.vaxes)
    dist = DistributionGridImpl{NX, NV, Polar, typeof(sf)}(sf)
    return Species(m, q, dist)
end
