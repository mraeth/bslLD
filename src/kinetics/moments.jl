function compute_density(
    f::DistributionGrid{DT,NX,NV,NXNV,Cart},
    grid::CartGrid,
) where {DT,NX,NV,NXNV}
    dim = ntuple(i -> NX + i, Val(NV))
    dv = DT(prod(grid.delta[(1+NX):(NX+NV)]))
    return ScalarField(
        reshape(sum(f.data, dims = dim) * dv, ntuple(i -> length(grid.xaxes[i]), Val(NX))),
    )
end

@inline function _moment_setup(
    f::DistributionGrid{DT,NX,NV,NXNV},
    grid::CartGrid,
) where {DT,NX,NV,NXNV}
    vdims = ntuple(i -> NX + i, Val(NV))
    dv = DT(prod(grid.delta[(NX+1):(NX+NV)]))
    xsize = ntuple(i -> length(grid.xaxes[i]), Val(NX))
    return vdims, dv, xsize
end

function _current_arrays(
    f::DistributionGrid{DT,NX,NV,NXNV,Cart},
    grid::CartGrid,
) where {DT,NX,NV,NXNV}
    vdims, dv, xsize = _moment_setup(f, grid)
    return ntuple(
        a -> begin
            v = backend_array(collect(grid.vaxes[a]))
            vshape = ntuple(k -> k == NX + a ? length(v) : 1, Val(NXNV))
            reshape(sum(f.data .* reshape(v, vshape), dims = vdims) * dv, xsize)
        end,
        Val(NV),
    )
end

function compute_current(
    f::DistributionGrid{DT,NX,NV,NXNV,Cart},
    grid::CartGrid,
) where {DT,NX,NV,NXNV}
    return VectorField([collect(J) for J in _current_arrays(f, grid)])
end

function compute_current(sp::Species, grid::CartGrid, phase::Real)
    f = sp.dist
    J = compute_current(f, grid)
    NV = length(J)
    1 <= NV <= 3 || error("This function expects 1 <= NV <= 3.")
    phi = -electric_acceleration_scale(sp) * phase
    Rot = R(grid.Bdir, phi)
    return Rot[1:NV, 1:NV] * J
end

function _momentum_tensor_arrays(
    f::DistributionGrid{DT,NX,NV,NXNV,Cart},
    grid::CartGrid,
) where {DT,NX,NV,NXNV}
    vdims, dv, xsize = _moment_setup(f, grid)
    return [
        begin
            va = backend_array(collect(grid.vaxes[a]))
            vb = backend_array(collect(grid.vaxes[b]))
            vashape = ntuple(k -> k == NX + a ? length(va) : 1, Val(NXNV))
            vbshape = ntuple(k -> k == NX + b ? length(vb) : 1, Val(NXNV))
            reshape(
                sum(f.data .* reshape(va, vashape) .* reshape(vb, vbshape), dims = vdims) *
                dv,
                xsize,
            )
        end for a = 1:NV, b = 1:NV
    ]
end

function compute_momentum_tensor(
    f::DistributionGrid{DT,NX,NV,NXNV,Cart},
    grid::CartGrid,
) where {DT,NX,NV,NXNV}
    return MatrixField(_momentum_tensor_arrays(f, grid))
end

function compute_momentum_tensor(sp::Species, grid::CartGrid, phase::Real)
    f = sp.dist
    Pi = compute_momentum_tensor(f, grid)
    NV = size(Pi, 1)
    1 <= NV <= 3 || error("This function expects 1 <= NV <= 3.")
    phi = -electric_acceleration_scale(sp) * phase
    Rot = R(grid.Bdir, phi)[1:NV, 1:NV]
    return Rot * Pi * Rot'
end

function compute_density(f::DistributionGrid, grid::PolarGrid)
    dim = Tuple(i for i = (length(grid.xaxes)+1):(length(grid.xaxes)+length(grid.vaxes)))
    dv = prod(grid.delta[(1+length(grid.xaxes)):end])
    return ScalarField(
        reshape(
            sum(f.data .* reshape(grid.vaxes[1], 1, :, 1), dims = dim) * dv,
            Tuple([length(axes) for axes in grid.xaxes]),
        ),
    )
end

# Species forwarding — pass through to underlying distribution
compute_density(sp::Species, grid) = compute_density(sp.dist, grid)
compute_current(sp::Species, grid) = compute_current(sp.dist, grid)
compute_momentum_tensor(sp::Species, grid) = compute_momentum_tensor(sp.dist, grid)
