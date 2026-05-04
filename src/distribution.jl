
abstract type Distribution end

struct DistributionGrid{DT,NX,NV,NXNV,ID}  <: Distribution
    data :: AbstractArray{DT,NXNV}
end

const DistributionGrid1d1v{T} = DistributionGrid{T,1,1,2}
const DistributionGrid1d2v{T} = DistributionGrid{T,1,2,3}
const DistributionGrid2d2v{T} = DistributionGrid{T,2,2,4}



function Distribution(grid::CartGrid, epsilon; 
    initFuncx = (x-> (01. .+ epsilon * sin(2pi/(grid.xaxes[1][end]+grid.delta[1])*x ))),
    initFuncv = (v-> exp(-v^2 / 2) / sqrt(2*pi)), initFuncv1 = initFuncv)
fct_sp(x) = initFuncx(x)
fct_v(v) = initFuncv(v)
dx = map(x->fct_sp.(x), grid.xaxes)
dv = map(x->fct_v.(x), grid.vaxes)
if length(dv)==2
    dv[2].=initFuncv1.(grid.vaxes[2])
end
da = vcat(dx,dv)
return DistributionGrid{Float64,length(grid.xaxes),length(grid.vaxes),length(grid.xaxes)+length(grid.vaxes), Cart }(outer_product(da))
end


function Distribution(grid::PolarGrid, epsilon; 
    initFuncx = (x-> (01. .+ epsilon * sin(2pi/(grid.xaxes[1][end]+grid.delta[1])*x ))),
    initFuncv = (v-> exp(-v^2 / 2) / sqrt(2*pi)))
fct_sp(x) = initFuncx(x)
fct_v(v) = initFuncv(v)
dx = map(x->fct_sp.(x), grid.xaxes)
dv = [fct_v.(grid.vaxes[1]), sin.(grid.vaxes[2])]
da = vcat(dx,dv)
return DistributionGrid{Float64,length(grid.xaxes),length(grid.vaxes),length(grid.xaxes)+length(grid.vaxes), Polar }(outer_product(da))
end

function compute_density(f :: DistributionGrid, grid::CartGrid)
    dim = Tuple(i for i=length(grid.xaxes)+1:length(grid.xaxes)+length(grid.vaxes))
    dv = prod(grid.delta[1+length(grid.xaxes):end])
    return ScalarField(reshape(sum(f.data, dims =dim)*dv, Tuple([length(axes) for axes in grid.xaxes])))
end

function compute_density(f :: DistributionGrid, grid::PolarGrid)
    dim = Tuple(i for i=length(grid.xaxes)+1:length(grid.xaxes)+length(grid.vaxes))
    dv = prod(grid.delta[1+length(grid.xaxes):end])
    return ScalarField(reshape(sum(f.data.* reshape(grid.vaxes[1], 1, :, 1), dims =dim)*dv, Tuple([length(axes) for axes in grid.xaxes])))
end