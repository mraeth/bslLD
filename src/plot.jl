function heatmap_fv(data::Array{DT,2}, grid::PolarGrid) where DT
    return heatmap(grid.vaxes[2],grid.vaxes[1],data[:,:]; projection = :polar)
end


function heatmap_fv(data::Array{DT,2}, grid::CartGrid) where DT
    return heatmap(grid.vaxes[2],grid.vaxes[1],data[:,:]', aspect_ratio = :equal, xlims = (minimum(grid.vaxes[2]), maximum(grid.vaxes[2])), ylims = (minimum(grid.vaxes[1]), maximum(grid.vaxes[1])))
end