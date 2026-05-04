module bslLD

using AbstractFFTs, FFTW, KernelAbstractions
using Plots, Dierckx, Base.Threads, StaticArrays

include("grid.jl")
include("distribution.jl")
include("fields.jl")
include("advector.jl")
include("plot.jl")

greet() = print("Hello World!")

end # module bslLD
