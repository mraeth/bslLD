# Writing diagnostics as chunk files. One `Diagnostic` per field to write. The field comes from a
# function, is copied to the host every `every` steps and written as one chunk file
# `<dir>/<name>_<firststep>.jls` per `buffer` records. A chunk is two serialized objects: a header
# (version, meta, grid, per-record times) and the data (one host array per record, a tuple of arrays
# for vector fields). `buffer = 1` is how the distribution function is written. Up to `buffer`
# records are lost if a run is interrupted. BSL6DPostProcessing reads this format
# (`BSL6DPostProcessing.bslLD()`); change `CHUNK_VERSION` and the header field list together with it.
const CHUNK_VERSION = 1

mutable struct Diagnostic
    source::Any             # () -> ScalarField or VectorField
    dir::String
    name::String
    every::Int
    buffer::Int
    meta::Dict{String,Any}
    grid::Any
    steps::Vector{Int}
    times::Vector{Float64}
    phases::Vector{Float64}
    data::Vector{Any}
end

function Diagnostic(source, grid; dir, name, every, buffer, meta = Dict{String,Any}())
    return Diagnostic(
        source,
        dir,
        name,
        every,
        buffer,
        meta,
        grid,
        Int[],
        Float64[],
        Float64[],
        [],
    )
end

host_copy(field::bslLD.ScalarField) = Array(field.data)
host_copy(field::bslLD.VectorField) = ntuple(i -> Array(field[i].data), length(field))

function git_state(dir)
    try
        git(args...) = readchomp(pipeline(`git -C $dir $args`; stderr = devnull))
        dirty = !isempty(git("status", "--porcelain", "--untracked-files=no"))
        return (hash = String(git("rev-parse", "--short", "HEAD")), dirty = dirty)
    catch
        return (hash = "unknown", dirty = missing)
    end
end

function chunk_header(d::Diagnostic)
    code = Dict(
        "script" => git_state(dirname(abspath(PROGRAM_FILE))),
        "bslLD" => git_state(pkgdir(bslLD)),
        "PlasmaCore" => git_state(pkgdir(PlasmaCore)),
    )
    meta = merge(d.meta, Dict{String,Any}("name" => d.name, "code" => code))
    return (
        version = CHUNK_VERSION,
        meta = meta,
        grid = d.grid,
        time = (step = d.steps, t = d.times, phase = d.phases),
        kind = first(d.data) isa Tuple ? :vector : :scalar,
    )
end

write_chunk(io::IO, header, data) = (serialize(io, header); serialize(io, data); nothing)

function read_chunk(io::IO; header_only = false)
    header = deserialize(io)
    return header_only ? (header = header,) : (header = header, data = deserialize(io))
end

function flush!(d::Diagnostic)
    isempty(d.data) && return nothing
    mkpath(d.dir)
    path = joinpath(d.dir, "$(d.name)_$(first(d.steps)).jls")
    open(io -> write_chunk(io, chunk_header(d), d.data), path * ".tmp", "w")
    mv(path * ".tmp", path; force = true)     # atomic: a chunk file is always complete
    empty!(d.steps)
    empty!(d.times)
    empty!(d.phases)
    empty!(d.data)
    return path
end

function write!(d::Diagnostic, simTime)
    simTime.step % d.every == 0 || return nothing
    push!(d.data, host_copy(d.source()))
    push!(d.steps, simTime.step)
    push!(d.times, simTime.current_T)
    push!(d.phases, simTime.phase)
    length(d.data) >= d.buffer && flush!(d)
    return nothing
end

close!(d::Diagnostic) = flush!(d)
