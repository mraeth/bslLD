using Serialization

const ITG_MIRROR = joinpath(@__DIR__, "..", "..", "examples", "itg_mirror.jl")

# Diagnostics live in bslLD.Production; the template example is run once at the end.
const D = bslLD.Production

# Fields every chunk header must carry; the post-processing reader of the `bslLD` format
# tests the same list against a chunk it writes from the documented format.
const CHUNK_HEADER_FIELDS = (:version, :meta, :grid, :time, :kind)

@testset "Production diagnostics" begin
    grid = bslLD.Grid([0.0, -2.0], [2pi, 2.0], [4, 6], 1)
    simTime = bslLD.SimulationTime(0.1, 1.0)
    n = Ref(1.0)
    source() = bslLD.ScalarField(fill(n[], 4, 3))

    @testset "chunk round trip in memory" begin
        d = D.Diagnostic(
            source,
            grid;
            dir = "unused",
            name = "n",
            every = 1,
            buffer = 10,
            meta = Dict{String,Any}("description" => "test"),
        )
        for k = 1:3
            n[] = Float64(k)
            bslLD.advance!(simTime, 0.0)
            D.write!(d, simTime)
        end
        io = IOBuffer()
        D.write_chunk(io, D.chunk_header(d), d.data)
        seekstart(io)
        chunk = D.read_chunk(io)
        @test propertynames(chunk.header) == CHUNK_HEADER_FIELDS
        @test chunk.header.version == D.CHUNK_VERSION == 1
        @test chunk.header.kind == :scalar
        @test chunk.header.meta["description"] == "test"
        @test chunk.header.meta["name"] == "n"
        @test chunk.header.time.step == [1, 2, 3]
        @test chunk.header.time.t ≈ [0.1, 0.2, 0.3]
        @test all(chunk.data[k] == fill(Float64(k), 4, 3) for k = 1:3)
        @test chunk.header.grid.delta == grid.delta

        seekstart(io)
        @test D.read_chunk(io; header_only = true) == (header = chunk.header,)
    end

    @testset "every, buffer, files, atomic write" begin
        mktempdir() do dir
            simTime = bslLD.SimulationTime(0.1, 1.0)
            d = D.Diagnostic(source, grid; dir, name = "n", every = 2, buffer = 2)
            for k = 1:9
                n[] = Float64(k)
                bslLD.advance!(simTime, 0.0)
                D.write!(d, simTime)
            end
            # records at steps 2, 4, 6, 8: two full chunks, nothing pending
            @test sort(readdir(dir)) == ["n_2.jls", "n_6.jls"]
            @test !any(endswith(".tmp"), readdir(dir))
            D.close!(d)                                       # nothing left to flush
            @test sort(readdir(dir)) == ["n_2.jls", "n_6.jls"]
            chunk = open(D.read_chunk, joinpath(dir, "n_6.jls"))
            @test chunk.header.time.step == [6, 8]
            @test chunk.data[2] == fill(8.0, 4, 3)

            n[] = 10.0
            bslLD.advance!(simTime, 0.0)                      # step 10: one pending record
            D.write!(d, simTime)
            D.close!(d)
            @test "n_10.jls" in readdir(dir)
        end
    end

    @testset "vector fields and buffer = 1" begin
        mktempdir() do dir
            simTime = bslLD.SimulationTime(0.1, 1.0)
            vec() =
                bslLD.VectorField([bslLD.ScalarField(fill(Float64(c), 4, 3)) for c = 1:3])
            d = D.Diagnostic(vec, grid; dir, name = "E", every = 1, buffer = 1)
            bslLD.advance!(simTime, 0.0)
            D.write!(d, simTime)
            @test readdir(dir) == ["E_1.jls"]
            chunk = open(D.read_chunk, joinpath(dir, "E_1.jls"))
            @test chunk.header.kind == :vector
            @test chunk.data[1] isa Tuple && length(chunk.data[1]) == 3
            @test chunk.data[1][3] == fill(3.0, 4, 3)
        end
    end

    @testset "git_state" begin
        @test D.git_state(pkgdir(bslLD)).hash isa String
        @test D.git_state(tempdir()).hash == "unknown"
    end
end

@testset "ITG mirror template runs (tiny grid, CPU)" begin
    root = dirname(dirname(@__DIR__))
    mktempdir() do dir
        # Pkg.test restricts JULIA_LOAD_PATH to the test environment; the child needs its own
        env = merge(ENV, Dict("BSLLD_TINY" => "1", "BSLLD_OUTPUT" => dir))
        delete!(env, "JULIA_LOAD_PATH")
        cmd = setenv(`$(Base.julia_cmd()) --project=$root $ITG_MIRROR`, env)
        @test success(pipeline(cmd; stdout = devnull, stderr = stderr))
        files = sort(readdir(dir))
        @test files == ["f_5.jls", "n_2.jls"]
        n = open(D.read_chunk, joinpath(dir, "n_2.jls"))
        @test n.header.time.step == [2, 4]
        @test all(isfinite, sum.(n.data))
        @test size(n.data[1]) == (16, 8, 8)
        f = open(D.read_chunk, joinpath(dir, "f_5.jls"))
        @test length(f.data) == 1 && size(f.data[1]) == (16, 8, 8, 9, 9, 9)
        # the ramp has deposited something, and it is positive on average
        @test sum(n.data[1]) / length(n.data[1]) > 0
    end
end
