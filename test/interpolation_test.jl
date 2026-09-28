using Test, bslLD, StaticArrays

"Textbook O(W^2) Lagrange weights, as an independent reference."
function naive_weights(W, alpha)
    off = -W÷2:W÷2-1
    [prod((-alpha - k) / (l - k) for k in off if k != l) for l in off]
end

@testset "Lagrange interpolation" begin
    widths = (2, 4, 6, 8, 12, 16, 24)

    @testset "partition of unity" begin
        for W in widths, alpha in range(0, 1, length = 11)
            w = bslLD.lagrange_weights(Val(W), alpha)
            @test sum(w) ≈ 1 atol = 1e-12
        end
    end

    @testset "matches the naive formula" begin
        for W in widths, alpha in (0.1, 0.3, 0.5, 0.75, 0.99)
            w = bslLD.lagrange_weights(Val(W), alpha)
            @test collect(w) ≈ naive_weights(W, alpha) rtol = 1e-10
        end
    end

    @testset "nodes give delta stencils" begin
        for W in widths
            w0 = bslLD.lagrange_weights(Val(W), 0.0)
            w1 = bslLD.lagrange_weights(Val(W), 1.0)
            @test w0 ≈ SVector{W}(ntuple(m -> m == W ÷ 2 + 1 ? 1.0 : 0.0, W))
            @test w1 ≈ SVector{W}(ntuple(m -> m == W ÷ 2 ? 1.0 : 0.0, W))
        end
    end

    @testset "exact on polynomials of degree < W" begin
        for W in (4, 8, 12), alpha in (0.2, 0.6)
            off = collect(-W÷2:W÷2-1)
            w = bslLD.lagrange_weights(Val(W), alpha)
            for d = 0:W-1
                @test sum(w[m] * float(off[m])^d for m = 1:W) ≈ (-alpha)^d atol = 1e-8
            end
        end
    end

    @testset "periodic shift converges at order W" begin
        W = 6
        f(x) = exp(sin(x))
        errs = map((32, 64, 128)) do n
            xs = range(0, 2pi, length = n + 1)[1:n]
            data = f.(xs)
            s = 0.37                                   # cells
            cells, alpha = bslLD.shift_split(s)
            w = bslLD.lagrange_weights(Val(W), alpha)
            maximum(1:n) do j
                base = bslLD.lagrange_base_index(j - cells, Val(W))
                got = bslLD.lagrange_gather_periodic(data, base, w)
                abs(got - f(xs[j] - s * step(xs)))
            end
        end
        rates = log2.(errs[1:end-1] ./ errs[2:end])
        @test all(rates .> W - 1)                      # ~ h^W
    end

    @testset "gather and weights are allocation-free" begin
        w = bslLD.lagrange_weights(Val(8), 0.3)
        src = collect(1.0:32.0)
        @test (@allocated bslLD.lagrange_weights(Val(8), 0.3)) == 0
        @test (@allocated bslLD.lagrange_gather(src, 4, w)) == 0
        @test (@allocated bslLD.lagrange_gather_periodic(src, 4, w)) == 0
        @test bslLD.lagrange_weights(Val(8), 0.3, Float32) isa SVector{8,Float32}
    end

    @testset "no amplification for a shifted mode" begin
        n, W = 128, 8
        alpha = 0.3
        w = bslLD.lagrange_weights(Val(W), alpha)
        for kh in (0.25pi, 0.5pi, 0.75pi)
            k = kh * n / 2pi
            xs = range(0, 2pi, length = n + 1)[1:n]
            c = cos.(k .* xs)
            s = sin.(k .* xs)
            j = n ÷ 4 + 1
            base = bslLD.lagrange_base_index(j, Val(W))
            g = (bslLD.lagrange_gather_periodic(c, base, w) +
                 im * bslLD.lagrange_gather_periodic(s, base, w)) / cis(k * xs[j])
            @test abs(g) <= 1 + 1e-12
        end
    end
end
