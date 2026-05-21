@testset "Index Helpers" begin
    @testset "N-D to 1-D" begin
        @test bslLD.index_nd_to_1d((1, 1), (3, 3)) == 1
        @test bslLD.index_nd_to_1d((2, 1), (3, 3)) == 2
        @test bslLD.index_nd_to_1d((1, 2), (3, 3)) == 4
        @test bslLD.index_nd_to_1d((3, 3), (3, 3)) == 9
        @test bslLD.index_nd_to_1d((1,), (5,)) == 1
        @test bslLD.index_nd_to_1d((1, 1, 1), (2, 3, 4)) == 1
        @test bslLD.index_nd_to_1d((2, 3, 4), (2, 3, 4)) == 24
    end

    @testset "1-D to N-D" begin
        @test bslLD.index_1d_to_nd(1, (3, 3)) == (1, 1)
        @test bslLD.index_1d_to_nd(2, (3, 3)) == (2, 1)
        @test bslLD.index_1d_to_nd(4, (3, 3)) == (1, 2)
        @test bslLD.index_1d_to_nd(9, (3, 3)) == (3, 3)
        @test bslLD.index_1d_to_nd(1, (5,)) == (1,)
        @test bslLD.index_1d_to_nd(1, (2, 3, 4)) == (1, 1, 1)
        @test bslLD.index_1d_to_nd(24, (2, 3, 4)) == (2, 3, 4)
    end

    @testset "Round trips" begin
        sizes = (2, 3, 4)
        for index in 1:prod(sizes)
            nd = bslLD.index_1d_to_nd(index, sizes)
            @test bslLD.index_nd_to_1d(nd, sizes) == index
        end
    end
end
