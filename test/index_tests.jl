using Test

# Include the advectorCart.jl file to access its functions directly
include("../src/advectorCart.jl")

@testset "Index Operations" begin
    # Test 1D to N-D index conversion (2D case)
    @test index_nd_to_1d((1, 1), (3, 3)) == 1
    @test index_nd_to_1d((2, 1), (3, 3)) == 2
    @test index_nd_to_1d((1, 2), (3, 3)) == 4
    @test index_nd_to_1d((3, 3), (3, 3)) == 9
    
    # Test N-D to 1D index conversion (2D case)
    @test index_1d_to_nd(1, (3, 3)) == (1, 1)
    @test index_1d_to_nd(2, (3, 3)) == (2, 1)
    @test index_1d_to_nd(4, (3, 3)) == (1, 2)
    @test index_1d_to_nd(9, (3, 3)) == (3, 3)
    
    # Test edge cases
    @test index_nd_to_1d((1,), (5,)) == 1
    @test index_1d_to_nd(1, (5,)) == (1,)
    
    # Test 3D case
    @test index_nd_to_1d((1, 1, 1), (2, 3, 4)) == 1
    @test index_nd_to_1d((2, 3, 4), (2, 3, 4)) == 24
    @test index_1d_to_nd(1, (2, 3, 4)) == (1, 1, 1)
    @test index_1d_to_nd(24, (2, 3, 4)) == (2, 3, 4)
end