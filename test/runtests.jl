using Test
using bslLD

include("differential_operators_test.jl")
include("maxwell_test.jl")

@testset "bslLD.jl" begin
    @testset "Grid" begin
        # Test 1D1v grid creation
        grid = bslLD.Grid([0.0, -6.0], [10.0, 6.0], [1, 1], 0.05, 100, 1)
        @test grid.dt == 0.05
        @test grid.nsteps == 100
        @test length(grid.xaxes) == 1
        @test length(grid.vaxes) == 1
    end

    @testset "Distribution" begin
        grid = bslLD.Grid([0.0, -6.0], [10.0, 6.0], [1, 1], 0.05, 100, 1)
        f = bslLD.Distribution(grid, 0.01)
        @test size(f.data, 1) == length(grid.xaxes[1])
        @test size(f.data, 2) == length(grid.vaxes[1])
    end

    @testset "Fields" begin
        # Using the proper backend setup
        bslLD.set_execution_space!(exec=bslLD.backend())
        grid = bslLD.Grid([0.0, -6.0], [10.0, 6.0], [1, 1], 0.05, 100, 1)
        f = bslLD.Distribution(grid, 0.01)
        rho = bslLD.compute_density(f, grid)
        @test length(rho.data) == length(grid.xaxes[1])

        scalar = bslLD.ScalarField(reshape(collect(1.0:4.0), 2, 2))
        vector = bslLD.VectorField([collect(1.0:4.0), collect(5.0:8.0)])
        @test scalar.data isa Array
        @test all(component -> component isa bslLD.ScalarField, vector.data)
        @test all(component -> component.data isa Array, vector.data)
    end

    @testset "Manufactured Solution Test" begin
        # Test advection with sinusoidal manufactured solution
        # Create a simple 1D1V grid
        grid = bslLD.Grid([0.0, -2.0], [4.0, 2.0], [16, 8], 0.01, 1, 1)
        f = bslLD.Distribution(grid, 0.01)
        
        # Initialize with a sinusoidal distribution in x-direction
        # u(x,v) = sin(π*x/L) where L is the spatial domain length
        xaxis = grid.xaxes[1]
        L = xaxis[end] - xaxis[1]
        for i in 1:length(xaxis)
            for j in 1:length(grid.vaxes[1])
                f.data[i,j] = sin(π * (xaxis[i] - xaxis[1]) / L)
            end
        end
        
        # Store initial distribution for comparison
        initial_data = copy(f.data)
        
        # Apply spatial advection (X-advection) for one time step
        bslLD.set_execution_space!(exec=bslLD.backend())
        bslLD.advectX!(f, grid)
        
        # Basic validation tests
        @test !all(iszero, f.data)  # Ensure we have meaningful data
        @test all(x -> !isnan(x) && !isinf(x), f.data)  # Ensure no NaN or Inf values
        
        # Test that the total particle number is conserved approximately
        initial_sum = sum(initial_data)
        final_sum = sum(f.data)
        # Allow for small numerical errors in conservation
        @test abs(initial_sum - final_sum) < 1e-10 * abs(initial_sum)
    end
end

    @testset "Distribution" begin
        grid = bslLD.Grid([0.0, -6.0], [10.0, 6.0], [1, 1], 0.05, 100, 1)
        f = bslLD.Distribution(grid, 0.01)
        @test size(f.data, 1) == length(grid.xaxes[1])
        @test size(f.data, 2) == length(grid.vaxes[1])
    end

    @testset "Fields" begin
        bslLD.use_cpu!()
        grid = bslLD.Grid([0.0, -6.0], [10.0, 6.0], [1, 1], 0.05, 100, 1)
        f = bslLD.Distribution(grid, 0.01)
        rho = bslLD.compute_density(f, grid)
        @test length(rho.data) == length(grid.xaxes[1])

        scalar = bslLD.ScalarField(reshape(collect(1.0:4.0), 2, 2))
        vector = bslLD.VectorField([collect(1.0:4.0), collect(5.0:8.0)])
        @test scalar.data isa Array
        @test all(component -> component isa bslLD.ScalarField, vector.data)
        @test all(component -> component.data isa Array, vector.data)
    end

    @testset "Manufactured Solution Test" begin
        # Test advection with sinusoidal manufactured solution
        # Create a simple 1D1V grid
        grid = bslLD.Grid([0.0, -2.0], [4.0, 2.0], [32, 16], 0.01, 1, 1)
        f = bslLD.Distribution(grid, 0.01)
        
        # Initialize with a sinusoidal distribution in x-direction
        # u(x,v) = sin(π*x/L) where L is the spatial domain length
        xaxis = grid.xaxes[1]
        L = xaxis[end] - xaxis[1]
        for i in 1:length(xaxis)
            for j in 1:length(grid.vaxes[1])
                f.data[i,j] = sin(π * (xaxis[i] - xaxis[1]) / L)
            end
        end
        
        # Store initial distribution for comparison
        initial_data = copy(f.data)
        
        # Apply spatial advection (X-advection) for one time step
        bslLD.use_cpu!()
        bslLD.advectX!(f, grid)
        
        # For sinusoidal with wave number π/L, the analytical solution should be:
        # u(x,v,t) = sin(π*(x-v*t)/L) 
        # Here we just verify that the function evolved appropriately 
        # (this is a qualitative test as we don't know exact velocity yet)
        
        # Check that we didn't get all zeros (indicating failure)
        @test !all(iszero, f.data)
        
        # Check that the distribution still has reasonable values (not NaN or Inf)
        @test all(x -> !isnan(x) && !isinf(x), f.data)
        
        # Test that the total particle number is conserved approximately
        initial_sum = sum(initial_data)
        final_sum = sum(f.data)
        # Allow for small numerical errors in conservation
        @test abs(initial_sum - final_sum) < 1e-10 * abs(initial_sum)
    end
end
