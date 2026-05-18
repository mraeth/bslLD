using Test
using bslLD

# Simplified manufactured solution test for advection functions
@testset "Manufactured Solution Advection Test" begin
    # Create a simple 1D1V grid for testing
    grid = bslLD.Grid([0.0, -2.0], [4.0, 2.0], [16, 8], 0.01, 1, 1)
    
    # Test X-advection with known sinusoidal solution
    # Create a distribution with known analytical behavior
    f = bslLD.Distribution(grid, 0.01)
    
    # Initialize with a simple sinusoid in x-direction
    xaxis = grid.xaxes[1]
    vaxis = grid.vaxes[1]
    L = xaxis[end] - xaxis[1]  # Spatial domain length
    
    # Initialize with sin(π*x/L) profile 
    for i in 1:length(xaxis)
        for j in 1:length(vaxis)
            f.data[i,j] = sin(π * (xaxis[i] - xaxis[1]) / L)
        end
    end
    
    # Store initial data for comparison
    initial_data = copy(f.data)
    
    # Apply X-advection for one time step
    # Using the backend directly instead of use_cpu!
    bslLD.set_execution_space!(exec=bslLD.backend())
    bslLD.advectX!(f, grid)
    
    # Basic validation tests
    @test !all(iszero, f.data)  # Ensure we have meaningful data
    @test all(x -> !isnan(x) && !isinf(x), f.data)  # Ensure no NaN or Inf values
    
    # Test conservation of total particles (approximately)
    initial_sum = sum(initial_data)
    final_sum = sum(f.data)
    @test abs(initial_sum - final_sum) < 1e-10 * abs(initial_sum)
    
    # Test V-advection with simple case
    # Create a new distribution with simple pattern
    f2 = bslLD.Distribution(grid, 0.01)
    
    # Initialize with constant in x, linear in v
    for i in 1:length(xaxis)
        for j in 1:length(vaxis)
            f2.data[i,j] = 1.0 + 0.1 * vaxis[j]  # Linear in velocity
        end
    end
    
    initial_data2 = copy(f2.data)
    
    # Create a simple electric field for V-advection
    e_field = bslLD.VectorField([bslLD.ScalarField(zeros(length(xaxis)))])
    
    # Apply V-advection
    bslLD.advectV!(f2, grid, e_field)
    
    # Basic validation for V-advection
    @test !all(iszero, f2.data)
    @test all(x -> !isnan(x) && !isinf(x), f2.data)
    
    # Test conservation for V-advection
    initial_sum2 = sum(initial_data2)
    final_sum2 = sum(f2.data)
    @test abs(initial_sum2 - final_sum2) < 1e-10 * abs(initial_sum2)
end