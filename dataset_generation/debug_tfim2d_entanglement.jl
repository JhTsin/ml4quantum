using ITensors, ITensorMPS
using Statistics
using DataFrames
using CSV
using Dates

include("Hamiltonian.jl")
include("mps_utils.jl")

println("Detailed TFIM-2D Entropy Analysis")
println("="^60)

# Test parameters
Nx, Ny = 2, 2
nsteps = 5  # Fixed circuit depth

# Define parameter ranges for systematic testing
θj_range = [-π, -2π/3, -π/2, -π/3, -π/4, -π/6]
θh_range = [π/6, π/4, π/3, π/2, 2π/3, 3π/4, π]

# Function to calculate average pair entropy
function calculate_avg_pair_entropy(psi::MPS, Nx::Int, Ny::Int)
    vals = Float64[]
    # Only horizontal pairs (adjacent in the MPS chain)
    for row in 0:(Ny-1)
        for col in 0:(Nx-2)
            a = row * Nx + col + 1
            b = a + 1
            ent = exact_renyi_entropy_size_two(copy(psi), a, b)
            push!(vals, ent)
        end
    end
    return mean(vals)
end

# Store results
results = []

println("Testing all combinations of four theta parameters...")
println("θj1, θj2, θj3 ∈ $θj_range")
println("θh1 ∈ $θh_range")
println("nsteps = $nsteps")
println()

# Test 1: All θj parameters equal, vary θh1
println("1. Testing with θj1=θj2=θj3 (equal coupling)")
println("-"^50)
for θj in θj_range
    for θh in θh_range
        try
            psi, sites = tfim_2d_quantum_circuit_pro(Nx, Ny; θj1=θj, θj2=θj, θj3=θj, θh1=θh, nsteps=nsteps)
            avg_entropy = calculate_avg_pair_entropy(psi, Nx, Ny)
            
            push!(results, (
                θj1=θj, θj2=θj, θj3=θj, θh1=θh, nsteps=nsteps,
                avg_entropy=avg_entropy,
                config_type="equal_coupling"
            ))
            
            println("θj1=θj2=θj3=$(round(θj/π,digits=3))π, θh1=$(round(θh/π,digits=3))π -> entropy=$(round(avg_entropy,digits=4))")
        catch e
            println("Error with θj1=θj2=θj3=$(round(θj/π,digits=3))π, θh1=$(round(θh/π,digits=3))π: $e")
        end
    end
end

println("\n2. Testing with θj1≠θj2≠θj3 (asymmetric coupling)")
println("-"^50)

# Test 2: Asymmetric coupling - sample some interesting combinations
asymmetric_configs = [
    # Strong-weak-strong pattern
    (-π, -π/4, -π, π/2),
    (-π, -π/2, -π, π/2),
    (-π/2, -π, -π/2, π/2),
    
    # Weak-strong-weak pattern  
    (-π/4, -π, -π/4, π/2),
    (-π/6, -π, -π/6, π/2),
    
    # Mixed patterns
    (-π, -π/2, -π/4, π/2),
    (-π/2, -π, -π/4, π/2),
    (-π/4, -π/2, -π, π/2),
    
    # Test with different θh1 values
    (-π, -π/2, -π/4, π/4),
    (-π, -π/2, -π/4, 3π/4),
    (-π/2, -π, -π/4, π/3),
    (-π/2, -π, -π/4, 2π/3),
]

for (θj1, θj2, θj3, θh1) in asymmetric_configs
    try
        psi, sites = tfim_2d_quantum_circuit_pro(Nx, Ny; θj1=θj1, θj2=θj2, θj3=θj3, θh1=θh1, nsteps=nsteps)
        avg_entropy = calculate_avg_pair_entropy(psi, Nx, Ny)
        
        push!(results, (
            θj1=θj1, θj2=θj2, θj3=θj3, θh1=θh1, nsteps=nsteps,
            avg_entropy=avg_entropy,
            config_type="asymmetric_coupling"
        ))
        
        println("θj1=$(round(θj1/π,digits=3))π, θj2=$(round(θj2/π,digits=3))π, θj3=$(round(θj3/π,digits=3))π, θh1=$(round(θh1/π,digits=3))π -> entropy=$(round(avg_entropy,digits=4))")
    catch e
        println("Error with θj1=$(round(θj1/π,digits=3))π, θj2=$(round(θj2/π,digits=3))π, θj3=$(round(θj3/π,digits=3))π, θh1=$(round(θh1/π,digits=3))π: $e")
    end
end

println("\n3. Testing extreme cases")
println("-"^50)

# Test 3: Extreme cases
extreme_configs = [
    # Very strong coupling
    (-π, -π, -π, π/2),
    (-π, -π, -π, π/4),
    (-π, -π, -π, 3π/4),
    
    # Very weak coupling
    (-π/6, -π/6, -π/6, π/2),
    (-π/6, -π/6, -π/6, π/4),
    (-π/6, -π/6, -π/6, 3π/4),
    
    # Mixed extreme
    (-π, -π/6, -π, π/2),
    (-π/6, -π, -π/6, π/2),
    (-π, -π/6, -π/6, π/2),
]

for (θj1, θj2, θj3, θh1) in extreme_configs
    try
        psi, sites = tfim_2d_quantum_circuit_pro(Nx, Ny; θj1=θj1, θj2=θj2, θj3=θj3, θh1=θh1, nsteps=nsteps)
        avg_entropy = calculate_avg_pair_entropy(psi, Nx, Ny)
        
        push!(results, (
            θj1=θj1, θj2=θj2, θj3=θj3, θh1=θh1, nsteps=nsteps,
            avg_entropy=avg_entropy,
            config_type="extreme_cases"
        ))
        
        println("θj1=$(round(θj1/π,digits=3))π, θj2=$(round(θj2/π,digits=3))π, θj3=$(round(θj3/π,digits=3))π, θh1=$(round(θh1/π,digits=3))π -> entropy=$(round(avg_entropy,digits=4))")
    catch e
        println("Error with θj1=$(round(θj1/π,digits=3))π, θj2=$(round(θj2/π,digits=3))π, θj3=$(round(θj3/π,digits=3))π, θh1=$(round(θh1/π,digits=3))π: $e")
    end
end

# Analyze results
println("\n" * "="^60)
println("ANALYSIS RESULTS")
println("="^60)

if !isempty(results)
    # Convert to DataFrame for analysis
    df = DataFrame(results)
    
    # Sort by entropy
    sort!(df, :avg_entropy, rev=true)
    
    println("\nTop 10 highest entropy configurations:")
    println("-"^50)
    for i in 1:min(10, nrow(df))
        row = df[i,:]
        println("$(i). θj1=$(round(row.θj1/π,digits=3))π, θj2=$(round(row.θj2/π,digits=3))π, θj3=$(round(row.θj3/π,digits=3))π, θh1=$(round(row.θh1/π,digits=3))π -> entropy=$(round(row.avg_entropy,digits=4))")
    end
    
    println("\nStatistics:")
    println("-"^30)
    println("Total configurations tested: $(nrow(df))")
    println("Maximum entropy: $(round(maximum(df.avg_entropy), digits=4))")
    println("Minimum entropy: $(round(minimum(df.avg_entropy), digits=4))")
    println("Mean entropy: $(round(mean(df.avg_entropy), digits=4))")
    println("Std entropy: $(round(std(df.avg_entropy), digits=4))")
    
    # Group by configuration type
    println("\nBy configuration type:")
    println("-"^30)
    for config_type in unique(df.config_type)
        subset_df = df[df.config_type .== config_type, :]
        println("$config_type: $(nrow(subset_df)) configs, max entropy = $(round(maximum(subset_df.avg_entropy), digits=4))")
    end
    
    # Save results
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    filename = "entropy_analysis_results_$(timestamp).csv"
    CSV.write(filename, df)
    println("\nResults saved to: $filename")
    
else
    println("No successful results obtained!")
end

println("\nAnalysis completed!")