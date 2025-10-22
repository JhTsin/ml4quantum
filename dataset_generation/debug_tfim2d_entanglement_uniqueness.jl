using ITensors, ITensorMPS
using Statistics

include("Hamiltonian.jl")
include("mps_utils.jl")

println("TFIM-2D Entanglement Uniqueness Analysis (5x5)")
println("="^60)

# Julia implementation of compute_uniqueness_ratio function
function compute_uniqueness_ratio(values, atol=1e-4)
    """计算数组中不重复数值占比：unique_count / total_count
    
    使用绝对容差进行浮点比较以避免由于数值误差导致的误判
    """
    arr = collect(values)
    total = length(arr)
    if total == 0
        return 0.0
    end
    
    # 先尝试直接使用unique
    unique_vals = unique(arr)
    
    # 如果unique值太多，说明可能有数值误差，需要容差处理
    if length(unique_vals) > total * 0.1  # 如果unique值超过总数的10%，可能有问题
        # 使用容差去重
        sorted_arr = sort(arr)
        unique_count = 1
        for i in 2:length(sorted_arr)
            if !isapprox(sorted_arr[i], sorted_arr[i-1], atol=atol, rtol=0.0)
                unique_count += 1
            end
        end
    else
        unique_count = length(unique_vals)
    end
    
    return unique_count / total
end

# Helper: calculate entropy uniqueness for all adjacent pairs
function calculate_entropy_uniqueness(psi::MPS, Nx::Int, Ny::Int)
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
    return compute_uniqueness_ratio(vals), mean(vals), vals
end

function run_uniqueness_suite()
    Nx, Ny = 3, 3  # 5x5 circuit
    
    # MODE SELECTION: Uncomment ONE of the following modes
    mode = "size_two"  # Options: "size_two", "cut", "block"
    
    if mode == "size_two"
        println("\nMode: exact_renyi_entropy_size_two uniqueness analysis (adjacent pairs)")
        println("-"^60)
        
        configs = [
            # Test different theta combinations with four parameters for uniqueness
            (name = "θh1 sweep (fixed θj1=θj2=θj3=-π/2, nsteps=5)", 
             sweep = [(θj1 = -π/2, θj2 = -π/2, θj3 = -π/2, θh1 = x, nsteps = 5) for x in (π/6, π/4, π/3, π/2, 2π/3, 3π/4, 5π/6)]),
            
            (name = "θj1 sweep (fixed θj2=θj3=-π/2, θh1=π/2, nsteps=5)", 
             sweep = [(θj1 = x, θj2 = -π/2, θj3 = -π/2, θh1 = π/2, nsteps = 5) for x in (-π, -2π/3, -π/2, -π/3, -π/4, -π/6)]),
            
            (name = "θj2 sweep (fixed θj1=θj3=-π/2, θh1=π/2, nsteps=5)", 
             sweep = [(θj1 = -π/2, θj2 = x, θj3 = -π/2, θh1 = π/2, nsteps = 5) for x in (-π, -2π/3, -π/2, -π/3, -π/4, -π/6)]),
            
            (name = "θj3 sweep (fixed θj1=θj2=-π/2, θh1=π/2, nsteps=5)", 
             sweep = [(θj1 = -π/2, θj2 = -π/2, θj3 = x, θh1 = π/2, nsteps = 5) for x in (-π, -2π/3, -π/2, -π/3, -π/4, -π/6)]),
            
            (name = "nsteps sweep (fixed θj1=θj2=θj3=-π/2, θh1=π/2)", 
             sweep = [(θj1 = -π/2, θj2 = -π/2, θj3 = -π/2, θh1 = π/2, nsteps = s) for s in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)]),
            
            # Test asymmetric configurations for high uniqueness
            (name = "Asymmetric θj1 variations (θj2=θj3=-π/2, θh1=π/2, nsteps=5)", 
             sweep = [(θj1 = x, θj2 = -π/2, θj3 = -π/2, θh1 = π/2, nsteps = 5) for x in (-π, -π/2, -π/4, -π/6, 0.0, π/6, π/4, π/2)]),
            
            (name = "Asymmetric θj2 variations (θj1=θj3=-π/2, θh1=π/2, nsteps=5)", 
             sweep = [(θj1 = -π/2, θj2 = x, θj3 = -π/2, θh1 = π/2, nsteps = 5) for x in (-π, -π/2, -π/4, -π/6, 0.0, π/6, π/4, π/2)]),
            
            (name = "Asymmetric θj3 variations (θj1=θj2=-π/2, θh1=π/2, nsteps=5)", 
             sweep = [(θj1 = -π/2, θj2 = -π/2, θj3 = x, θh1 = π/2, nsteps = 5) for x in (-π, -π/2, -π/4, -π/6, 0.0, π/6, π/4, π/2)]),
            
            # Test extreme asymmetric combinations
            (name = "Extreme asymmetric (θj1=-π, θj2=0.0, θj3=π/2, θh1=π/2, nsteps=5)", 
             sweep = [(θj1 = -π, θj2 = 0.0, θj3 = π/2, θh1 = π/2, nsteps = 5)]),
            
            (name = "Mixed signs (θj1=-π/2, θj2=π/4, θj3=-π/4, θh1=π/2, nsteps=5)", 
             sweep = [(θj1 = -π/2, θj2 = π/4, θj3 = -π/4, θh1 = π/2, nsteps = 5)]),
            
            (name = "Strong contrast (θj1=-π, θj2=-π/6, θj3=-π/3, θh1=π/2, nsteps=5)", 
             sweep = [(θj1 = -π, θj2 = -π/6, θj3 = -π/3, θh1 = π/2, nsteps = 5)]),
            
            # Test different θh1 with asymmetric coupling
            (name = "θh1 sweep with asymmetric coupling (θj1=-π, θj2=-π/4, θj3=-π/2, nsteps=5)", 
             sweep = [(θj1 = -π, θj2 = -π/4, θj3 = -π/2, θh1 = x, nsteps = 5) for x in (π/6, π/4, π/3, π/2, 2π/3, 3π/4, 5π/6)]),
            
            # Test more complex asymmetric patterns
            (name = "Complex pattern 1 (θj1=-π/3, θj2=-2π/3, θj3=-π/6, θh1=π/2, nsteps=5)", 
             sweep = [(θj1 = -π/3, θj2 = -2π/3, θj3 = -π/6, θh1 = π/2, nsteps = 5)]),
            
            (name = "Complex pattern 2 (θj1=-π/4, θj2=-3π/4, θj3=-π/8, θh1=π/2, nsteps=5)", 
             sweep = [(θj1 = -π/4, θj2 = -3π/4, θj3 = -π/8, θh1 = π/2, nsteps = 5)]),
            
            (name = "Complex pattern 3 (θj1=-π/6, θj2=-5π/6, θj3=-π/12, θh1=π/2, nsteps=5)", 
             sweep = [(θj1 = -π/6, θj2 = -5π/6, θj3 = -π/12, θh1 = π/2, nsteps = 5)]),
        ]

        # Store all results for analysis
        all_results = []
        
        for cfg in configs
            println("\n" * cfg.name)
            println("-"^60)
            for p in cfg.sweep
                psi = tfim_2d_quantum_circuit_pro_pauli(Nx, Ny; θj1=p.θj1, θj2=p.θj2, θj3=p.θj3, θh1=p.θh1, nsteps=p.nsteps)
                
                # Calculate entropy uniqueness for all adjacent pairs
                uniqueness, mean_entropy, entropy_values = calculate_entropy_uniqueness(psi, Nx, Ny)
                
                # Store results
                push!(all_results, (
                    θj1 = p.θj1, θj2 = p.θj2, θj3 = p.θj3, θh1 = p.θh1, nsteps = p.nsteps,
                    uniqueness = uniqueness, mean_entropy = mean_entropy, 
                    entropy_values = entropy_values, config_name = cfg.name
                ))
                
                println("θj1=$(round(p.θj1/π,digits=3))π, θj2=$(round(p.θj2/π,digits=3))π, θj3=$(round(p.θj3/π,digits=3))π, θh1=$(round(p.θh1/π,digits=3))π, nsteps=$(p.nsteps)")
                println("  -> uniqueness=$(round(uniqueness,digits=6)), mean=$(round(mean_entropy,digits=6)), values=$(round.(entropy_values,digits=4))")
            end
        end

        # Analyze results for maximum uniqueness
        println("\n" * "="^60)
        println("UNIQUENESS ANALYSIS RESULTS")
        println("="^60)
        
        # Sort by uniqueness
        sort!(all_results, by=x->x.uniqueness, rev=true)
        
        println("\nTop 10 highest uniqueness configurations:")
        println("-"^50)
        for i in 1:min(10, length(all_results))
            result = all_results[i]
            println("$(i). θj1=$(round(result.θj1/π,digits=3))π, θj2=$(round(result.θj2/π,digits=3))π, θj3=$(round(result.θj3/π,digits=3))π, θh1=$(round(result.θh1/π,digits=3))π, nsteps=$(result.nsteps)")
            println("    uniqueness=$(round(result.uniqueness,digits=6)), mean=$(round(result.mean_entropy,digits=6)), values=$(round.(result.entropy_values,digits=4))")
        end
        
        # Statistics
        uniquenesses = [r.uniqueness for r in all_results]
        means = [r.mean_entropy for r in all_results]
        
        println("\nUniqueness Statistics:")
        println("-"^30)
        println("Total configurations tested: $(length(all_results))")
        println("Maximum uniqueness: $(round(maximum(uniquenesses), digits=6))")
        println("Minimum uniqueness: $(round(minimum(uniquenesses), digits=6))")
        println("Mean uniqueness: $(round(mean(uniquenesses), digits=6))")
        println("Std uniqueness: $(round(std(uniquenesses), digits=6))")
        
        println("\nMean Entropy Statistics:")
        println("-"^30)
        println("Maximum mean entropy: $(round(maximum(means), digits=6))")
        println("Minimum mean entropy: $(round(minimum(means), digits=6))")
        println("Mean of means: $(round(mean(means), digits=6))")
        println("Std of means: $(round(std(means), digits=6))")
        
        # Find configurations with high uniqueness
        high_uniqueness_configs = filter(r -> r.uniqueness > 0.5, all_results)
        println("\nConfigurations with uniqueness > 0.5:")
        println("-"^40)
        for (i, result) in enumerate(high_uniqueness_configs)
            println("$(i). θj1=$(round(result.θj1/π,digits=3))π, θj2=$(round(result.θj2/π,digits=3))π, θj3=$(round(result.θj3/π,digits=3))π, θh1=$(round(result.θh1/π,digits=3))π")
            println("    uniqueness=$(round(result.uniqueness,digits=6)), mean=$(round(result.mean_entropy,digits=6))")
        end
        
        # Compare nsteps=5 vs 10 for high uniqueness configurations
        println("\nCompare nsteps=5 vs 10 for high uniqueness configs")
        println("-"^60)
        high_uni_configs = [
            (-π, -π/4, -π/2, π/2),  # Strong contrast
            (-π/2, π/4, -π/4, π/2),  # Mixed signs
            (-π, 0.0, π/2, π/2),       # Extreme asymmetric
            (-π/3, -2π/3, -π/6, π/2), # Complex pattern 1
        ]
        
        for (θj1, θj2, θj3, θh1) in high_uni_configs
            println("\nTesting θj1=$(round(θj1/π,digits=3))π, θj2=$(round(θj2/π,digits=3))π, θj3=$(round(θj3/π,digits=3))π, θh1=$(round(θh1/π,digits=3))π")
            for s in (5, 10)
                psi = tfim_2d_quantum_circuit_pro_pauli(Nx, Ny; θj1=θj1, θj2=θj2, θj3=θj3, θh1=θh1, nsteps=s)
                uniqueness, mean_entropy, entropy_values = calculate_entropy_uniqueness(psi, Nx, Ny)
                println("  nsteps=$(s): uniqueness=$(round(uniqueness,digits=6)), mean=$(round(mean_entropy,digits=6)), values=$(round.(entropy_values,digits=4))")
            end
        end
        
    elseif mode == "cut"
        println("\nMode: exact_renyi_entropy_cut uniqueness analysis (bond cuts)")
        println("-"^60)
        # Similar structure but for bond cuts
        # Implementation would be similar to size_two but using exact_renyi_entropy_cut
        
    elseif mode == "block"
        println("\nMode: exact_renyi_entropy_block uniqueness analysis (blocks)")
        println("-"^60)
        # Similar structure but for blocks
        # Implementation would be similar to size_two but using exact_renyi_entropy_block
    end
end

run_uniqueness_suite()