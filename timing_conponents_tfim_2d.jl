using ITensors, ITensorMPS
using DataFrames
using CSV
using ArgParse
using PastaQ
using Statistics
using Dates
include("dataset_generation/mps_utils.jl")
include("dataset_generation/utils.jl")
include("dataset_generation/Hamiltonian.jl")
include("dataset_generation/shadow.jl")

# Benchmark parameters
samples_num = 4
shots = 512
Nx = 3
Ny = 3

qubits = Nx * Ny
N_bonds = 2 * Nx * Ny - Nx - Ny

println("Starting benchmark for TFIM 2D timing analysis...")
println("Parameters: samples=$samples_num, shots=$shots, lattice=($Nx x $Ny)")

# Initialize timing storage
timing_results = DataFrame(
    sample = Int[],
    psi_generation_time = Float64[],
    shadow_generation_time = Float64[],
    approx_correlation_time = Float64[],
    exact_correlation_time = Float64[],
    approx_entropy_time = Float64[],
    exact_entropy_time = Float64[],
    total_time = Float64[]
)

total_start_time = now()

for i = 1:samples_num
    sample_start_time = now()
    nstep = rand(2:9)
    # Step 1: Generate quantum state (psi)
    psi_start = now()
    if i == 1
        psi, sites = tfim_2d_quantum_circuit_pro(Nx, Ny; θj1=-π/2,θj2=-π/2,θj3=-π/2, θh1=π/2, nsteps=nstep)
        elseif i == 2
        psi, sites = tfim_2d_quantum_circuit_pro(Nx, Ny; θj1=-0.435,θj2=-π/2,θj3=-π/2, θh1=π/4, nsteps=nstep)
        elseif i == 3
        psi, sites = tfim_2d_quantum_circuit_pro(Nx, Ny; θj1=-π/2,θj2=-0.432,θj3=-π/2, θh1=π/4, nsteps=nstep)
        else
        psi, sites = tfim_2d_quantum_circuit_pro(Nx, Ny; θj1=-π/2,θj2=-π/4,θj3=-0.523, θh1=π/2, nsteps=nstep)
    end
    psi_time = (now() - psi_start).value / 1000.0  # Convert to seconds
    
    # Step 2: Generate shadow measurements
    shadow_start = now()
    obs = getsamples(psi, randombases(qubits, shots, local_basis=["X", "Y", "Z"]))
    shadow_time = (now() - shadow_start).value / 1000.0
    
    # Step 3: Calculate approximate correlation from shadow
    approx_corr_start = now()
    meas_samples, approx_correlation = cal_shadow_correlation(obs)
    approx_corr_time = (now() - approx_corr_start).value / 1000.0
    
    # Step 4: Calculate exact correlation
    exact_corr_start = now()
    exact_correlation = (compute_correlation_paulix_norm(psi) .+ compute_correlation_pauliy_norm(psi) .+ compute_correlation_pauliz_norm(psi)) ./ 3.0
    exact_corr_time = (now() - exact_corr_start).value / 1000.0
    
    # Step 5: Calculate approximate entropy
    approx_entropy_start = now()
    approx_entropy = []
    for row in 0:(Ny-1)
        for col in 0:(Nx-2)
            site_a = row * Nx + col + 1
            site_b = site_a + 1
            push!(approx_entropy, cal_shadow_renyi_entropy(obs, [site_a, site_b]))
        end
    end
    approx_entropy_time = (now() - approx_entropy_start).value / 1000.0
    
    # Step 6: Calculate exact entropy
    exact_entropy_start = now()
    exact_entropy = []
    for row in 0:(Ny-1)
        for col in 0:(Nx-2)
            site_a = row * Nx + col + 1
            site_b = site_a + 1
            push!(exact_entropy, exact_renyi_entropy_size_two(psi, site_a, site_b))
        end
    end
    exact_entropy_time = (now() - exact_entropy_start).value / 1000.0
    
    sample_total_time = (now() - sample_start_time).value / 1000.0
    
    # Store results
    push!(timing_results, (
        i,
        psi_time,
        shadow_time,
        approx_corr_time,
        exact_corr_time,
        approx_entropy_time,
        exact_entropy_time,
        sample_total_time
    ))
    
    println("Sample $i completed in $(round(sample_total_time, digits=3))s")
end

total_time = (now() - total_start_time).value / 1000.0

# Calculate statistics
println("\n=== TIMING BENCHMARK RESULTS ===")
println("Total execution time: $(round(total_time, digits=3))s")
println("Average time per sample: $(round(total_time/samples_num, digits=3))s")
println()

# Calculate averages
avg_psi_time = mean(timing_results.psi_generation_time)
avg_shadow_time = mean(timing_results.shadow_generation_time)
avg_approx_corr_time = mean(timing_results.approx_correlation_time)
avg_exact_corr_time = mean(timing_results.exact_correlation_time)
avg_approx_entropy_time = mean(timing_results.approx_entropy_time)
avg_exact_entropy_time = mean(timing_results.exact_entropy_time)

println("Average timing breakdown (seconds):")
println("  Quantum state generation: $(round(avg_psi_time, digits=4))")
println("  Shadow measurement generation: $(round(avg_shadow_time, digits=4))")
println("  Approximate correlation calculation: $(round(avg_approx_corr_time, digits=4))")
println("  Exact correlation calculation: $(round(avg_exact_corr_time, digits=4))")
println("  Approximate entropy calculation: $(round(avg_approx_entropy_time, digits=4))")
println("  Exact entropy calculation: $(round(avg_exact_entropy_time, digits=4))")

println("\nKey comparisons:")
println("  Correlation: approx vs exact = $(round(avg_approx_corr_time, digits=4))s vs $(round(avg_exact_corr_time, digits=4))s (ratio: $(round(avg_approx_corr_time/avg_exact_corr_time, digits=2)))")
println("  Entropy: approx vs exact = $(round(avg_approx_entropy_time, digits=4))s vs $(round(avg_exact_entropy_time, digits=4))s (ratio: $(round(avg_approx_entropy_time/avg_exact_entropy_time, digits=2)))")

# Save detailed results
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
filename = "benchmark_timing_results_$(samples_num)samples_$(shots)shots_$(Nx)x$(Ny)_$timestamp.csv"

# Create summary statistics
summary_stats = DataFrame(
    metric = ["psi_generation", "shadow_generation", "approx_correlation", "exact_correlation", "approx_entropy", "exact_entropy"],
    average_time_seconds = [avg_psi_time, avg_shadow_time, avg_approx_corr_time, avg_exact_corr_time, avg_approx_entropy_time, avg_exact_entropy_time],
    std_time_seconds = [std(timing_results.psi_generation_time), std(timing_results.shadow_generation_time), std(timing_results.approx_correlation_time), std(timing_results.exact_correlation_time), std(timing_results.approx_entropy_time), std(timing_results.exact_entropy_time)],
    min_time_seconds = [minimum(timing_results.psi_generation_time), minimum(timing_results.shadow_generation_time), minimum(timing_results.approx_correlation_time), minimum(timing_results.exact_correlation_time), minimum(timing_results.approx_entropy_time), minimum(timing_results.exact_entropy_time)],
    max_time_seconds = [maximum(timing_results.psi_generation_time), maximum(timing_results.shadow_generation_time), maximum(timing_results.approx_correlation_time), maximum(timing_results.exact_correlation_time), maximum(timing_results.approx_entropy_time), maximum(timing_results.exact_entropy_time)]
)

CSV.write(filename, timing_results)
CSV.write("summary_$filename", summary_stats)

println("\nResults saved to:")
println("  Detailed results: $filename")
println("  Summary statistics: summary_$filename")

println("\nBenchmark completed successfully!")
