using ITensors, ITensorMPS
using DataFrames
using CSV
using ArgParse
using PastaQ
using Statistics
include("mps_utils.jl")
include("utils.jl")
include("Hamiltonian.jl")
include("shadow.jl")


# Model parameters
spin = "1/2"
nsweeps = 10
maxdim = [20, 60, 100, 150, 200]
cutoff = 1E-6

# Command-line argument parsing
s = ArgParseSettings()
@add_arg_table! s begin
    "--samples_num", "-n"
    help = "number of samples to generate"
    required = true
    default = 100
    arg_type = Int

    "--shots", "-s"
    help = "number of measurement shots for classical shadow"
    required = true
    default = 1000
    arg_type = Int

    "--Nx"
    help = "lattice size in x-direction"
    required = true
    default = 3
    arg_type = Int
    
    "--Ny"
    help = "lattice size in y-direction"
    required = true
    default = 3
    arg_type = Int
    
    "--h_min"
    help = "minimum transverse field strength"
    default = 0.5
    arg_type = Float64
    
    "--h_max"
    help = "maximum transverse field strength"
    default = 4.0
    arg_type = Float64
end

args = parse_args(s)

samples_num = args["samples_num"]
shots = args["shots"]
Nx = args["Nx"]
Ny = args["Ny"]
h_min = args["h_min"]
h_max = args["h_max"]

qubits = Nx * Ny
N_bonds = 2 * Nx * Ny - Nx - Ny


# Initialize data storage
coupling_strength_list = []
h_field_list = []
meas_sample_list = []
ground_state_energy_list = []
approx_correlation_matrix_list = []
exact_correlation_matrix_list = []
exact_entropy_list, approx_entropy_list = [], []


for i = 1:samples_num
    J = make_coupling_strength_tfim_2d(Nx, Ny)
    
    h = h_min + rand() * (h_max - h_min)
    
    push!(coupling_strength_list, vec(J))
    push!(h_field_list, h)
    
    # Solve for ground state using DMRG
    try
        energy, psi, H = transverse_field_ising_2d(Nx, Ny, spin, nsweeps, maxdim, cutoff, J, h)
        push!(ground_state_energy_list, Float32(energy))
        
        obs = getsamples(psi, randombases(qubits, shots, local_basis=["X", "Y", "Z"]))
        meas_samples, approx_correlation = cal_shadow_correlation(obs)
        exact_correlation = (compute_correlation_paulix_norm(psi) .+ compute_correlation_pauliy_norm(psi) .+ compute_correlation_pauliz_norm(psi)) ./ 3.0
        
        push!(meas_sample_list, meas_samples) 
        push!(approx_correlation_matrix_list, approx_correlation)
        push!(exact_correlation_matrix_list, exact_correlation)
        # Compute entanglement entropy for adjacent pairs
        # For 2D lattice, we compute entropy for horizontally adjacent qubits
        exact_entropy = []
        approx_entropy = []
        
        for row in 0:(Ny-1)
            for col in 0:(Nx-2)
                site_a = row * Nx + col + 1
                site_b = site_a + 1
                
                # Exact Renyi entropy
                push!(exact_entropy, exact_renyi_entropy_size_two(psi, site_a, site_b))
                
                # Approximate Renyi entropy from shadow
                push!(approx_entropy, cal_shadow_renyi_entropy(obs, [site_a, site_b]))
            end
        end
        
        push!(exact_entropy_list, exact_entropy)
        push!(approx_entropy_list, approx_entropy)
        
        # Progress reporting
        if i % 10 == 0
            avg_energy = mean(ground_state_energy_list)
            println("Progress: $i/$samples_num | Avg Energy: $(round(avg_energy, digits=4))")
        end
        
    catch e
        println("Warning: Failed to compute sample $i - $e")
        println("Skipping this sample...")
        # Remove the added parameters
        pop!(coupling_strength_list)
        pop!(h_field_list)
        continue
    end
end

df = DataFrame(
    #Nx = fill(Nx, length(ground_state_energy_list)),
    #Ny = fill(Ny, length(ground_state_energy_list)),
    coupling_strength = coupling_strength_list,
    #transverse_field = h_field_list,
    measurement_samples = meas_sample_list,
    ground_state_energy = ground_state_energy_list, 
    approx_correlation = approx_correlation_matrix_list,
    approx_entropy = approx_entropy_list,
    exact_correlation = exact_correlation_matrix_list,
    exact_entropy = exact_entropy_list
)
println("Writing dataset to CSV file... Info: ($Nx, $Ny), samples $samples_num, shots: $shots. ...Success!")
mkpath("dataset_generation/dataset_results/tfim_2d")
CSV.write("dataset_generation/dataset_results/tfim_2d/n$samples_num|X(coupling, meas$shots)_y(energy,entropy,corrs)_q($Nx, $Ny).csv", df);


