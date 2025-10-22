
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

   # 1 2 3
    nstep = rand(2:9)
    psi = tfim_2d_quantum_circuit_pp(Nx, Ny; nsteps=10)
    #psi, sites = tfim_2d_quantum_circuit_pro(Nx, Ny; θj1=-π,θj2=-π/4,θj3=-π/2, θh1=π/2, nsteps=nstep)
    #psi, sites = tfim_2d_quantum_circuit_pro(Nx, Ny; θj1=-π/3,θj2=-2*π/3,θj3=-π/6, θh1=π/2, nsteps=5)
    #psi, sites = tfim_2d_quantum_circuit_pro(Nx, Ny; θj1=-π/2,θj2=-π/2,θj3=-2*π/3, θh1=π/2, nsteps=5)
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


    # 局部  pair  熵
    approx_entropy = [cal_shadow_renyi_entropy(obs, [a, a+1]) for a in 1:Nx*Ny-1];
    exact_entropy = [exact_renyi_entropy_size_two(psi, a, a+1) for a in 1:Nx*Ny-1];
    # for row in 0:(Ny-1)
    #     for col in 0:(Nx-2)
    #         site_a = row * Nx + col + 1
    #         site_b = site_a + 1
            
    #         # Exact Renyi entropy
    #         push!(exact_entropy, exact_renyi_entropy_size_two(psi, site_a, site_b))
            
    #         # Approximate Renyi entropy from shadow
    #         push!(approx_entropy, cal_shadow_renyi_entropy(obs, [site_a, site_b]))
    #     end
    # end
    
    push!(exact_entropy_list, exact_entropy)
    push!(approx_entropy_list, approx_entropy)
    
end

df = DataFrame(
    measurement_samples = meas_sample_list,
    approx_correlation = approx_correlation_matrix_list,
    approx_entropy = approx_entropy_list,
    exact_correlation = exact_correlation_matrix_list,
    exact_entropy = exact_entropy_list
)
println("Writing dataset to CSV file... Info: ($Nx, $Ny), samples $samples_num, shots: $shots. ...Success!")
mkpath("dataset_generation/dataset_results/tfim_2d_new")
CSV.write("dataset_generation/dataset_results/tfim_2d_new/n$samples_num|X(coupling, meas$shots)_y(energy,entropy,corrs)_q($Nx, $Ny).csv", df);


