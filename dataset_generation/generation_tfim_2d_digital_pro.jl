
using ITensors, ITensorMPS
using DataFrames
using CSV
using ArgParse
using PastaQ
using Statistics
using PyCall
using JSON

# Configure PyCall to use conda Python
ENV["PYTHON"] = "/home/ubuntu/miniconda3/bin/python3"
include("mps_utils.jl")
include("utils.jl")
include("Hamiltonian.jl")
include("shadow.jl")

# Import Python function
# Add current directory to Python path
sys = pyimport("sys")
push!(sys.path, pwd())

# Import the Python file directly
py"""
import sys
sys.path.append('.')
import tfim_2d_pennyLane
"""

py_tfim = pyimport("tfim_2d_pennyLane")

# Julia wrapper for Python tfim_2d_quantum_circuit_pro
function tfim_2d_quantum_circuit_pro(Nx::Int, Ny::Int; θj1=-π, θj2=-π/4, θj3=-π/2, θh1=π/2, nsteps=5)
    # Time Python evolution
    python_start = time()
    result = py_tfim.get_quantum_state_for_julia(Nx, Ny, θj1, θj2, θj3, θh1, nsteps)
    python_elapsed = time() - python_start
    
    # Parse JSON result
    data = JSON.parse(result)
    
    if !data["success"]
        error("Python function failed: $(get(data, "error", "Unknown error"))")
    end
    
    # Time Julia conversion
    conversion_start = time()
    
    # Create sites (assuming spin-1/2)
    N = Nx * Ny
    sites = siteinds("S=1/2", N)
    
    # Convert state vector to complex numbers
    state_vector = [complex(Float64(s[1]), Float64(s[2])) for s in data["state_vector"]]
    
    # Create MPS from state vector using ITensors
    # First, create an ITensor from the state vector
    # The state vector has 2^N elements, we need to reshape it
    state_array = reshape(state_vector, tuple([2 for _ in 1:N]...))
    state_tensor = ITensor(state_array, sites...)
    
    # Create MPS from the ITensor using SVD
    psi = MPS(state_tensor, sites)
    
    conversion_elapsed = time() - conversion_start
    
    # Print timing information
    total_time = (python_elapsed + conversion_elapsed) * 1000
    python_percent = (python_elapsed / (python_elapsed + conversion_elapsed)) * 100
    julia_percent = (conversion_elapsed / (python_elapsed + conversion_elapsed)) * 100
    
    println("Total: $(round(total_time, digits=2)) ms | Python演化: $(round(python_elapsed*1000, digits=2)) ms ($(round(python_percent, digits=1))%) | Julia转化: $(round(conversion_elapsed*1000, digits=2)) ms ($(round(julia_percent, digits=1))%)")
    
    return psi, sites
end


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
angle_list = []


for i = 1:samples_num

    # 1 2 3
    θj1= rand()*pi  
    θj2= rand()*pi
    θj3= rand()*pi
    θh1= rand()*pi
    nstep = rand(1:10)
    
    # Store angle parameters
    push!(angle_list, [θj1, θj2, θj3, θh1, nstep])

    psi, sites = tfim_2d_quantum_circuit_pro_old(Nx, Ny; θj1=θj1, θj2=θj2, θj3=θj3, θh1=θh1, nsteps=nstep)

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



    approx_entropy = [cal_shadow_renyi_entropy(obs, [a, a+1]) for a in 1:Nx*Ny-1];
    exact_entropy = [exact_renyi_entropy_size_two(psi, a, a+1) for a in 1:Nx*Ny-1];
    
    push!(exact_entropy_list, exact_entropy)
    push!(approx_entropy_list, approx_entropy)
    
end

df = DataFrame(
    angle_parameters = angle_list,
    measurement_samples = meas_sample_list,
    approx_correlation = approx_correlation_matrix_list,
    approx_entropy = approx_entropy_list,
    exact_correlation = exact_correlation_matrix_list,
    exact_entropy = exact_entropy_list,
    
)
println("Writing dataset to CSV file... Info: ($Nx, $Ny), samples $samples_num, shots: $shots. ...Success!")
mkpath("dataset_generation/dataset_results/tfim_2d_new")
CSV.write("dataset_generation/dataset_results/tfim_2d_new/n$samples_num|X(coupling, meas$shots)_y(energy,entropy,corrs)_q($Nx, $Ny).csv", df);



