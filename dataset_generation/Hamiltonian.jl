using ITensors, ITensorMPS

# Tang et al. 2024 ICLR
function heisenberg_1d_tang(N, spin, nsweeps, maxdim, cutoff, coupling_strength)
    J = coupling_strength
    sites = siteinds("S=$spin", N)
    os = OpSum()
    for i=1:N-1
        for j=i+1:N
            os += J[i, j], "Sx", i, "Sx", j
            os += J[i, j], "Sy", i, "Sy", j
            os += J[i, j], "Sz", i, "Sz", j
        end
    end
    H = MPO(os, sites)
    
    psi0 = MPS(sites, n -> isodd(n) ? "Up" : "Dn")
    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff, outputlevel=0)
    return energy, psi, H
end;

# Huang et al. 2022 Science
function heisenberg_2d_huang(Nx, Ny, spin, nsweeps, coupling_strength, maxdim, cutoff)
    N = Nx * Ny
    J = coupling_strength
    sites = siteinds("S=$spin", N)
    lattice = square_lattice(Nx, Ny; yperiodic=false)
    i = 1
    os = OpSum()
    for b in lattice
        os += J[i] / 2, "S+", b.s1, "S-", b.s2
        os += J[i] / 2, "S-", b.s1, "S+", b.s2
        os += J[i], "Sz", b.s1, "Sz", b.s2
        i += 1
    end
    H = MPO(os, sites)
    # H = map(os -> device(MPO(Float32, os, sites)), os)
    state = [isodd(n) ? "Up" : "Dn" for n=1:N]
    # psi0 = device(complex.(MPS(sites, state)))
    psi0 = MPS(sites, state)
    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff, outputlevel=0)
    
    return energy, psi, H
end;



# Zhu et al. 2022
function xxz_1d(N, spin, nsweeps, maxdim, cutoff, delta)
    sites = siteinds("S=$spin", N)
    os = OpSum()
    for i=1:N-1
        os -= delta[i] / 2, "Sx", i, "Sx", i+1
        os -= delta[i] / 2, "Sy", i, "Sy", i+1
        os -= 1, "Sz", i, "Sz", i+1
    end;
    H = MPO(os, sites)
    psi0 = MPS(sites, N -> isodd(N) ? "Up" : "Dn")
    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff,  outputlevel=0);
    return energy, psi
end;


# Wu et al. 2023
function bond_alter_xxz_1d(N, spin, nsweeps, maxdim, cutoff, coupling_strength, coupling_strength2, delta)
    sites = siteinds("S=$spin", N)
    os = OpSum()
    J = coupling_strength
    J_prime = coupling_strength2
    
    for i=1:N/2
        os += J / 2, "S+", 2*i-1, "S-", 2*i
        os += J / 2, "S-", 2*i-1, "S+", 2*i
        os += J*delta, "Sz", 2*i-1, "Sz", 2*i
    end;
    
    for i=1:(N/2-1)
        os += J_prime / 2, "S+", 2*i, "S-", 2*i+1
        os += J_prime / 2, "S-", 2*i, "S+", 2*i+1
        os += J_prime*delta, "Sz", 2*i, "Sz", 2*i+1
    end;
    H = MPO(os, sites)
    psi0 = MPS(sites, N -> isodd(N) ? "Up" : "Dn")
    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)
    return energy, psi
end;


# Wu et al. 2023
function cluster_ising_1d(N, spin, nsweeps, maxdim, cutoff, h1, h2)
    sites = siteinds("S=$spin", N)
    os = OpSum() 
    for i=1:(N-2)
        os -= 1, "Sz", i, "Sx", i+1, "Sz", i+2 
    end
    for i=1:N
        os -= h1, "Sx", i
    end
    for i=1:N-1
        os -= h2, "Sx", i, "Sx", i+1
    end
    H = MPO(os, sites);
    psi0 = MPS(sites, N -> isodd(N) ? "Up" : "Dn");
    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff, outputlevel=0);
    return energy, psi 
end;

# Zhu et al. 2022
# ferromagnetic_ising_1d
function transverse_field_ising_1d(N, spin, nsweeps, maxdim, cutoff, coupling_strength, eigsolve_krylovdim)
    sites = siteinds("S=$spin", N)
    os = OpSum()
    J = coupling_strength
    h = 4.0 # theortical bound = 2*abs(h-1)
    weight = 200 * h
    noise = [1E-6]
    for i=1:N-1
        os -= J[i], "Sz", i, "Sz", i+1
    end;
    for i=1:N
        os -= 1, "Sx", i
    end;
    H = MPO(os, sites)
    psi_init = MPS(sites, N -> isodd(N) ? "Up" : "Dn")
    ground_state_energy, psi1 = dmrg(H, psi_init; nsweeps, maxdim, cutoff, eigsolve_krylovdim, outputlevel=0);
    psi_init2 = MPS(sites, N -> isodd(N) ? "Up" : "Dn")
    first_excited_energy, psi2 = dmrg(H, [psi1], psi_init2; nsweeps, maxdim, cutoff, weight, eigsolve_krylovdim, outputlevel=0);
    return ground_state_energy, psi1, first_excited_energy, psi2
end;


# Wu et al. 2023
function perturbed_cluster_ising_1d(N, spin, nsweeps, maxdim, cutoff, h1, h2, h3)
    sites = siteinds("S=$spin", N)
    os = OpSum()
    for i=1:N-2
        os -= 1, "Sz", i, "Sx", i+1, "Sz", i+2 
    end;
    for i=1:N
        os -= h1, "Sx", i
    end;
    for i=1:N-1
        os -= h2, "Sx", i, "Sx", i+1
    end;
    for i=1:N-2
        os += h3, "Sz", i, "Sz", i+2
    end;
    H = MPO(os, sites)
    psi0 = MPS(sites, N -> isodd(N) ? "Up" : "Dn")
    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)
    return energy, psi
end


# 2D Transverse Field Ising Model
# 2023 Nature Evidence for the utility of quantum computing before fault tolerance
function transverse_field_ising_2d(Nx, Ny, spin, nsweeps, maxdim, cutoff, coupling_strength, h)
    N = Nx * Ny
    sites = siteinds("S=$spin", N)
    lattice = square_lattice(Nx, Ny; yperiodic=false)
    
    os = OpSum()
    
    # Ising interaction: -J * Sz_i * Sz_j for nearest neighbors
    bond_idx = 1
    for b in lattice
        os -= coupling_strength[bond_idx], "Sz", b.s1, "Sz", b.s2
        bond_idx += 1
    end
    
    # Transverse field: -h * Sx_i for all sites
    for i=1:N
        os -= h, "Sx", i
    end
    
    H = MPO(os, sites)
    
    # Initialize with alternating up/down pattern
    state = [isodd(n) ? "Up" : "Dn" for n=1:N]
    psi0 = MPS(sites, state)
    
    # Run DMRG to find ground state
    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff, outputlevel=0)
    
    return energy, psi, H
end;


#------------------- digital quantum simulation -------------------

"""
2D TFIM Trotter化量子电路
基于Nature 2023论文: Figure 1的电路结构

电路结构:
每个Trotter步骤 = RX(θh)层 + 3层RZZ(θj)门
- RX(θh) = exp(-i*θh/2*X) 作用到所有比特
- RZZ(θj) = exp(-i*θj/2*Z⊗Z) 三组并行层

参数:
- Nx, Ny: 格子尺寸
- θj, θh: 旋转角度
- nsteps: Trotter步数
"""
function tfim_2d_quantum_circuit(Nx::Int, Ny::Int; 
    θj::Float64=-π/2, θh::Float64=π/4, nsteps::Int=5)
    
    N = Nx * Ny
    sites = siteinds("Qubit", N)
    
    # 初态: |0⟩⊗|0⟩...⊗|0⟩
    psi = MPS(sites, ["0" for _ in 1:N])
    
    # Trotter演化
    for step in 1:nsteps
        # 1. RX层
        for i in 1:N
            gate = op("Rx", sites, i; θ=θh)
            psi = apply(gate, psi; cutoff=1e-10, maxdim=512)
        end
        
        # 2. 水平偶数列RZZ层
        for r in 0:Ny-1
            for c in 0:2:Nx-2
                i = r*Nx + c + 1
                j = i + 1
                gate = op("Rzz", sites, i, j; ϕ=θj) # ITensors使用ϕ作为参数名
                psi = apply(gate, psi; cutoff=1e-10, maxdim=512)
            end
        end
        
        # 3. 水平奇数列RZZ层
        for r in 0:Ny-1
            for c in 1:2:Nx-2
                i = r*Nx + c + 1
                j = i + 1
                gate = op("Rzz", sites, i, j; ϕ=θj)
                psi = apply(gate, psi; cutoff=1e-10, maxdim=512)
            end
        end
        
        # 4. 垂直RZZ层
        for r in 0:Ny-2
            for c in 0:Nx-1
                i = r*Nx + c + 1
                j = i + Nx
                gate = op("Rzz", sites, i, j; ϕ=θj)
                psi = apply(gate, psi; cutoff=1e-10, maxdim=512)
            end
        end
    end
    
    return psi, sites
end


