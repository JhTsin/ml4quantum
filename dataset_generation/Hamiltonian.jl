using ITensors, ITensorMPS

using PauliPropagation

using LinearAlgebra



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
    sites = siteinds("S=1/2", N)
    
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

function tfim_2d_quantum_circuit_pro_old(Nx::Int, Ny::Int; 
    θj1::Float64=-π/2, θj2::Float64=-π/2,θj3::Float64=-π/2,θh1::Float64=π/4, nsteps::Int=5)
    
    N = Nx * Ny
    sites = siteinds("S=1/2", N)
    # 初态: |0⟩⊗|0⟩...⊗|0⟩
    psi = MPS(sites, ["0" for _ in 1:N])

    # Trotter演化
    for step in 1:nsteps
        # 1. RX层
        for i in 1:N
            gate = op("Rx", sites, i; θ=θh1)
            psi = ITensors.apply(gate, psi; cutoff=1e-10, maxdim=512)
        end
        
        # 2. 水平偶数列RZZ层
        for r in 0:Ny-1
            for c in 0:2:Nx-2
                i = r*Nx + c + 1
                j = i + 1
                gate = op("Rzz", sites, i, j; ϕ=θj1) # ITensors使用ϕ作为参数名
                psi = ITensors.apply(gate, psi; cutoff=1e-10, maxdim=512)
            end
        end
        
        # 3. 水平奇数列RZZ层
        for r in 0:Ny-1
            for c in 1:2:Nx-2
                i = r*Nx + c + 1
                j = i + 1
                gate = op("Rzz", sites, i, j; ϕ=θj2)
                psi = ITensors.apply(gate, psi; cutoff=1e-10, maxdim=512)
            end
        end
        
        # 4. 垂直RZZ层
        for r in 0:Ny-2
            for c in 0:Nx-1
                i = r*Nx + c + 1
                j = i + Nx
                gate = op("Rzz", sites, i, j; ϕ=θj3)
                psi = ITensors.apply(gate, psi; cutoff=1e-10, maxdim=512)
            end
        end
    end
    
    return psi, sites
end


"""
2D TFIM Trotter化量子电路 - 基于Pauli Propagation框架
借鉴notebook中的架构来演化二维TFIM，完全参照ipynb的函数写法和调用

参数:
- Nx, Ny: 格子尺寸
- dt: 时间步长 (默认0.05，与notebook一致)
- J: 耦合强度 (默认2.0，与notebook一致)
- h: 横向场强度 (默认1.0，与notebook一致)
- nsteps: Trotter步数 (默认20，与notebook一致)
- noise: 噪声开关 (默认false)
- min_abs_coeff: 截断系数 (默认1e-6，与notebook一致)
- max_weight: 最大权重 (默认15，与notebook一致)
- depol_strength: 去极化噪声强度 (默认4/3*0.018)
- dephase_strength: 退相干噪声强度 (默认2*0.012)
- noise_level: 噪声水平 (默认1.0)
- topology: 拓扑结构 (默认使用2D方形格子)
- observable: 可观测量 (默认⟨ZZ⟩平均值)

返回:
- expvals: 期望值时间演化
- psum: 最终的PauliSum (如果output_psum=true)

-对于薛定谔绘景下的演化，要把初始观测量O变为初始态ρ，然后1、电路取反 2、角度取负 3、非角度门，CliffordGate 门使用 transposecliffordmap() 处理
"""
function tfim_2d_quantum_circuit_pp(Nx::Int, Ny::Int;
    dt::Float64=0.05, J::Float64=2.0, h::Float64=1.0, nsteps::Int=20,
    noise::Bool=false, min_abs_coeff::Float64=1e-6, max_weight::Int=15,
    depol_strength::Float64=4/3*0.018, dephase_strength::Float64=2*0.012, 
    noise_level::Float64=1.0, topology::Union{Nothing, Vector{Tuple{Int,Int}}}=nothing,
    observable::Union{Nothing, PauliSum}=nothing, output_psum::Bool=true)
    
    # 开始计时
    total_start_time = time()
    println("=== TFIM 2D Quantum Circuit PP 计时开始 ===")
    
    N = Nx * Ny

    # 如果没有提供拓扑，使用2D方形格子
    if topology === nothing
        topology = Vector{Tuple{Int,Int}}()
        # 水平连接
        for r in 0:Ny-1
            for c in 0:Nx-2
                i = r * Nx + c + 1
                j = i + 1
                push!(topology, (i, j))
            end
        end
        # 垂直连接
        for r in 0:Ny-2
            for c in 0:Nx-1
                i = r * Nx + c + 1
                j = i + Nx
                push!(topology, (i, j))
            end
        end
    end
    
    # 计时：可观测量初始化
    observable_start_time = time()
    println("开始初始化可观测量...")
    
    function rho_zero_state(N::Int)
        rho = PauliSum(N)
        # 枚举 0..(2^N-1) 的所有 Z-子集
        for mask in 0:(UInt(1) << N) - 1
            paulis = Symbol[]   # 存放 :Z
            idxs   = Int[]      # 存放对应比特索引
            for i in 1:N
                if (mask >> (i-1)) & 0x1 == 1
                    push!(paulis, :Z)
                    push!(idxs, i)
                end
            end
            # 空 paulis/idxs 表示 I^{⊗N}（整串恒等）
            PauliPropagation.add!(rho, paulis, idxs, 1.0)
        end
        rho /= 2.0^N
        return rho
    end

    # 如果没有提供可观测量，使用⟨ZZ⟩平均值
    if observable === nothing
        observable = rho_zero_state(N)
    end
    
    observable_time = time() - observable_start_time
    println("可观测量初始化完成，耗时: $(round(observable_time, digits=4)) 秒")

    
    # 定义单层电路 - 使用PP框架的tfitrottercircuit, false后得到先X后ZZ的电路 --> U= RzzRx
    layer = tfitrottercircuit(N, 1, topology=topology, start_with_ZZ=false)
    
    # 定义参数 - 使用notebook中的define_thetas函数
    function define_thetas(circuit::Vector{Gate}, dt::Float64, J::Float64=2.0, h::Float64=1.0)
        # get indices indicating which angle corresponds to which gate 
        rzz_indices = getparameterindices(circuit, PauliRotation, [:Z, :Z])
        rx_indices = getparameterindices(circuit, PauliRotation, [:X])
        
        nparams = countparameters(circuit)
        thetas = zeros(nparams)
        
        # following eq. (3) from notebook
        thetas[rzz_indices] .= - J * dt * 2 * (3.0 + 2*rand())
        thetas[rx_indices] .= h * dt * 2 * (13.0 + 2*rand())
        return thetas
    end
    # 角度取反后，外加上之前的false，所以后面propagate的结果变成了 UOU^† ，最后需要让输入O变为初始态
    thetas = -define_thetas(layer, dt, J, h)

    # 计时：Trotter时间演化
    evolution_start_time = time()
    println("开始Trotter时间演化...")

    # 噪声应用函数 - 从notebook复制
    function applynoiselayer(psum::PauliSum; depol_strength=0.02, dephase_strength=0.02, noise_level=1.0)
        for (pstr, coeff) in psum
            set!(psum, pstr, 
                coeff*(1-noise_level*depol_strength)^countweight(pstr)*(1-noise_level*dephase_strength)^countxy(pstr))
        end 
    end
    
    # Trotter时间演化 - 从notebook复制并修改
    function trotter_time_evolution(
        steps::Int, layer::Vector{Gate}, observable::PauliSum, thetas::Vector{Float64};
        noise=false, min_abs_coeff=1e-10, max_weight=Inf,
        depol_strength=0.02, dephase_strength=0.02, noise_level=1.0,
        output_psum=true
    )
        obs = deepcopy(observable)
        expvals = [overlapwithzero(obs)]
        psum = obs  # initialize to have it in scope

        for _ in 1:steps
            psum = propagate!(layer, obs, thetas; min_abs_coeff, max_weight)
            
            if noise
                applynoiselayer(psum; depol_strength, dephase_strength, noise_level)
            end

            push!(expvals, overlapwithzero(psum))
        end

        return output_psum ? (expvals, psum) : expvals
    end
    
    expvals, psum = trotter_time_evolution(
        nsteps, layer, observable, thetas;
        noise=noise, min_abs_coeff=min_abs_coeff, max_weight=max_weight,
        depol_strength=depol_strength, dephase_strength=dephase_strength, 
        noise_level=noise_level, output_psum=true)
    
    evolution_time = time() - evolution_start_time
    println("Trotter时间演化完成，耗时: $(round(evolution_time, digits=4)) 秒")

    # 计时：PauliSum到MPS转换
    conversion_start_time = time()
    println("开始PauliSum到MPS转换...")

    # psum --> MPS
    # Step 1: PauliSum -> Dense Matrix
    function paulisum_to_dense_matrix(psum::PauliSum)
        N = psum.nqubits
        # 支持整数或符号键
        σ = Dict{Any, Matrix{ComplexF64}}(
            0x00 => [1 0; 0 1],          # I
            0x01 => [0 1; 1 0],          # X
            0x02 => [0 -im; im 0],       # Y
            0x03 => [1 0; 0 -1],         # Z
            :I   => [1 0; 0 1],
            :X   => [0 1; 1 0],
            :Y   => [0 -im; im 0],
            :Z   => [1 0; 0 -1],
        )

        ρ = zeros(ComplexF64, 2^N, 2^N)

        for (pstr, coeff) in psum
            ops = [σ[getpauli(pstr, i)] for i in 1:N]
            P = reduce(kron, ops)
            ρ += coeff * P
        end
        return ρ
    end

    # Step 2: Dense -> ITensor (MPO/MPS)

    function dense_to_mps_matrix(ρ::Matrix{ComplexF64})
        N = Int(log2(size(ρ, 1)))
        ψ = vec(ρ[:, 1])  # 提取第一列，对应 |ψ⟩
        s = siteinds("S=1/2", N)
        return MPS(ψ, s)
    end


    # Step 3: Wrapper with detailed timing
    function paulisum_to_mps(psum::PauliSum)
        # 计时：PauliSum到密集矩阵转换
        dense_start_time = time()
        println("  开始PauliSum到密集矩阵转换...")
        ρ = paulisum_to_dense_matrix(psum)
        dense_time = time() - dense_start_time
        println("  PauliSum到密集矩阵转换完成，耗时: $(round(dense_time, digits=4)) 秒")
        
        # 计时：密集矩阵到MPS转换
        mps_start_time = time()
        println("  开始密集矩阵到MPS转换...")
        psi = dense_to_mps_matrix(ρ)
        mps_time = time() - mps_start_time
        println("  密集矩阵到MPS转换完成，耗时: $(round(mps_time, digits=4)) 秒")
        
        # 计时：MPS归一化
        norm_start_time = time()
        println("  开始MPS归一化...")
        normalize!(psi)
        norm_time = time() - norm_start_time
        println("  MPS归一化完成，耗时: $(round(norm_time, digits=4)) 秒")
        
        return psi
    end
    psi= paulisum_to_mps(psum)
    
    conversion_time = time() - conversion_start_time
    println("PauliSum到MPS转换完成，耗时: $(round(conversion_time, digits=4)) 秒")
    
    # 计算总时间并输出详细计时信息
    total_time = time() - total_start_time
    println("\n=== 详细计时报告 ===")

    println("可观测量初始化: $(round(observable_time, digits=4)) 秒")

    println("Trotter时间演化: $(round(evolution_time, digits=4)) 秒")
    println("PauliSum到MPS转换: $(round(conversion_time, digits=4)) 秒")
    println("总时间: $(round(total_time, digits=4)) 秒")
    println("=== 计时结束 ===\n")
    
    return psi
end


