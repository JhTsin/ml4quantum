using ITensors, ITensorMPS
using Statistics

include("Hamiltonian.jl")
include("mps_utils.jl")

println("TFIM-2D entanglement debug runner")
println("="^60)

# Helper: compute Renyi-2 bond entanglement across bond b
function bond_entropy_renyi2(psi::MPS, b::Int)
    ps = copy(psi)
    orthogonalize!(ps, b)
    U, S, V = svd(ps[b], (linkinds(ps, b-1)..., siteinds(ps, b)...))
    λ = diag(S)
    λ ./= norm(λ)
    return -log2(sum(λ .^ 4))
end

# Helper: average pair entropy (horizontal and vertical adjacent pairs)
function avg_pair_entropy_renyi2(psi::MPS, Nx::Int, Ny::Int)
    entropies = Float64[]
    # horizontal pairs
    for row in 0:(Ny-1)
        for col in 0:(Nx-2)
            a = row * Nx + col + 1
            b = a + 1
            ps = copy(psi)
            orthogonalize!(ps, a)
            psdag = prime(dag(ps), linkinds(ps))
            rho_ab = prime(ps[a], linkinds(ps, a-1)) * prime(psdag[a], siteinds(ps)[a])
            rho_ab *= prime(ps[b], linkinds(ps, b)) * prime(psdag[b], siteinds(ps)[b])
            D, _ = eigen(rho_ab)
            vals = real(diag(D))
            s = sum(vals)
            if s != 0
                vals ./= s
            end
            purity = sum(vals .^ 2)
            push!(entropies, purity > 1e-12 ? -log2(purity) : 0.0)
        end
    end
    # vertical pairs
    for row in 0:(Ny-2)
        for col in 0:(Nx-1)
            a = row * Nx + col + 1
            b = a + Nx
            ps = copy(psi)
            orthogonalize!(ps, a)
            psdag = prime(dag(ps), linkinds(ps))
            rho_ab = prime(ps[a], linkinds(ps, a-1)) * prime(psdag[a], siteinds(ps)[a])
            rho_ab *= prime(ps[b], linkinds(ps, b)) * prime(psdag[b], siteinds(ps)[b])
            D, _ = eigen(rho_ab)
            vals = real(diag(D))
            s = sum(vals)
            if s != 0
                vals ./= s
            end
            purity = sum(vals .^ 2)
            push!(entropies, purity > 1e-12 ? -log2(purity) : 0.0)
        end
    end
    return mean(entropies)
end

function run_suite()
    Nx, Ny = 2, 2
    
    # MODE SELECTION: Uncomment ONE of the following modes
    mode = "size_two"  # Options: "size_two", "cut", "block"
    
    if mode == "size_two"
        println("\nMode: exact_renyi_entropy_size_two (adjacent pairs)")
        println("-"^60)
        
        configs = [
            (name = "theta_h sweep (fixed θj=-π/2, nsteps=10)", sweep = [(θj = -π/2, θh = x, nsteps = 10) for x in (π/4, 3π/8, π/2, 5π/8, 3π/4)]),
            (name = "theta_j sweep (fixed θh=π/2, nsteps=10)", sweep = [(θj = x, θh = π/2, nsteps = 10) for x in (-π/6, -π/4, -π/3, -π/2)]),
            (name = "nsteps sweep (fixed θj=-π/2, θh=π/2)", sweep = [(θj = -π/2, θh = π/2, nsteps = s) for s in (1, 3, 5, 7, 10, 15, 20)])
        ]

        for cfg in configs
            println("\n" * cfg.name)
            println("-"^60)
            for p in cfg.sweep
                psi, sites = tfim_2d_quantum_circuit_pro(Nx, Ny; θj=p.θj, θh=p.θh, nsteps=p.nsteps)
                
                # Calculate exact_renyi_entropy_size_two for all adjacent pairs
                vals = Float64[]
                for row in 0:(Ny-1)
                    for col in 0:(Nx-2)
                        a = row * Nx + col + 1
                        b = a + 1
                        ent = exact_renyi_entropy_size_two(copy(psi), a, b)
                        push!(vals, ent)
                    end
                end
                avg_entropy = mean(vals)
                println("θj=$(round(p.θj/π,digits=3))π, θh=$(round(p.θh/π,digits=3))π, nsteps=$(p.nsteps) -> size_two_avg=$(round(avg_entropy,digits=6))")
            end
        end

        # Compare nsteps=5 vs 10
        println("\nCompare nsteps=5 vs 10 at θj=-π/2, θh=π/2")
        println("-"^60)
        for s in (5, 10)
            psi, sites = tfim_2d_quantum_circuit_pro(Nx, Ny; θj=-π/2, θh=π/2, nsteps=s)
            vals = Float64[]
            for row in 0:(Ny-1)
                for col in 0:(Nx-2)
                    a = row * Nx + col + 1
                    b = a + 1
                    ent = exact_renyi_entropy_size_two(copy(psi), a, b)
                    push!(vals, ent)
                end
            end
            avg_entropy = mean(vals)
            println("nsteps=$(s): size_two_avg=$(round(avg_entropy,digits=6))")
        end
        
    elseif mode == "cut"
        println("\nMode: exact_renyi_entropy_cut (bond cuts)")
        println("-"^60)
        
        configs = [
            (name = "theta_h sweep (fixed θj=-π/2, nsteps=10)", sweep = [(θj = -π/2, θh = x, nsteps = 10) for x in (π/4, 3π/8, π/2, 5π/8, 3π/4)]),
            (name = "theta_j sweep (fixed θh=π/2, nsteps=10)", sweep = [(θj = x, θh = π/2, nsteps = 10) for x in (-π/6, -π/4, -π/3, -π/2)]),
            (name = "nsteps sweep (fixed θj=-π/2, θh=π/2)", sweep = [(θj = -π/2, θh = π/2, nsteps = s) for s in (1, 3, 5, 7, 10, 15, 20)])
        ]

        for cfg in configs
            println("\n" * cfg.name)
            println("-"^60)
            for p in cfg.sweep
                psi, sites = tfim_2d_quantum_circuit_pro(Nx, Ny; θj=p.θj, θh=p.θh, nsteps=p.nsteps)
                
                # Calculate exact_renyi_entropy_cut for all bonds
                vals = Float64[]
                for bond in 1:(Nx*Ny-1)
                    ent = exact_renyi_entropy_cut(copy(psi), bond)
                    push!(vals, ent)
                end
                avg_entropy = mean(vals)
                println("θj=$(round(p.θj/π,digits=3))π, θh=$(round(p.θh/π,digits=3))π, nsteps=$(p.nsteps) -> cut_avg=$(round(avg_entropy,digits=6))")
            end
        end

        # Compare nsteps=5 vs 10
        println("\nCompare nsteps=5 vs 10 at θj=-π/2, θh=π/2")
        println("-"^60)
        for s in (5, 10)
            psi, sites = tfim_2d_quantum_circuit_pro(Nx, Ny; θj=-π/2, θh=π/2, nsteps=s)
            vals = Float64[]
            for bond in 1:(Nx*Ny-1)
                ent = exact_renyi_entropy_cut(copy(psi), bond)
                push!(vals, ent)
            end
            avg_entropy = mean(vals)
            println("nsteps=$(s): cut_avg=$(round(avg_entropy,digits=6))")
        end
        
    elseif mode == "block"
        println("\nMode: exact_renyi_entropy_block (blocks)")
        println("-"^60)
        
        configs = [
            (name = "theta_h sweep (fixed θj=-π/2, nsteps=10)", sweep = [(θj = -π/2, θh = x, nsteps = 10) for x in (π/4, 3π/8, π/2, 5π/8, 3π/4)]),
            (name = "theta_j sweep (fixed θh=π/2, nsteps=10)", sweep = [(θj = x, θh = π/2, nsteps = 10) for x in (-π/6, -π/4, -π/3, -π/2)]),
            (name = "nsteps sweep (fixed θj=-π/2, θh=π/2)", sweep = [(θj = -π/2, θh = π/2, nsteps = s) for s in (1, 3, 5, 7, 10, 15, 20)])
        ]

        for cfg in configs
            println("\n" * cfg.name)
            println("-"^60)
            for p in cfg.sweep
                psi, sites = tfim_2d_quantum_circuit_pro(Nx, Ny; θj=p.θj, θh=p.θh, nsteps=p.nsteps)
                
                # Calculate exact_renyi_entropy_block for adjacent pairs
                vals = Float64[]
                # Horizontal adjacent pairs
                for row in 0:(Ny-1)
                    for col in 0:(Nx-2)
                        a = row * Nx + col + 1
                        blk = [a, a+1]
                        ent = exact_renyi_entropy_block(copy(psi), blk)
                        push!(vals, ent)
                    end
                end
                # Vertical adjacent pairs
                for row in 0:(Ny-2)
                    for col in 0:(Nx-1)
                        a = row * Nx + col + 1
                        blk = [a, a+Nx]
                        ent = exact_renyi_entropy_block(copy(psi), blk)
                        push!(vals, ent)
                    end
                end
                avg_entropy = mean(vals)
                println("θj=$(round(p.θj/π,digits=3))π, θh=$(round(p.θh/π,digits=3))π, nsteps=$(p.nsteps) -> block_avg=$(round(avg_entropy,digits=6))")
            end
        end

        # Compare nsteps=5 vs 10
        println("\nCompare nsteps=5 vs 10 at θj=-π/2, θh=π/2")
        println("-"^60)
        for s in (5, 10)
            psi, sites = tfim_2d_quantum_circuit_pro(Nx, Ny; θj=-π/2, θh=π/2, nsteps=s)
            vals = Float64[]
            # Horizontal adjacent pairs
            for row in 0:(Ny-1)
                for col in 0:(Nx-2)
                    a = row * Nx + col + 1
                    blk = [a, a+1]
                    ent = exact_renyi_entropy_block(copy(psi), blk)
                    push!(vals, ent)
                end
            end
            # Vertical adjacent pairs
            for row in 0:(Ny-2)
                for col in 0:(Nx-1)
                    a = row * Nx + col + 1
                    blk = [a, a+Nx]
                    ent = exact_renyi_entropy_block(copy(psi), blk)
                    push!(vals, ent)
                end
            end
            avg_entropy = mean(vals)
            println("nsteps=$(s): block_avg=$(round(avg_entropy,digits=6))")
        end
    end
end

run_suite()


