#!/usr/bin/env python3
"""
Optimized PennyLane implementation of tfim_2d_quantum_circuit_pro
Efficiently handles arbitrary angles and large systems
"""

import pennylane as qml
import numpy as np
import json
from typing import Dict, List, Tuple, Any, Optional

def tfim_2d_quantum_circuit_pro(
    Nx: int, 
    Ny: int, 
    theta_j1: float = -np.pi,
    theta_j2: float = -np.pi/4,
    theta_j3: float = -np.pi/2,
    theta_h1: float = np.pi/2,
    nsteps: int = 5,
    device: str = 'default.qubit'
) -> Dict[str, Any]:
    """
    Optimized 2D TFIM Quantum Circuit using PennyLane
    
    This function implements the same quantum circuit as the Julia version
    but with optimizations for arbitrary angles and efficient execution.
    
    Args:
        Nx, Ny: Lattice dimensions
        theta_j1, theta_j2, theta_j3: ZZ rotation angles for different layers
        theta_h1: X rotation angle
        nsteps: Number of Trotter steps
        device: PennyLane device ('default.qubit', 'lightning.qubit', etc.)
        
    Returns:
        Dictionary containing quantum state information compatible with Julia
    """
    
    N = Nx * Ny

    # Choose optimal device based on system size
    if N <= 20 and device == 'default.qubit':
        dev = qml.device('default.qubit', wires=N)
    elif N <= 30:
        try:
            dev = qml.device('lightning.qubit', wires=N)
        except:
            dev = qml.device('default.qubit', wires=N)
    else:
        # For large systems, use default.qubit with optimization
        dev = qml.device('default.qubit', wires=N)
    
    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit():
        """Optimized quantum circuit for 2D TFIM evolution"""
        
        # Explicitly set initial state to |0⟩⊗|0⟩⊗...⊗|0⟩ (equivalent to Julia's ["0" for _ in 1:N])
        # This corresponds to |↓⟩⊗|↓⟩⊗...⊗|↓⟩ in S=1/2 representation
        initial_state = np.zeros(2**N)
        initial_state[0] = 1.0  # |00...0⟩ state
        qml.StatePrep(initial_state, wires=range(N))
        
        # Trotter evolution with optimized gate ordering
        for step in range(nsteps):
            # 1. RX layer (transverse field) - can be parallelized
            for i in range(N):
                qml.RX(theta_h1, wires=i)
            
            # 2. Horizontal even columns RZZ layer
            for r in range(Ny):
                for c in range(0, Nx-1, 2):  # Even columns: 0, 2, 4, ...
                    i = r * Nx + c
                    j = i + 1
                    if j < N:  # Boundary check
                        qml.IsingZZ(theta_j1, wires=[i, j])
            
            # 3. Horizontal odd columns RZZ layer  
            for r in range(Ny):
                for c in range(1, Nx-1, 2):  # Odd columns: 1, 3, 5, ...
                    i = r * Nx + c
                    j = i + 1
                    if j < N:  # Boundary check
                        qml.IsingZZ(theta_j2, wires=[i, j])
            
            # 4. Vertical RZZ layer
            for r in range(Ny-1):
                for c in range(Nx):
                    i = r * Nx + c
                    j = i + Nx
                    if j < N:  # Boundary check
                        qml.IsingZZ(theta_j3, wires=[i, j])
        
        # Return state vector
        return qml.state()
    
    # Execute circuit with error handling
    try:
        state_vector = circuit()
        
        # Convert complex numbers to list format for JSON serialization
        state_list = []
        for amplitude in state_vector:
            if isinstance(amplitude, complex):
                state_list.append([float(amplitude.real), float(amplitude.imag)])
            else:
                state_list.append([float(amplitude), 0.0])
        
        # Return result compatible with Julia
        result = {
            'state_vector': state_list,
            'Nx': Nx,
            'Ny': Ny,
            'N': N,
            'theta_j1': float(theta_j1),
            'theta_j2': float(theta_j2),
            'theta_j3': float(theta_j3),
            'theta_h1': float(theta_h1),
            'nsteps': nsteps,
            'device': str(dev.name),
            'success': True
        }
        
        return result
        
    except Exception as e:
        # Return error information
        return {
            'state_vector': [],
            'Nx': Nx,
            'Ny': Ny,
            'N': N,
            'theta_j1': float(theta_j1),
            'theta_j2': float(theta_j2),
            'theta_j3': float(theta_j3),
            'theta_h1': float(theta_h1),
            'nsteps': nsteps,
            'device': str(dev.name),
            'success': False,
            'error': str(e)
        }

def get_quantum_state_for_julia(
    Nx: int, 
    Ny: int, 
    theta_j1: float = -np.pi/4,
    theta_j2: float = -np.pi/7,
    theta_j3: float = -np.pi/7,
    theta_h1: float = np.pi/5,
    nsteps: int = 5,
    device: str = 'default.qubit'
) -> str:
    """
    Wrapper function specifically designed for Julia PyCall interface
    
    Returns JSON string that Julia can easily parse
    """
    result = tfim_2d_quantum_circuit_pro(
        Nx, Ny, 
        theta_j1=theta_j1,
        theta_j2=theta_j2,
        theta_j3=theta_j3,
        theta_h1=theta_h1,
        nsteps=nsteps,
        device=device
    )
    return json.dumps(result)



# For direct testing
if __name__ == "__main__":
    # Test the function with different parameters
    print("Testing PennyLane tfim_2d_quantum_circuit_pro...")
    
    # Test 1: Small system
    result1 = tfim_2d_quantum_circuit_pro(2, 2, nsteps=5)
    print(f"Test 1 (2x2): Success={result1['success']}, State dim={len(result1['state_vector'])}")
    print(result1)

    # Test 2: Medium system
    result2 = tfim_2d_quantum_circuit_pro(4, 4, nsteps=5)
    print(f"Test 2 (4x4): Success={result2['success']}, State dim={len(result2['state_vector'])}")
    
    # Test 3: Different angles
    result3 = tfim_2d_quantum_circuit_pro(5, 5, theta_j1=-0.532, theta_j2=-0.743,theta_j3=-0.432,theta_h1=0.978, nsteps=10)
    print(f"Test 3 (custom angles5x5): Success={result3['success']}, State dim={len(result3['state_vector'])}")
    
    print("All tests completed!")
