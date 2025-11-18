"""
Demo script showing how to use the discrete penalty to promote binary density solutions.

This script demonstrates topology optimization with gradually increasing discrete penalty
to push the density field toward 0 or 1 values.
"""

import numpy as np
import femsolver_cont as femsolver
import gradient
import matplotlib.pyplot as plt

def run_discrete_optimization():
    """
    Run topology optimization with discrete penalty.
    """
    # Material properties
    E = 200e9  # Young's modulus (Pa)
    nu = 0.3   # Poisson's ratio
    L = 0.01   # Side length (m)
    t = 0.1    # Thickness (m)
    
    # Setup problem
    Ke = femsolver.element_stiffness_matrix(E, nu, L, t)
    
    # Define initial voxel grid
    voxels = np.array([[0,1], [1,1]])
    voxels = femsolver.sub_divide(voxels, 4)
    height, width = voxels.shape
    
    # Fixed nodes (bottom edge)
    fixed_nodes = list(range((height)*width, (height+1)*(width+1)))
    
    # Initialize with continuous densities
    voxels = voxels.astype(np.float64)
    orig_voxels = voxels.copy()
    
    # Setup stiffness and force
    K = femsolver.global_stiffness_matrix(Ke, voxels)
    n_dofs = K.shape[0]
    F = np.zeros((n_dofs, 1))
    F = femsolver.add_force_to_node(4, F, np.array([0.5, 0.5]))
    
    # Optimization parameters
    B = 500  # Target stress limit
    lambda_ = 0.0  # Regularization weight
    
    # Phase 1: Optimize without discrete penalty
    print("=" * 60)
    print("PHASE 1: Continuous optimization (discrete_weight=0)")
    print("=" * 60)
    discrete_weight = 0.0
    voxels, K = run_optimization_phase(voxels, K, Ke, F, fixed_nodes, B, lambda_, 
                                      discrete_weight, n_iters=50, step_size=0.2)
    
    print("\nPhase 1 results:")
    print_density_stats(voxels)
    
    # Phase 2: Light discrete penalty
    print("\n" + "=" * 60)
    print("PHASE 2: Light discrete penalty (discrete_weight=1e8)")
    print("=" * 60)
    discrete_weight = 1e8
    voxels, K = run_optimization_phase(voxels, K, Ke, F, fixed_nodes, B, lambda_,
                                      discrete_weight, n_iters=30, step_size=0.15)
    
    print("\nPhase 2 results:")
    print_density_stats(voxels)
    
    # Phase 3: Strong discrete penalty
    print("\n" + "=" * 60)
    print("PHASE 3: Strong discrete penalty (discrete_weight=1e9)")
    print("=" * 60)
    discrete_weight = 1e9
    voxels, K = run_optimization_phase(voxels, K, Ke, F, fixed_nodes, B, lambda_,
                                      discrete_weight, n_iters=30, step_size=0.1)
    
    print("\nFinal results:")
    print_density_stats(voxels)
    
    return voxels

def run_optimization_phase(voxels, K, Ke, F, fixed_nodes, B, lambda_, discrete_weight,
                          n_iters=30, step_size=0.2):
    """
    Run one phase of optimization with fixed discrete_weight.
    """
    E = 200e9
    nu = 0.3
    L = 0.01
    
    old_voxels = voxels.copy()
    threshold = 1e-8 * np.max(np.abs(K.data))
    
    for it in range(n_iters):
        # Solve FEM
        solver = femsolver.Solver(K, F)
        u, _ = solver.solve(K, F, fixed_nodes)
        
        # Compute stresses
        eps = femsolver.get_element_strains_fast(u, voxels, L)
        sigma = femsolver.get_element_stresses_fast(eps, E, nu)
        n_sigma = femsolver.get_node_values_fast(sigma, voxels, L)
        von_mises = femsolver.von_mises_stresses_node(n_sigma)
        
        # Compute objective and gradient
        if discrete_weight > 0:
            obj_val = gradient.obj_1_discrete(von_mises, voxels, B, lambda_, discrete_weight)
            grad = gradient.obj_1_discrete_grad(u, B, K, Ke, n_sigma, von_mises, voxels,
                                               lambda_, fixed_nodes, discrete_weight)
        else:
            obj_val = gradient.obj_1(von_mises, voxels, B, lambda_)
            grad = gradient.obj_1_grad(u, B, K, Ke, n_sigma, von_mises, voxels,
                                      lambda_, fixed_nodes)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 0 and it == 0:
            scaling = 1.0 / grad_norm
        elif grad_norm > 0:
            scaling = scaling
        
        # Update voxels
        new_voxels = voxels - step_size * grad * scaling
        new_voxels = np.clip(new_voxels, 0.0, 1.0)
        
        # Update stiffness matrix
        K, delta = femsolver.update_global_stiffness_matrix(K, old_voxels, new_voxels, Ke, threshold)
        
        old_voxels = voxels.copy()
        voxels = new_voxels
        
        # Print progress
        if it % 10 == 0:
            n_discrete = np.sum((voxels < 0.1) | (voxels > 0.9))
            print(f"Iter {it:3d}: obj={obj_val:.2e}, discrete={n_discrete}/{voxels.size} "
                  f"({100*n_discrete/voxels.size:.1f}%), "
                  f"density_mean={voxels.mean():.3f}")
    
    return voxels, K

def print_density_stats(voxels):
    """Print statistics about density distribution."""
    n_zero = np.sum(voxels < 0.1)
    n_one = np.sum(voxels > 0.9)
    n_intermediate = np.sum((voxels >= 0.1) & (voxels <= 0.9))
    
    print(f"Density statistics:")
    print(f"  Near 0 (< 0.1): {n_zero}/{voxels.size} ({100*n_zero/voxels.size:.1f}%)")
    print(f"  Near 1 (> 0.9): {n_one}/{voxels.size} ({100*n_one/voxels.size:.1f}%)")
    print(f"  Intermediate:   {n_intermediate}/{voxels.size} ({100*n_intermediate/voxels.size:.1f}%)")
    print(f"  Min: {voxels.min():.3f}, Max: {voxels.max():.3f}, Mean: {voxels.mean():.3f}")
    print(f"  Std: {voxels.std():.3f}")

if __name__ == "__main__":
    voxels = run_discrete_optimization()
    print("\nFinal voxel densities:")
    print(voxels)
    print(f"\nOptimization complete. Use femplotter to visualize the result.")
