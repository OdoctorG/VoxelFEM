"""
Simple gradient computation without adjoint method for debugging.
"""

import femsolver_cont as femsolver
import numpy as np

def obj_1_grad_simple(voxels, B, Ke, lambda_, fixed_nodes):
    """
    Compute gradient of obj_1 using direct finite differences.
    This is slow but correct, for debugging purposes.
    """
    E = 200e9
    nu = 0.3
    L = 0.01
    
    height, width = voxels.shape
    grad = np.zeros((height, width))
    
    # Get the force vector
    K = femsolver.global_stiffness_matrix(Ke, voxels)
    n_dofs = K.shape[0]
    F = np.zeros((n_dofs, 1))
    F = femsolver.add_force_to_node(4, F, np.array([0.5, 0.5]))
    
    # Solve baseline
    solver = femsolver.Solver(K, F)
    u, _ = solver.solve(K, F, fixed_nodes)
    eps = femsolver.get_element_strains_fast(u, voxels, L)
    sigma = femsolver.get_element_stresses_fast(eps, E, nu)
    n_sigma = femsolver.get_node_values_fast(sigma, voxels, L)
    von_mises = femsolver.von_mises_stresses_node(n_sigma)
    
    from gradient import obj_1, reg_grad
    f0 = obj_1(von_mises, voxels, B, lambda_)
    
    # Compute gradient for each voxel using finite differences
    eps_fd = 1e-7
    for i in range(height):
        for j in range(width):
            if voxels[i, j] == 0:
                grad[i, j] = 0
                continue
                
            # Forward perturbation
            vox_plus = voxels.copy().astype(np.float64)
            vox_plus[i, j] += eps_fd
            K_plus = femsolver.global_stiffness_matrix(Ke, vox_plus)
            solver_plus = femsolver.Solver(K_plus, F)
            u_plus, _ = solver_plus.solve(K_plus, F, fixed_nodes=fixed_nodes)
            eps_plus = femsolver.get_element_strains_fast(u_plus, vox_plus, L)
            sigma_plus = femsolver.get_element_stresses_fast(eps_plus, E, nu)
            n_sigma_plus = femsolver.get_node_values_fast(sigma_plus, vox_plus, L)
            von_mises_plus = femsolver.von_mises_stresses_node(n_sigma_plus)
            f_plus = obj_1(von_mises_plus, vox_plus, B, lambda_)
            
            # Backward perturbation  
            vox_minus = voxels.copy().astype(np.float64)
            vox_minus[i, j] -= eps_fd
            K_minus = femsolver.global_stiffness_matrix(Ke, vox_minus)
            solver_minus = femsolver.Solver(K_minus, F)
            u_minus, _ = solver_minus.solve(K_minus, F, fixed_nodes=fixed_nodes)
            eps_minus = femsolver.get_element_strains_fast(u_minus, vox_minus, L)
            sigma_minus = femsolver.get_element_stresses_fast(eps_minus, E, nu)
            n_sigma_minus = femsolver.get_node_values_fast(sigma_minus, vox_minus, L)
            von_mises_minus = femsolver.von_mises_stresses_node(n_sigma_minus)
            f_minus = obj_1(von_mises_minus, vox_minus, B, lambda_)
            
            grad[i, j] = (f_plus - f_minus) / (2 * eps_fd)
            
    return grad
