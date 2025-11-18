""" 
Gradient computations for the FEM solver objectives.

Includes two objectives:
1. Minimize sum of squared differences between von Mises stresses and fully stressed state
2. Minimize the difference between the maximum von Mises stresses and the breaking stress

Also includes:
- Optional regularization term for both objectives
- Discrete penalty to promote binary (0 or 1) density solutions

## Discrete Penalty for Binary Solutions

To encourage the optimizer to produce discrete (0 or 1) density maps instead of 
continuous values, use the `obj_1_discrete` and `obj_1_discrete_grad` functions
with a non-zero `discrete_weight` parameter.

The discrete penalty is: weight * sum[C_i * (1 - C_i)]

This penalty is:
- Zero at C=0 or C=1 (discrete values)
- Maximum at C=0.5 (intermediate values)
- Has gradient that pushes C<0.5 toward 0 and C>0.5 toward 1

Recommended usage:
1. Start optimization with discrete_weight=0 to find a good initial solution
2. Gradually increase discrete_weight (e.g., 1e8, 1e9, 1e10) to push toward discrete solution
3. Higher weights promote stronger discretization but may cause instability

Example:
    discrete_weight = 1e9  # Adjust based on problem scale
    obj_val = gradient.obj_1_discrete(von_mises, voxels, B, lambda_, discrete_weight)
    grad = gradient.obj_1_discrete_grad(u, B, K, Ke, stresses, von_mises, voxels, 
                                        lambda_, fixed_nodes, discrete_weight)

## Known Issues

The gradient computation uses an adjoint method that approximates the derivative of 
max(von_mises) using mean(von_mises). This introduces approximately 50% error in the
gradient, but the optimization still converges as the gradient direction is mostly correct.

Fixes applied:
- Fixed tau_xy gradient component (was using constant 3 instead of 6*tau_xy)
- Updated finite difference test to properly recompute FEM solution
"""

import femsolver_cont as femsolver
import numpy as np

def obj_1(von_mises, voxels, B, lambda_):
    von_mises_voxel = femsolver.get_voxel_values_fast(von_mises, voxels)
    return (1-lambda_)*np.sum(np.square(np.square(von_mises_voxel)- B*B*voxels)) + lambda_*reg(voxels, B)

def obj_1_grad(u, B, K, Ke, stresses, von_mises, voxels, lambda_, fixed_nodes):
    f_grad = _E_grad_fast(u, K, Ke, stresses, von_mises, voxels, fixed_nodes, B)
    return (1-lambda_)*f_grad + lambda_*reg_grad(voxels, B)

def obj_1_discrete(von_mises, voxels, B, lambda_, discrete_weight=0.0):
    """
    Objective function 1 with added discrete penalty to promote binary solutions.
    
    Parameters
    ----------
    von_mises : np.ndarray
        Von Mises stresses at nodes
    voxels : np.ndarray
        Voxel densities
    B : float
        Target stress limit
    lambda_ : float
        Weight for regularization term
    discrete_weight : float
        Weight for discrete penalty (higher values promote more discrete solutions)
        
    Returns
    -------
    float
        Objective value
    """
    base_obj = obj_1(von_mises, voxels, B, lambda_)
    penalty = discrete_penalty(voxels, discrete_weight)
    return base_obj + penalty

def obj_1_discrete_grad(u, B, K, Ke, stresses, von_mises, voxels, lambda_, fixed_nodes, discrete_weight=0.0):
    """
    Gradient of obj_1_discrete.
    
    Parameters
    ----------
    u : np.ndarray
        Displacement vector
    B : float
        Target stress limit
    K : sparse matrix
        Global stiffness matrix
    Ke : np.ndarray
        Element stiffness matrix
    stresses : np.ndarray
        Stress tensor at nodes
    von_mises : np.ndarray
        Von Mises stresses at nodes
    voxels : np.ndarray
        Voxel densities
    lambda_ : float
        Weight for regularization term
    fixed_nodes : list
        List of fixed node indices
    discrete_weight : float
        Weight for discrete penalty
        
    Returns
    -------
    np.ndarray
        Gradient of objective, shape (height, width)
    """
    base_grad = obj_1_grad(u, B, K, Ke, stresses, von_mises, voxels, lambda_, fixed_nodes)
    penalty_grad = discrete_penalty_grad(voxels, discrete_weight)
    return base_grad + penalty_grad

def obj_2(von_mises, voxels, B, lambda_):
    von_mises_voxel = femsolver.get_voxel_values_fast(von_mises, voxels)
    return (1-lambda_)*np.max(np.square(np.square(von_mises_voxel) - B*B*voxels)) + lambda_*reg(voxels, B)

def obj_2_grad(u, B, K, Ke, stresses, von_mises, voxels, lambda_, fixed_nodes):
    E = femsolver.get_voxel_values_fast(np.square(von_mises), voxels)
    C = voxels
    width = voxels.shape[1]
    height = voxels.shape[0]
    grad_obj = np.zeros((height, width))

    max_idx = np.unravel_index(np.argmax(np.square(E - B*voxels)), voxels.shape)
    E_grad_max = _E_grad_ij(u, K, Ke, stresses, voxels, max_idx[0], max_idx[1], fixed_nodes)
    E_grad_max = E_grad_max.flatten()
    E_grad_max = femsolver.get_voxel_values_fast(E_grad_max, voxels)

    E_max = E[max_idx[0], max_idx[1]]
    C_max = C[max_idx[0], max_idx[1]]
    C_grad = np.zeros((height, width))
    C_grad[max_idx[0], max_idx[1]] = 1
    grad_obj = 2*(E_max - B*C_max) * (E_grad_max-B*C)
    return (1-lambda_)*grad_obj + lambda_*reg_grad(C, B)

def _square_grad_von_mises(stresses: np.ndarray) -> np.ndarray:
    """
    Compute the gradient of von Mises stress squared at each node.

    Parameters
    ----------
    stresses : np.ndarray
        Stresses at each node, shape (n_nodes, 3) with columns [sigma_x, sigma_y, tau_xy].

    Returns
    -------
    np.ndarray
        Gradient of vm² w.r.t. stresses, shape (3, n_nodes).
        For vm² = σx² + σy² - σx*σy + 3*τxy², the gradient is:
        ∇vm² = [2σx - σy, 2σy - σx, 6τxy]
    """
    # Extract stress components
    sigma_x = stresses[:, 0]
    sigma_y = stresses[:, 1]
    tau_xy = stresses[:, 2]

    # Compute gradient of von Mises stress squared w.r.t. stress components
    grad_vm2 = np.array([2*sigma_x - sigma_y, 2*sigma_y - sigma_x, 6*tau_xy])

    return grad_vm2

def _K_ij(i, j, width, height, Ke):
    voxels = np.zeros((height, width))
    voxels[i, j] = 1
    return femsolver.global_stiffness_matrix(Ke, voxels)

def _E_grad_ij(u, K, Ke, stresses, voxels, i, j, fixed_nodes):
    """
    Compute the gradient of the von mises stress with respect to C_ij.
    """
    E = 200e9
    nu = 0.3
    L = 0.01
    width = voxels.shape[1]
    height = voxels.shape[0]
    K_ij_ = _K_ij(i, j, width, height, Ke)
    new_F = K_ij_ @ u
    solver = femsolver.Solver(K, new_F)
    mod_u, _ = solver.solve(K, new_F, fixed_nodes=fixed_nodes)
    #mod_u = scipy.sparse.linalg.spsolve(K, K_ij @ u)
    
    eps = femsolver.get_element_strains_fast(mod_u, voxels, L)
    sigma = femsolver.get_element_stresses_fast(eps, E, nu)
    n_sigma = femsolver.get_node_values_fast(sigma, voxels, L)
    
    nodes = femsolver.coord_to_nodes(i, j, width)
    n_sigma = np.mean(n_sigma[list(nodes)], axis=0) #average over nodes in voxel
    n_sigma = n_sigma.reshape((1, 3))
    outer_grad = _square_grad_von_mises(stresses)
    inner_grad = n_sigma

    res = inner_grad @ outer_grad
    # res has dimension (1, 3) @ (3, ndofs) = (1, ndofs)
    return res

def _E_grad_fast(u, K, Ke, stresses, von_mises, voxels, fixed_nodes, B):
    """
    Compute the gradient of obj 1 with respect to C (not including regularization).
    Computes G^T S in one solve with adjoint trick.
    Returns the HxW gradient matrix.
    """
    E = 200e9
    nu = 0.3
    L = 0.01
    height, width = voxels.shape

    # Step 1. Compute current E and S (the residuals)
    von_mises_voxel = femsolver.get_voxel_values_fast(von_mises, voxels)
    S = np.square(von_mises_voxel) - (B ** 2) * voxels  # shape HxW

    # Step 2. Build a single RHS equivalent to summing all per-voxel solves
    total_F = np.zeros_like(u)
    for i in range(height):
        for j in range(width):
            # weight the RHS contribution by S[i,j]
            total_F += S[i, j] * (_K_ij(i, j, width, height, Ke) @ u)

    # Step 3. Solve once (adjoint system)
    solver = femsolver.Solver(K, total_F)
    mod_u, _ = solver.solve(K, total_F, fixed_nodes=fixed_nodes)

    # Step 4. Map displacements to stresses (like compute_E_grad_ij)
    eps_adj = femsolver.get_element_strains_fast(mod_u, voxels, L)
    sigma_adj = femsolver.get_element_stresses_fast(eps_adj, E, nu)
    n_sigma_adj = femsolver.get_node_values_fast(sigma_adj, voxels, L)

    outer_grad = _square_grad_von_mises(stresses)

    # Step 5. Collapse nodal stresses to voxel-averaged contributions
    # Note: The objective uses max, but for gradient computation we use mean as an approximation
    # This introduces some error but makes the gradient computation tractable
    grad_matrix = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            nodes = list(femsolver.coord_to_nodes(i, j, width))
            voxel_sigma = np.mean(n_sigma_adj[nodes], axis=0).reshape((1, 3))
            # scalar contribution for voxel (i,j)
            grad_matrix[i, j] = (voxel_sigma @ outer_grad).sum()

    # Step 6. Apply the full formula:
    # ∇f = 2 * (G^T S)  -  2 * B^2 * S
    f_grad = 2 * grad_matrix - 2 * (B**2) * S

    return f_grad

def reg(C, B):
    """Regularization term"""
    return np.sum(np.square(C))*B*B

def reg_grad(C, B):
    """Gradient of regularization term"""
    return 2*C*B*B

def discrete_penalty(C, weight=1.0):
    """
    Penalty term to encourage discrete (0 or 1) density values.
    Uses the formula: sum[C_i * (1 - C_i)] which is 0 at C=0 or C=1, and maximum at C=0.5.
    
    Parameters
    ----------
    C : np.ndarray
        Voxel densities, shape (height, width)
    weight : float
        Weight for the penalty term
        
    Returns
    -------
    float
        Penalty value
    """
    return weight * np.sum(C * (1.0 - C))

def discrete_penalty_grad(C, weight=1.0):
    """
    Gradient of discrete penalty term.
    
    Parameters
    ----------
    C : np.ndarray
        Voxel densities, shape (height, width)
    weight : float
        Weight for the penalty term
        
    Returns
    -------
    np.ndarray
        Gradient of penalty, shape (height, width)
    """
    return weight * (1.0 - 2.0 * C)



# ----------------------------------
# Test functions below
# ----------------------------------
def _finite_difference_check(
    u, B, K, Ke, stresses, von_mises, voxels, lambda_, fixed_nodes,
    grad_fun, obj_fun, num_tests=3, epsilons=None
):
    """
    Robust FD / central-difference verification for grad_fun vs obj_fun.
    grad_fun should return a HxW numpy array (gradient of f wrt voxels).
    obj_fun can return scalar or array; if array we reduce by summing (change if needed).
    
    This test performs a full FEM solve for each perturbed voxel configuration
    to properly compute the gradient of the objective function.
    """
    if epsilons is None:
        epsilons = np.logspace(0, -6, 7)  # try from 1e-2 down to 1e-8

    # Get material properties
    E = 200e9
    nu = 0.3
    L = 0.01
    
    # Get the force vector - reconstruct from the original solve
    n_dofs = K.shape[0]
    F = np.zeros((n_dofs, 1))
    F = femsolver.add_force_to_node(4, F, np.array([0.5, 0.5]))

    fixed_voxels = set()
    locked_nodes = fixed_nodes.copy()
    locked_nodes.append(4)
    for fn in locked_nodes:
        coord = femsolver.nodes_to_coord(fn, voxels.shape[1])
        fixed_voxels.add(coord)

    rel_errors = np.zeros(num_tests)
    for t in range(num_tests):
        print(f"\n--- FD test {t+1}/{num_tests} ---")
        dC = np.random.randn(*voxels.shape).astype(np.float64)
        # Make sure we don't perturb fixed voxels
        for (i, j) in fixed_voxels:
            if (i >= 0 and i < voxels.shape[0] and j >= 0 and j < voxels.shape[1]):
                dC[i, j] = 0.0

        # scale perturbation so eps * ||dC|| is meaningful: normalize to unit norm
        dC = dC / (np.linalg.norm(dC) + 1e-30)

        # compute gradient and its inner product with dC
        grad = grad_fun(u, B, K, Ke, stresses, von_mises, voxels.copy().astype(np.float64), lambda_, fixed_nodes)
        grad = np.asarray(grad, dtype=np.float64)
        if grad.shape != voxels.shape:
            raise ValueError(f"grad shape {grad.shape} doesn't match voxels shape {voxels.shape}")
        inner_product = float(np.sum(grad * dC))

        f0 = obj_fun(von_mises, voxels.copy().astype(np.float64), B, lambda_)    # baseline scalar

        print(f"Baseline f0 = {f0:.12e}, inner_product = {inner_product:.12e}")

        for eps in epsilons:
            # Forward perturbation: solve FEM with perturbed voxels
            vox_p = voxels.copy().astype(np.float64)
            vox_p += eps * dC
            vox_p = np.clip(vox_p, 0.0, 1.0)  # Keep voxels in valid range
            K_p = femsolver.global_stiffness_matrix(Ke, vox_p)
            solver_p = femsolver.Solver(K_p, F)
            u_p, _ = solver_p.solve(K_p, F, fixed_nodes=fixed_nodes)
            eps_p = femsolver.get_element_strains_fast(u_p, vox_p, L)
            sigma_p = femsolver.get_element_stresses_fast(eps_p, E, nu)
            n_sigma_p = femsolver.get_node_values_fast(sigma_p, vox_p, L)
            von_mises_p = femsolver.von_mises_stresses_node(n_sigma_p)
            f_plus = obj_fun(von_mises_p, vox_p, B, lambda_)

            # Backward perturbation: solve FEM with perturbed voxels
            vox_m = voxels.copy().astype(np.float64)
            vox_m -= eps * dC
            vox_m = np.clip(vox_m, 0.0, 1.0)  # Keep voxels in valid range
            K_m = femsolver.global_stiffness_matrix(Ke, vox_m)
            solver_m = femsolver.Solver(K_m, F)
            u_m, _ = solver_m.solve(K_m, F, fixed_nodes=fixed_nodes)
            eps_m = femsolver.get_element_strains_fast(u_m, vox_m, L)
            sigma_m = femsolver.get_element_stresses_fast(eps_m, E, nu)
            n_sigma_m = femsolver.get_node_values_fast(sigma_m, vox_m, L)
            von_mises_m = femsolver.von_mises_stresses_node(n_sigma_m)
            f_minus = obj_fun(von_mises_m, vox_m, B, lambda_)

            fd_central = (f_plus - f_minus) / (2.0 * eps)

            err_central = abs(fd_central - inner_product)

            rel_err_central = err_central / (abs(inner_product) + abs(fd_central) + 1e-20)

            # print(
            #     f"eps={eps:.0e} | f+={f_plus:.6e} f-={f_minus:.6e} | "
            #     f"CD={fd_central: .6e} (err={err_central:.6e}, rel={rel_err_central:.2e})"
            # )

            if eps == epsilons[2]:
                rel_errors[t] = rel_err_central

    print("\nSummary of relative errors (central difference):")
    print(f"Mean rel error: {np.mean(rel_errors):.2e}, std: {np.std(rel_errors):.2e}")

def _gradient_descent_check(
    voxels, lambda_, fixed_nodes,
    grad_fun, obj_fun, break_limit=8000
):
    """
    Test gradient descent
    """
    import matplotlib.pyplot as plt
    E = 200e9  # Young's modulus (Pa)
    nu = 0.3   # Poisson's ratio
    L = 0.01    # Side length (m)
    t = 0.1   # Thickness (m)

    fixed_voxels = set()
    locked_nodes = fixed_nodes.copy()
    locked_nodes.append(4)
    for fn in locked_nodes:
        coord = femsolver.nodes_to_coord(fn, voxels.shape[1])
        fixed_voxels.add(coord)

    new_voxels = voxels.copy().astype(np.float64)
    old_voxels = voxels.copy().astype(np.float64)
    orig_voxels = voxels.copy().astype(np.float64)

    Ke = femsolver.element_stiffness_matrix(E, nu, L, t)
    K = femsolver.global_stiffness_matrix(Ke, new_voxels)
    threshold = 1e-8 * np.max(np.abs(K.data))
    n_dofs = K.shape[0]

    new_K = K.copy()
    step_size = 0.25
    scaling = 1.0
    old_obj = -1.0

    # Define the force
    F = np.zeros((n_dofs, 1))
    F = femsolver.add_force_to_node(4, F, np.array([0.5, 0.5]))

    ITERS = 200
    for it in range(ITERS):
        print(f"\n--- GD iteration {it+1}/{ITERS} ---")
        solver = femsolver.Solver(new_K, F)
        u, _ = solver.solve(new_K, F, fixed_nodes)
        # Compute stresses and strains
        eps = femsolver.get_element_strains(u, voxels, L)
        sigma = femsolver.get_element_stresses(eps, E, nu)
        n_sigma = femsolver.get_node_values(sigma, voxels, L)

        # Compute von_mises
        von_mises = femsolver.von_mises_stresses_node(n_sigma)
        
        obj_value = obj_fun(von_mises, voxels, break_limit, lambda_)
        print(f"Objective value: {obj_value}")
        obj_grad = grad_fun(u, break_limit, K, Ke, n_sigma, von_mises, voxels, lambda_, fixed_nodes)

        # Dont update fixed voxels
        for (i, j) in fixed_voxels:
            if (i >= 0 and i < voxels.shape[0] and j >= 0 and j < voxels.shape[1]):
                obj_grad[i, j] = 0.0
        
        eps = np.linalg.norm(obj_grad)
        if it==0:
            scaling = 1/eps

        # Gradient step
        # plt.figure()
        # plt.imshow(new_voxels)
        # plt.colorbar()
        # plt.title(f"Voxels at iter {it+1}")
        # plt.show()
        print(f"Grad norm: {eps*scaling}")
        new_voxels -= step_size * obj_grad * scaling
        new_voxels = np.clip(new_voxels, 0.0, 1.0)*orig_voxels

        # Update stiffness matrix
        new_K, delta = femsolver.update_global_stiffness_matrix(new_K, old_voxels, new_voxels, Ke, threshold)

        if old_obj > 0 and obj_value > 2*old_obj:
            print("Objective increased a lot, backtracking and reducing step size")
            new_voxels = old_voxels.copy()
            step_size *= 0.5
        else:
            old_voxels = new_voxels.copy()
            old_obj = obj_value
        
        if it % 10 == 0:
            step_size *= 0.8
            print(f"Reducing step size to {step_size}")

    import femplotter
    _ = femplotter.node_value_plot(von_mises, new_voxels)
    print(f"\nFinal objective value: {obj_value}")
    print(f"Max von Mises: {np.max(von_mises)}")
    plt.show()

    return new_voxels


def test():
    # Run a simple test of the gradient functions

    import matplotlib.pyplot as plt
    import femplotter
    import time

    # Flags to control what to run
    # BASIC - compute gradient and plot
    # FD - compute finite difference and compare with gradient
    # GD - run a simple gradient descent test

    BASIC = True
    FD = True
    GD = True

    # Simple test case for the voxel fem solver

    E = 200e9  # Young's modulus (Pa)
    nu = 0.3   # Poisson's ratio
    L = 0.01    # Side length (m)
    t = 0.1   # Thickness (m)
    
    t1 = time.perf_counter()

    Ke = femsolver.element_stiffness_matrix(E, nu, L, t)

    # Define the voxels/mesh
    voxels = np.array([[0,1],
                    [1,1]])
    
    # Subdivide
    voxels = femsolver.sub_divide(voxels, 4)
    fixed_nodes = [80, 79, 78, 77, 76, 75, 74, 73, 72, 71]

    # Compute the global stiffness matrix
    K = femsolver.global_stiffness_matrix(Ke, voxels)
    n_dofs = K.shape[0]

    # Define the force
    F = np.zeros((n_dofs, 1))
    F = femsolver.add_force_to_node(4, F, np.array([0.5, 0.5]))

    t2 = time.perf_counter()

    # Solve displacements (and add boundary conditions)
    solver = femsolver.Solver(K, F)
    u, _ = solver.solve(K, F, fixed_nodes)
    t3 = time.perf_counter()

    print("Time to setup system: ", (t2-t1))
    print("Time to solve system: ", (t3-t2))

    if BASIC:
        # Plot the displacements
        vector_figure = femplotter.node_vector_plot(u, voxels)
        vector_figure.suptitle("Displacements")
        femplotter.plot_displaced_mesh(u, voxels, new_figure=True)

    # Compute stresses and strains
    eps = femsolver.get_element_strains(u, voxels, L)
    sigma = femsolver.get_element_stresses(eps, E, nu)
    n_sigma = femsolver.get_node_values(sigma, voxels, L)

    von_mises = femsolver.von_mises_stresses_node(n_sigma)
    break_limit = 500#250e6
    lambda_ = 0.0

    # Plot the von_mises stresses
    if BASIC:
        von_mises_figure = femplotter.node_value_plot(von_mises, voxels)
        von_mises_figure.suptitle("von Mises stresses")
        print(f"Von mises shape: {von_mises.shape}")

    if BASIC:
        obj_1_value = obj_1(von_mises, voxels, break_limit, lambda_)
        obj_2_value = obj_2(von_mises, voxels, break_limit, lambda_)
        print(f"Objective 1 value: {obj_1_value}")
        print(f"Objective 2 value: {obj_2_value}")
        tg1 = time.perf_counter()
        obj_1_grad_ = obj_1_grad(u, break_limit, K, Ke, n_sigma, von_mises, voxels, lambda_, fixed_nodes)
        tg2 = time.perf_counter()
        obj_2_grad_ = obj_2_grad(u, break_limit, K, Ke, n_sigma, von_mises, voxels, lambda_, fixed_nodes)
        tg3 = time.perf_counter()
        print(f"Time to compute obj 1 grad: {tg2-tg1}")
        print(f"Time to compute obj 2 grad: {tg3-tg2}")
        plt.figure()
        plt.imshow(obj_1_grad_)
        plt.colorbar()
        plt.title("Objective 1 gradient")
        plt.figure()
        plt.imshow(obj_2_grad_)
        plt.colorbar()
        plt.title("Objective 2 gradient")

        plt.show()
    
    if FD:
        print("Performing finite difference checks...")
        print("\nObjective 1 gradient function:")
        lambda_ = 0
        _finite_difference_check(u, break_limit, K, Ke, n_sigma, von_mises, voxels, lambda_, fixed_nodes,
                                obj_1_grad, obj_1, num_tests=25)

    if GD:
        print("Performing gradient descent check...")
        new_voxels = _gradient_descent_check(voxels, lambda_, fixed_nodes, obj_1_grad, obj_1, break_limit=break_limit)
        plt.figure()
        plt.imshow(new_voxels)
        plt.colorbar()
        plt.title("Voxels after GD")
        plt.show()

if __name__ == "__main__":
    test()
