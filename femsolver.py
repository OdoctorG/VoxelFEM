""" 
Voxel 2D FEM Solver 

Uses square elements with four nodes and bilinear form functions.
The element is defined as follows:
N1 --- N2
|       |
|       |
N3 --- N4

Form functions:
N₁ = (1 - ξ)(1 - η) / 4
N₂ = (1 + ξ)(1 - η) / 4
N₃ = (1 - ξ)(1 + η) / 4
N₄ = (1 + ξ)(1 + η) / 4

"""

import math
import numpy as np
import scipy.signal
import scipy.sparse
import scipy.sparse.linalg
import time

import os
os.add_dll_directory("C://Users/greno/miniforge3/Library/bin")
from sksparse.cholmod import cholesky

# Number of Gauss points, if set to 1 you get very bad results. 2 and up recommended
NGAUSS = 2

def B_matrix(xi, eta, L):
    """
    Generate the B-matrix for a square element with side length L.
    
    :param xi: Natural coordinate xi (-1 to 1)
    :param eta: Natural coordinate eta (-1 to 1)
    :param L: Side length of the square element
    :return: B-matrix as a NumPy array
    """
    B = np.array([
        [-(1-eta)/(2*L),  0,              (1-eta)/(2*L),   0,              -(1+eta)/(2*L),  0,              (1+eta)/(2*L),   0            ],
        [0,              -(1-xi)/(2*L),   0,              -(1+xi)/(2*L),   0,               (1-xi)/(2*L),   0,               (1+xi)/(2*L) ],
        [-(1-xi)/(2*L),  -(1-eta)/(2*L),  -(1+xi)/(2*L),  (1-eta)/(2*L),   (1-xi)/(2*L),   -(1+eta)/(2*L),  (1+xi)/(2*L),    (1+eta)/(2*L)]
    ])
    
    return B


def shape_function(xi: float, eta: float) -> float:
    N = np.array([(1 - xi)*(1 - eta), (1 + xi)*(1 - eta), (1 - xi)*(1 + eta), (1 + xi)*(1 + eta)])

    return N

def gauss_points(N: int) -> tuple[list, list]:
    """
    Get the Gauss points and weights for N-point Gaussian quadrature integration.
    
    Parameters
    ----------
    N : int
        Number of Gauss points to compute (N = 1, 2, 3, 4)
        
    Returns
    -------
    tuple[list, list]
        A tuple with the Gauss points and weights. If N is not in the range 1-4, returns None, None.
    """

    if N == 1:
        return ([0], [2])
    if N == 2:
        return ([-1/np.sqrt(3), 1/np.sqrt(3)], [1,1])
    if N == 3:
        return ([-np.sqrt(3/5), 0, np.sqrt(3/5)], [5/9, 8/9, 5/9])
    if N == 4:
        return ([0.339981633, -0.339981633, 0.861136311, -0.861136311], [0.652145, 0.652145, 0.347855, 0.347855])
    return (None, None)

def element_stiffness_matrix(E: float, nu: float, L, t) -> np.ndarray:
    """
    Compute the 8x8 element stiffness matrix for the square element, given material properties E and nu.

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    L : float
        Side length of the element.
    t : float
        Thickness of the element.

    Returns
    -------
    K : np.ndarray
        The element stiffness matrix.
    """

    # Material property matrix for plane stress
    D = (E / (1 - nu**2)) * np.array([[1, nu, 0],
                                    [nu, 1, 0],
                                    [0, 0, (1-nu)/2]])
    
    # Gauss points and weights
    gauss_p, w = gauss_points(NGAUSS)
    
    K = np.zeros((8, 8))

    xi_idx = 0
    for xi in gauss_p:
        eta_idx = 0
        for eta in gauss_p:
            # Compute B matrix at Gauss point
            B = B_matrix(xi, eta, L)
            
            # Compute Jacobian determinant
            J = L**2 / 4
            
            # Accumulate contribution to stiffness matrix
            K += w[xi_idx] * w[eta_idx] * np.dot(np.dot(B.T, D), B) * J * t
            eta_idx += 1
        xi_idx += 1
    return K

def get_element_strains(u: np.ndarray, voxels: np.ndarray, L: float) -> np.ndarray:
    """
    Compute strains at each gauss point in each element, given the displacements at each node.

    Parameters
    ----------
    u : np.ndarray
        Displacements at each node.
    voxels : np.ndarray
        2D array of voxels, where 1 indicates a solid voxel and 0 indicates a void voxel.
    L : float
        Side length of the element.

    Returns
    -------
    np.ndarray
        Strains at each element.
    """
    H = voxels.shape[0] # Number of rows
    W = voxels.shape[1] # Number of columns

    # Gauss points and weights
    gauss_p, w = gauss_points(NGAUSS)
    n_gauss_points = len(gauss_p)

    strains = np.zeros((voxels.shape[0]*n_gauss_points,voxels.shape[1]*n_gauss_points,3), dtype=object)
    for i in range(voxels.shape[0]):
        for j in range(voxels.shape[1]):

            if voxels[i, j] == 1:
                # Convert voxel coordinates to node indices
                nodes = coord_to_nodes(i, j, W)
                # Get nodal displacements
                el_u = np.zeros(8)
                for k in range(len(nodes)):
                    el_u[2*k] = u[2*nodes[k]]
                    el_u[2*k+1] = u[2*nodes[k]+1]

                # Loop over all Gauss points
                xi_idx = 0
                for xi in gauss_p:
                    eta_idx = 0
                    for eta in gauss_p:
                        # Compute B matrix at Gauss point
                        B = B_matrix(xi, eta, L)
                
                        # Compute Jacobian determinant
                        J = 0.25
                        # Compute strains at Gauss point
                        strains[n_gauss_points*i+eta_idx,n_gauss_points*j+xi_idx][0] += w[xi_idx] * w[eta_idx] * np.dot(B[0], el_u) * J
                        strains[n_gauss_points*i+eta_idx,n_gauss_points*j+xi_idx][1] += w[xi_idx] * w[eta_idx] * np.dot(B[1], el_u) * J
                        strains[n_gauss_points*i+eta_idx,n_gauss_points*j+xi_idx][2] += w[xi_idx] * w[eta_idx] * np.dot(B[2], el_u) * J

                        eta_idx += 1
                    xi_idx += 1
        
    return strains


def get_element_strains_fast(u: np.ndarray, voxels: np.ndarray, L: float) -> np.ndarray:
    """
    Compute strains at each Gauss point using vectorized operations.
    
    Parameters
    ----------
    u : np.ndarray
        Nodal displacements (1D array).
    voxels : np.ndarray
        2D array indicating solid (1) and void (0) voxels.
    L : float
        Element side length.

    Returns
    -------
    np.ndarray
        Strains at each Gauss point (shape: [H*NGAUSS, W*NGAUSS, 3]).
    """
    H, W = voxels.shape
    n_g = NGAUSS
    gauss_p, w = gauss_points(n_g)
    n_g_sq = n_g ** 2

    # Precompute scaled B matrices for all Gauss points
    B_scaled = np.zeros((n_g_sq, 3, 8))
    for gp_idx in range(n_g_sq):
        xi_idx, eta_idx = divmod(gp_idx, n_g)
        xi = gauss_p[xi_idx]
        eta = gauss_p[eta_idx]
        B = B_matrix(xi, eta, L)
        B_scaled[gp_idx] = B * (w[xi_idx] * w[eta_idx] * 0.25)  # Include weights and J

    # Get solid element indices
    solid_i, solid_j = np.where(voxels == 1)
    n_solid = len(solid_i)
    if n_solid == 0:
        return np.zeros((H*n_g, W*n_g, 3))

    # Get nodal DOF indices for all solid elements
    nodes = coord_to_nodes_vectorized(solid_i, solid_j, W)
    dof_indices = np.stack([2*nodes, 2*nodes+1], axis=2).reshape(n_solid, 8)
    u_elements = u[dof_indices]  # Shape: (n_solid, 8)

    # Compute all strains in one shot using Einstein summation
    strains_all = np.einsum('gab,sb->sga', B_scaled, u_elements)  # Shape: (n_solid, n_g_sq, 3)

    # Initialize output array and assign values
    strains = np.zeros((H*n_g, W*n_g, 3))
    for gp_idx in range(n_g_sq):
        eta_idx, xi_idx = divmod(gp_idx, n_g)  # xi varies faster in original loop
        rows = solid_i * n_g + eta_idx
        cols = solid_j * n_g + xi_idx
        strains[rows, cols, :] = strains_all[:, gp_idx, :]

    return strains

def get_node_values(element_s: np.ndarray, voxels: np.ndarray, L: float) -> np.ndarray:
    """
    Compute the average strain/stress at each node in the mesh, given the strains/stresses at each gauss point in each element.

    Parameters
    ----------
    element_s : np.ndarray
        Strains or stresses at each element.
    voxels : np.ndarray
        2D array of voxels, where 1 indicates a solid voxel and 0 indicates a void voxel.
    L : float
        Side length of the element.

    Returns
    -------
    np.ndarray
        Average strains/stresses at each node.
    """
    W = voxels.shape[1] # Number of columns

    # Gauss points and weights
    gauss_p, w = gauss_points(NGAUSS)
    n_gauss_points = len(gauss_p)

    strains = np.zeros(((voxels.shape[0]+1)*(voxels.shape[1]+1),3))
    avg_count = np.zeros(((voxels.shape[0]+1)*(voxels.shape[1]+1),1), dtype=int)
    W = voxels.shape[1] # Number of columns

    for i in range(voxels.shape[0]):
        for j in range(voxels.shape[1]):
            if voxels[i, j] == 1:
                nodes = coord_to_nodes(i,j, W)
                # Loop over all Gauss points
                xi_idx = 0
                for xi in gauss_p:
                    eta_idx = 0
                    for eta in gauss_p:
                        elem_strain = element_s[n_gauss_points*i+eta_idx,n_gauss_points*j+xi_idx]

                        N = shape_function(xi, eta)

                        for k in range(len(nodes)):
                            for l in range(3):
                                strains[nodes[k]][l] += N[k] * elem_strain[l]
                                avg_count[nodes[k]][0] += 1
                        eta_idx += 1
                    xi_idx += 1
    
    return strains

def get_node_values_fast(element_s: np.ndarray, voxels: np.ndarray, L: float) -> np.ndarray:
    """
    Compute the average strain/stress at each node in the mesh using vectorized operations.
    """
    H, W = voxels.shape
    n_gauss = NGAUSS
    n_gauss_sq = n_gauss ** 2

    # Precompute shape functions for all Gauss points in correct xi/eta order
    gauss_p = gauss_points(n_gauss)[0]
    xi, eta = np.meshgrid(gauss_p, gauss_p, indexing='ij')  # xi varies first
    xi = xi.flatten()  # Order: xi_0, xi_0, ..., xi_1, ...
    eta = eta.flatten()
    N = np.array([shape_function(xi_, eta_) for xi_, eta_ in zip(xi, eta)])  # (n_gauss_sq, 4)

    # Reshape element_s to (H, W, n_gauss, n_gauss, 3) with xi first
    element_s_4d = element_s.reshape(H, n_gauss, W, n_gauss, 3).transpose(0, 2, 3, 1, 4)
    # Now element_s_4d[i,j] is (n_gauss_xi, n_gauss_eta, 3)

    # Get indices of solid elements
    solid_i, solid_j = np.where(voxels == 1)
    num_solid = len(solid_i)
    if num_solid == 0:
        return np.zeros(((H + 1) * (W + 1), 3))

    # Extract strains for all Gauss points in correct xi/eta order
    all_strains = element_s_4d[solid_i, solid_j, :, :, :].reshape(-1, 3)  # (num_solid * n_gauss_sq, 3)

    # Compute node indices for all solid elements
    nodes = coord_to_nodes_vectorized(solid_i, solid_j, W)

    # Expand nodes to match Gauss points and accumulate
    nodes_repeated = nodes.repeat(n_gauss_sq, axis=0)  # (num_solid * n_gauss_sq, 4)
    N_tiled = np.tile(N, (num_solid, 1))  # (num_solid * n_gauss_sq, 4)

    # Initialize strains array
    strains = np.zeros(((H + 1) * (W + 1), 3))
    
    # Accumulate contributions using vectorized operations
    for k in range(4):  # For each node in the element
        node_indices = nodes_repeated[:, k]
        weights = N_tiled[:, k]
        # Vectorized: strains[node_indices] += weights[:, None] * all_strains
        for l in range(3):  # For each strain component
            np.add.at(strains[:, l], node_indices, weights * all_strains[:, l])

    return strains

def get_voxel_values(node_values, voxels):
    # Average over the node value of each corner in each voxel
    # TODO: change to integrate over the gauss points inside the voxel instead.
    voxel_values = np.zeros((voxels.shape[0], voxels.shape[1]))
    for i in range(voxels.shape[0]):
        for j in range(voxels.shape[1]):
            if voxels[i, j] == 1:
                nodes = coord_to_nodes(i, j, voxels.shape[1])
                mean = 0
                vals = [node_values[n] for n in nodes]
                for n in nodes:
                    mean += node_values[n]
                voxel_values[i, j] = np.max(vals)#mean/len(nodes)
    
    return voxel_values

def get_voxel_values_fast(node_values, voxels):
    # Width
    W = voxels.shape[1]
    # Get indices of solid elements
    solid_i, solid_j = np.where(voxels == 1)
    nodes = coord_to_nodes_vectorized(solid_i, solid_j, W)

    voxel_values = np.zeros((voxels.shape[0], voxels.shape[1]))
    voxel_values[solid_i, solid_j] = np.max(node_values[nodes], axis=1) # Use max
    # voxel_values[solid_i, solid_j] = np.mean(node_vals, axis=1)  # Use mean

    return voxel_values


def get_element_stresses(element_strains: np.ndarray, E: float, nu: float) -> np.ndarray:
    """
    Compute stresses at each gauss point in each element, from element strains, given material properties E and nu.

    Parameters
    ----------
    element_strains : np.ndarray
        Strains at each element.
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.

    Returns
    -------
    np.ndarray
        Stresses at each element.
    """

    # Material property matrix for plane stress
    D = (E / (1 - nu**2)) * np.array([[1, nu, 0],
                                    [nu, 1, 0],
                                    [0, 0, (1-nu)/2]])
    
    # Compute stresses
    stresses = np.dot(element_strains, D)

    return stresses

def get_element_stresses_fast(element_strains: np.ndarray, E: float, nu: float) -> np.ndarray:
    """
    Compute stresses at each Gauss point using vectorized operations.
    Optimized for speed by precomputing the material property matrix.

    Parameters
    ----------
    element_strains : np.ndarray
        Strains at each Gauss point (shape: [H*NGAUSS, W*NGAUSS, 3]).
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.

    Returns
    -------
    np.ndarray
        Stresses at each Gauss point (shape: [H*NGAUSS, W*NGAUSS, 3]).
    """
    # Precompute material property matrix for plane stress
    D = (E / (1 - nu**2)) * np.array([[1, nu, 0],
                                      [nu, 1, 0],
                                      [0, 0, (1-nu)/2]])

    # Vectorized computation of stresses
    stresses = np.einsum('...ij,jk->...ik', element_strains, D)

    return stresses


def von_mises_stresses_node(stresses: np.ndarray) -> np.ndarray:
    """
    Compute von Mises stresses from stress tensor at each node

    Parameters
    ----------
    stresses : np.ndarray
        Stresses at each node.

    Returns
    -------
    np.ndarray
        von Mises stresses at each node.
    """
    # Extract stress components
    sigma_x = stresses[:, 0]
    sigma_y = stresses[:, 1]
    tau_xy = stresses[:, 2]

    # Compute von Mises stresses in a vectorized manner
    von_mises = np.sqrt(
        sigma_x**2 + sigma_y**2 - sigma_x * sigma_y + 3 * tau_xy**2
    )

    return von_mises


def coord_to_nodes(i,j, width) -> tuple[int, int, int, int]:
    """Convert voxel coordinates to node indices"""
    return (i*(width+1)+j, i*(width+1)+j+1, (i+1)*(width+1)+j, (i+1)*(width+1)+j+1)

def coord_to_nodes_vectorized(i: np.ndarray, j: np.ndarray, width: int) -> np.ndarray:
    """Vectorized computation of node indices for given voxel coordinates."""
    nodes = np.empty((len(i), 4), dtype=int)
    nodes[:, 0] = i * (width + 1) + j
    nodes[:, 1] = i * (width + 1) + j + 1
    nodes[:, 2] = (i + 1) * (width + 1) + j
    nodes[:, 3] = (i + 1) * (width + 1) + j + 1
    return nodes


def nodes_to_coord(node_idx, width) -> tuple[int, int]:
    """Convert node indices to voxel coordinates"""
    i = node_idx // (width+1)
    j = node_idx % (width+1)
    return (i, j)

def global_stiffness_matrix(Ke: np.ndarray, voxels: np.ndarray) -> scipy.sparse.csr_matrix:
    """
    Compute the global stiffness matrix (2 dofs per node) from the element stiffness matrix and the voxel representation of the geometry. 

    Parameters
    ----------
    Ke : np.ndarray
        Element stiffness matrix.
    voxels : np.ndarray
        2D array of voxels, where 1 indicates a solid voxel and 0 indicates a void voxel.

    Returns
    -------
    np.ndarray
        Global stiffness matrix.

    Notes
    -----
    The function assumes that the voxels are numbered in column-major order, i.e., the first column is 0, 1, 2, ... and the second column is Width, Width+1, Width+2, ...
    """
    dof_per_node = 2
    n_dofs = (voxels.shape[0] + 1) * (voxels.shape[1] + 1) * dof_per_node
    width = voxels.shape[1]

    # Estimate number of non-zero entries
    nnz = np.sum(voxels == 1) * 64  # Each voxel contributes 8x8 stiffness entries

    # Initialize sparse matrix storage
    data = np.zeros(nnz, dtype=float)
    row_indices = np.zeros(nnz, dtype=int)
    col_indices = np.zeros(nnz, dtype=int)

    # Get the indices of the solid voxels
    solid_voxel_indices = np.argwhere(voxels == 1)

    idx = 0
    for i, j in solid_voxel_indices:
        my_nodes = coord_to_nodes(i, j, width)  # Global node indices for this element

        for local_i in range(8):  # Loop over DOFs in element
            global_i = my_nodes[local_i // 2] * 2 + local_i % 2
            for local_j in range(8):
                global_j = my_nodes[local_j // 2] * 2 + local_j % 2
                data[idx] = Ke[local_i, local_j]
                row_indices[idx] = global_i
                col_indices[idx] = global_j
                idx += 1

    # Assemble sparse matrix
    K = scipy.sparse.csr_matrix((data, (row_indices, col_indices)), shape=(n_dofs, n_dofs))
    K.eliminate_zeros()
    return K

def update_global_stiffness_matrix(K_old: scipy.sparse.csr_matrix, old_voxels: np.ndarray, new_voxels: np.ndarray, Ke: np.ndarray, threshold: float) -> tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]:
    """
    Efficiently update the global stiffness matrix for a new voxel configuration by adding/removing contributions from changed elements.

    Parameters
    ----------
    K_old : scipy.sparse.csr_matrix
        The previous global stiffness matrix.
    old_voxels : np.ndarray
        2D array of the previous voxel configuration (1 = solid, 0 = void).
    new_voxels : np.ndarray
        2D array of the updated voxel configuration (1 = solid, 0 = void).
    Ke : np.ndarray
        The 8x8 element stiffness matrix (assumed uniform for all elements).
    threshold : float
        The threshold for determining if an element has changed.

    Returns
    -------
    scipy.sparse.csr_matrix
        The updated global stiffness matrix.
    """
    width = new_voxels.shape[1]

    # Determine which voxels have been added or removed
    added = np.logical_and(new_voxels == 1, old_voxels == 0)
    removed = np.logical_and(new_voxels == 0, old_voxels == 1)

    print("Added:", np.sum(added), "Removed:", np.sum(removed))

    # Collect data for delta matrix (changes to apply)
    delta_data = []
    delta_rows = []
    delta_cols = []

    # Process added voxels: add their Ke contributions
    added_indices = np.argwhere(added)
    for i, j in added_indices:
        nodes = coord_to_nodes(i, j, width)
        for local_i in range(8):
            global_i = nodes[local_i // 2] * 2 + (local_i % 2)
            for local_j in range(8):
                global_j = nodes[local_j // 2] * 2 + (local_j % 2)
                delta_data.append(Ke[local_i, local_j])
                delta_rows.append(global_i)
                delta_cols.append(global_j)

    # Process removed voxels: subtract their Ke contributions
    removed_indices = np.argwhere(removed)
    for i, j in removed_indices:
        nodes = coord_to_nodes(i, j, width)
        for local_i in range(8):
            global_i = nodes[local_i // 2] * 2 + (local_i % 2)
            for local_j in range(8):
                global_j = nodes[local_j // 2] * 2 + (local_j % 2)
                delta_data.append(-Ke[local_i, local_j])
                delta_rows.append(global_i)
                delta_cols.append(global_j)

    # Create delta matrix in COO format
    n_dofs = K_old.shape[0]
    delta_matrix = scipy.sparse.csr_matrix(
        (delta_data, (delta_rows, delta_cols)),
        shape=(n_dofs, n_dofs)
    )  # Convert to CSR for efficient arithmetic

    print("non-zero elements in delta matrix: ", delta_matrix.nnz)
    # Update the global stiffness matrix
    K_new = K_old + delta_matrix
    
    # Round small values to zero to improve numerical stability
    K_new.data[np.abs(K_new.data) < threshold] = 0
    K_new.eliminate_zeros()  # Remove explicit zeros from the matrix

    return K_new, delta_matrix



def add_force_to_node(node_idx, F: np.ndarray, force: np.ndarray) -> np.ndarray:
    """
    Add a force to a node.

    Parameters
    ----------
    node_idx : int
        Node index to add the force to.
    F : np.ndarray
        Force vector to add the force to.
    force : np.ndarray
        Force vector to add to the node.

    Returns
    -------
    F : np.ndarray
        Updated force vector.
    """

    F[node_idx*2] = force[0]
    F[node_idx*2+1] = force[1]
    return F

def add_force_to_voxel(i, j, width, F: np.ndarray, force: np.ndarray) -> np.ndarray:
    """
    Distribute a force over a voxel.
    """
    nodes = coord_to_nodes(i, j, width)
    for node in nodes:
        F[node*2] = force[0]/4
        F[node*2+1] = force[1]/4

    return F


def add_force_to_nodes(node_indices, F: np.ndarray, force: np.ndarray) -> np.ndarray:
    """
    Distribute a force over multiple nodes.
    
    Parameters
    ----------
    node_indices : int
        Node indices to add the force to.
    F : np.ndarray
        Force vector to add the force to.
    force : np.ndarray
        Force vector to add to the node.

    Returns
    -------
    F : np.ndarray
        Updated force vector.
    """

    for node_idx in node_indices:
        F[node_idx*2] = force[0]/len(node_indices)
        F[node_idx*2+1] = force[1]/len(node_indices)
    
    return F

def fix_boundary_nodes(node_indices, K: scipy.sparse.csr_matrix, F: np.ndarray) -> tuple[scipy.sparse.csr_matrix, np.ndarray]:
    # Apply boundary conditions to K and F - Fairly fast implementation (help from ChatGPT)

    # Degrees of freedom indices
    dof_indices = np.array([node_idx * 2 for node_idx in node_indices] + [node_idx * 2 + 1 for node_idx in node_indices])
    
    # Compute the mean of diagonal elements
    Kdiag = np.mean(K.diagonal())
    diag = K.diagonal()
    
    # Update the diagonal for the specified DOF indices
    diag[dof_indices] = Kdiag
    
    # Create a mask for rows/columns to zero out
    mask = np.zeros(K.shape[0], dtype=bool)
    mask[dof_indices] = True
    
    # Zero out the corresponding rows and columns
    newK = K.tolil()
    newK[mask, :] = 0
    newK[:, mask] = 0
    
    # Set the updated diagonal
    newK.setdiag(diag)
    
    # Zero out the corresponding entries in F
    F[dof_indices] = 0
    
    return newK.tocsr(), F


def fix_boundary_nodes_fast(node_indices, K: scipy.sparse.csr_matrix, F: np.ndarray) -> tuple[scipy.sparse.csr_matrix, np.ndarray]:
    # Vectorize DOF indices calculation.
    node_indices = np.asarray(node_indices)
    dof_indices = np.concatenate([node_indices * 2, node_indices * 2 + 1])
    
    # Cache the diagonal and compute its mean.
    diag = K.diagonal().copy()
    Kdiag = diag.mean()
    diag[dof_indices] = Kdiag
    
    # Convert to COO so we can filter out entries in boundary rows/columns.
    K_coo = K.tocoo()
    
    # Create a mask to keep only those entries whose row AND column are *not*
    # boundary DOFs.
    mask = ~np.logical_or(np.isin(K_coo.row, dof_indices),
                          np.isin(K_coo.col, dof_indices))
    
    # Build a new COO matrix with only the filtered entries.
    new_coo = scipy.sparse.coo_matrix(
        (K_coo.data[mask], (K_coo.row[mask], K_coo.col[mask])), shape=K.shape
    )
    newK = new_coo.tocsr()
    
    # At this point, many rows corresponding to boundary DOFs may have no stored
    # diagonal entry. To be sure that the new diagonal is exactly what we want,
    # we remove any existing diagonal and add a fresh diagonal.
    # Subtract any existing diagonal:
    newK = newK - scipy.sparse.diags(newK.diagonal(), shape=K.shape, format='csr')
    # Add the new diagonal.
    newK = newK + scipy.sparse.diags(diag, shape=K.shape, format='csr')
    
    # Zero out the corresponding entries in F.
    F[dof_indices] = 0
    
    return newK, F

def fix_null_nodes(K: scipy.sparse.csr_matrix, F: np.ndarray) -> tuple[scipy.sparse.csr_matrix, np.ndarray, list]:
    # Fix null/zero nodes of K and F
    null_nodes = K.diagonal() == 0
    keep_nodes = ~null_nodes

    K = K[keep_nodes, :][:, keep_nodes]
    F = F[keep_nodes]

    return K, F, np.where(null_nodes)[0]

def n_components(matrix):
    # Number of components in the graph rep. of matrix
    n_components, _ = scipy.sparse.csgraph.connected_components(matrix)  # Find connected components

    return n_components  # If only one component, graph is connected

def solve(K: scipy.sparse.csr_matrix, F: np.ndarray, fixed_nodes: list, debug: bool = False) -> np.ndarray:
    """
    Solve the linear system K @ u = F, subject to displacement boundary conditions.

    Parameters
    ----------
    K : np.ndarray
        Stiffness matrix.
    F : np.ndarray
        Force vector.
    fixed_nodes : list
        List of node indices with zero displacement.

    Returns
    -------
    u : np.ndarray
        Solution vector with displacements at each node.
    components : int
        Number of connected components in the graph.
    """
    # Apply boundary conditions and solve the linear system and 
    ## TODO improve performance of fixing boundary nodes - perhaps by reusing K_red from previous iteration
    pret1 = time.perf_counter()
    K_red, F_red = fix_boundary_nodes(fixed_nodes, K, F) # Remove fixed nodes (zero displacements)
    K_red, F_red, null_nodes = fix_null_nodes(K_red, F_red) # Remove null nodes (unconnected nodes)
    compt1 = time.perf_counter()
    components = n_components(K_red)
    compt2 = time.perf_counter()
    print(components, " Components")
    print(f"Calculating components took {compt2-compt1} seconds")

    pret2 = time.perf_counter()
    if debug:
        print(f"Preprocessing took {pret2-pret1} seconds")
    
    t1 = time.perf_counter()
    u_red = scipy.sparse.linalg.spsolve(K_red.tocsr(), F_red)
    t2 = time.perf_counter()

    if debug: 
        print(f"Solved in {t2-t1} seconds using spsolve")

    u = u_red

    for i in range(len(null_nodes)): # Add back null nodes
        u = np.insert(u, null_nodes[i], 0, axis=0)
    
    return u, components

def get_B_matrix(F: np.ndarray) -> np.ndarray:
    B = np.zeros((F.shape[0], 3))
    B[0::2, 0] = 1
    B[1::2, 1] = 1
    B[0::2, 2] = -1
    B[1::2, 2] = 1

    return B

import warnings

class Solver:
    factor = None

    def __init__(self, K: scipy.sparse.csr_matrix, F: np.ndarray):
        """
        Initialize the Solver class.

        Parameters
        ----------
        K : scipy.sparse.csr_matrix
            Stiffness matrix.
        F : np.ndarray
            Force vector (not actually used)
        """
        K_red, F_red, null_nodes = fix_null_nodes(K, F)
        self.K_red = K_red.tocsc()
        self.factor = cholesky(K_red.tocsc())
        self.threshold = 1e-10 * np.max(np.abs(K.data))
    
    def refactor(self, K: scipy.sparse.csr_matrix, remove_null_nodes=False):
        """
        Update the factorization of the stiffness matrix.
        """
        if remove_null_nodes:
            K, _, null_nodes = fix_null_nodes(K, np.zeros(K.shape[0]))

        self.K_red = K.tocsc()
        try:
            self.factor = cholesky(K.tocsc(), ordering_method = "colamd")
        except Exception as e:
            self.factor = None

    def solve(self, K: scipy.sparse.csr_matrix, F: np.ndarray, fixed_nodes, debug: bool = False) -> np.ndarray:
        """
        Solve the linear system K @ u = F, subject to displacement boundary conditions.

        Parameters
        ----------
        K : scipy.sparse.csr_matrix
            Stiffness matrix.
        F : np.ndarray    
            Force vector.
        fixed_nodes : list
            List of node indices with zero displacement.

        Returns
        -------
        u : np.ndarray
            Solution vector with displacements at each node.
        components : int
            Number of connected components in the graph.
        """
        
        t1 = time.perf_counter()
        K, F = fix_boundary_nodes_fast(list(fixed_nodes), K, F)
        t2 = time.perf_counter()

        if debug:
            print(f"Fixing boundary nodes took {t2-t1} seconds")

        pret1 = time.perf_counter()
        K.data[np.abs(K.data) < self.threshold] = 0
        K.eliminate_zeros()
        K_red, F_red, null_nodes = fix_null_nodes(K, F) # Remove null nodes (unconnected nodes)

        compt1 = time.perf_counter()
        components = n_components(K_red)
        compt2 = time.perf_counter()

        if debug:
            print(components, " Components")
            print(f"Calculating components took {compt2-compt1} seconds")
        
        pret2 = time.perf_counter()
        if debug:
            print(f"Preprocessing took {pret2-pret1} seconds")
        
        t1 = time.perf_counter()
        self.refactor(K_red)

        # If matrix is PSD we can use cholesky, otherwise use spsolve
        if self.factor is None:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", scipy.sparse.linalg.MatrixRankWarning)
                #u_red = pypardiso.spsolve(K_red, F_red)
                u_red = scipy.sparse.linalg.spsolve(K_red, F_red)
                t2 = time.perf_counter()
                if debug: 
                    print(f"Solved in {t2-t1} seconds using spsolve")
                
                # Catch the case where the matrix is singular
                if any(issubclass(warn.category, scipy.sparse.linalg.MatrixRankWarning) for warn in w):
                    print("Matrix is singular - abort")
                    return None, None
        else:
            u_red = np.squeeze(self.factor(F_red))
            t2 = time.perf_counter()
            if debug: 
                print(f"Solved in {t2-t1} seconds using cholesky")

        # Now we want to add back all the null nodes

        # Create a boolean mask for all indices
        mask = np.ones(len(u_red) + len(null_nodes), dtype=bool)

        # Set False for null node indices
        mask[null_nodes] = False

        # Use the mask to insert zeros at null node positions
        u_with_nulls = np.zeros((len(u_red) + len(null_nodes),), dtype=u_red.dtype)
        u_with_nulls[mask] = u_red

        # Replace the original 'u' with the new array containing null nodes
        u = u_with_nulls
        
        return u, components


def sub_divide(voxels: np.ndarray, factor: int) -> np.ndarray:
    new_voxels = np.zeros((factor*voxels.shape[0], factor*voxels.shape[1]))

    # Sub divide voxels
    for i in range(voxels.shape[0]):
        for j in range(voxels.shape[1]):
            if (voxels[i,j] == 1):
                new_voxels[i*factor:(i+1)*factor, j*factor:(j+1)*factor] = 1
    
    # Round corners with use of 2D convolution
    bot_right = [[0, 1, 0], [1, -1, -1], [0, -1, 0]]
    bot_left = [[0, 1, 0], [-1, -1, 1], [0, -1, 0]]
    top_right = [[0, -1, 0], [1, -1, -1], [0, 1, 0]]
    top_left = [[0, -1, 0], [-1, -1, 1], [0, 1, 0]]

    #br
    br = scipy.signal.convolve2d(new_voxels, bot_right, mode='same', boundary='fill', fillvalue=0)==2
    #bl
    bl = scipy.signal.convolve2d(new_voxels, bot_left, mode='same', boundary='fill', fillvalue=0)==2
    #tr
    tr = scipy.signal.convolve2d(new_voxels, top_right, mode='same', boundary='fill', fillvalue=0)==2
    #tl
    tl = scipy.signal.convolve2d(new_voxels, top_left, mode='same', boundary='fill', fillvalue=0)==2

    new_voxels = (new_voxels + br + bl + tr + tl) > 0
    return new_voxels

def test():
    import matplotlib.pyplot as plt
    import femplotter
    import time
    # Simple test case for the voxel fem solver

    E = 200e9  # Young's modulus (Pa)
    nu = 0.3   # Poisson's ratio
    L = 0.01    # Side length (m)
    t = 0.1   # Thickness (m)
    
    t1 = time.perf_counter()

    Ke = element_stiffness_matrix(E, nu, L, t)

    # Define the voxels/mesh
    voxels = np.array([[0,1],
                    [1,1]])
    
    # Subdivide
    voxels = sub_divide(voxels, 2)

    # Compute the global stiffness matrix
    K = global_stiffness_matrix(Ke, voxels)
    n_dofs = K.shape[0]

    # Define the force
    F = np.zeros((n_dofs, 1))
    F = add_force_to_node(4, F, np.array([0.5, 0.5]))

    t2 = time.perf_counter()

    # Solve displacements (and add boundary conditions)
    solver = Solver(K, F)
    u, _ = solver.solve(K, F, [5, 20, 21, 22, 23, 24])
    t3 = time.perf_counter()

    print("Time to setup system: ", (t2-t1))
    print("Time to solve system: ", (t3-t2))

    # Plot the displacements
    vector_figure = femplotter.node_vector_plot(u, voxels)
    vector_figure.suptitle("Displacements")
    femplotter.plot_displaced_mesh(u, voxels, new_figure=True)

    # Compute stresses and strains
    eps = get_element_strains(u, voxels, L)
    sigma = get_element_stresses(eps, E, nu)
    n_eps = get_node_values(eps, voxels, L)
    n_sigma = get_node_values(sigma, voxels, L)

    # Plot the von_mises stresses
    von_mises = von_mises_stresses_node(n_sigma)
    von_mises_figure = femplotter.node_value_plot(von_mises, voxels)
    von_mises_figure.suptitle("von Mises stresses")
    plt.show()

if __name__ == "__main__":
    test()

