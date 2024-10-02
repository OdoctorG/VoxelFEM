import math
import numpy as np
import matplotlib.pyplot as plt


""" 
Voxel 2D FEM Solver 

Uses square elements with four nodes and bilinear form functions.
The element is defined as follows:
N1 --- N2
|       |
|       |
N3 --- N4

N₁ = (1 - ξ)(1 - η) / 4
N₂ = (1 + ξ)(1 - η) / 4
N₃ = (1 - ξ)(1 + η) / 4
N₄ = (1 + ξ)(1 + η) / 4

"""
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

def element_stiffness_matrix(E: float, nu: float, L, t):
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
    gauss_points = [-1/np.sqrt(3), 1/np.sqrt(3)]
    w = 1
    
    K = np.zeros((8, 8))
    
    for xi in gauss_points:
        for eta in gauss_points:
            # Compute B matrix at Gauss point
            B = B_matrix(xi, eta, L)
            
            # Compute Jacobian determinant
            J = L**2 / 4
            
            # Accumulate contribution to stiffness matrix
            K += w * w * np.dot(np.dot(B.T, D), B) * J * t

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
    gauss_points = [-1/np.sqrt(3), 1/np.sqrt(3)]
    n_gauss_points = len(gauss_points)
    w = 1
    strains = np.zeros((voxels.shape[0]*n_gauss_points,voxels.shape[1]*n_gauss_points,3), dtype=object)
    for i in range(voxels.shape[0]):
        for j in range(voxels.shape[1]):

            if voxels[i, j] == 1:
                # Convert voxel coordinates to node indices
                nodes = coord_to_nodes(i, j, W)
                # Get nodal displacements
                el_u = np.zeros(8)
                for k in range(len(nodes)):
                    el_u[k] = u[nodes[k]][0]
                    el_u[k+1] = u[nodes[k]+1][0]

                # Loop over all Gauss points
                xi_idx = 0
                for xi in gauss_points:
                    eta_idx = 0
                    for eta in gauss_points:
                        # Compute B matrix at Gauss point
                        B = B_matrix(xi, eta, L)
                
                        # Compute Jacobian determinant
                        J = 0.25
                        # Compute strains at Gauss point
                        strains[n_gauss_points*i+eta_idx,n_gauss_points*j+xi_idx][0] += w * w * np.dot(B[0], el_u) * J
                        strains[n_gauss_points*i+eta_idx,n_gauss_points*j+xi_idx][1] += w * w * np.dot(B[1], el_u) * J
                        strains[n_gauss_points*i+eta_idx,n_gauss_points*j+xi_idx][2] += w * w * np.dot(B[2], el_u) * J

                        eta_idx += 1
                    xi_idx += 1
        
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
    gauss_points = [-1/np.sqrt(3), 1/np.sqrt(3)]
    n_gauss_points = len(gauss_points)
    w = 1
    strains = np.zeros(((voxels.shape[0]+1)*(voxels.shape[1]+1),3))
    avg_count = np.zeros(((voxels.shape[0]+1)*(voxels.shape[1]+1),1), dtype=int)
    W = voxels.shape[1] # Number of columns

    for i in range(voxels.shape[0]):
        for j in range(voxels.shape[1]):
            if voxels[i, j] == 1:
                nodes = coord_to_nodes(i,j, W)
                # Loop over all Gauss points
                xi_idx = 0
                for xi in gauss_points:
                    eta_idx = 0
                    for eta in gauss_points:
                        elem_strain = element_s[n_gauss_points*i+eta_idx,n_gauss_points*j+xi_idx]

                        N = shape_function(xi, eta)

                        for k in range(len(nodes)):
                            for l in range(3):
                                strains[nodes[k]][l] += N[k] * elem_strain[l]
                                avg_count[nodes[k]][0] += 1
                        eta_idx += 1
                    xi_idx += 1
    
    return strains





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
    stresses = np.zeros((element_strains.shape[0], element_strains.shape[1]), dtype=object)
    for i in range(element_strains.shape[0]):
        for j in range(element_strains.shape[1]):
            stresses[i,j] = np.dot(D, element_strains[i,j])
    
    return stresses

def von_mises_stresses(stresses: np.ndarray) -> np.ndarray:
    """
    Compute von Mises stresses from stress tensor at each  element.

    Parameters
    ----------
    stresses : np.ndarray
        Stresses at each element.

    Returns
    -------
    np.ndarray
        von Mises stresses at each element.
    """
    von_mises = np.zeros((stresses.shape[0], stresses.shape[1]))

    # Compute von Mises stresses
    for i in range(stresses.shape[0]):
        for j in range(stresses.shape[1]):
            von_mises[i,j] = np.sqrt(np.square(stresses[i,j][0]) + np.square(stresses[i,j][1]) - stresses[i,j][0] * stresses[i,j][1] + 3*np.square(stresses[i,j][2]))

    return von_mises

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
    von_mises = np.zeros(stresses.shape[0])

    # Compute von Mises stresses
    for i in range(stresses.shape[0]):
        von_mises[i] = np.sqrt(np.square(stresses[i][0]) + np.square(stresses[i][1]) - stresses[i][0] * stresses[i][1] + 3*np.square(stresses[i][2]))

    return von_mises


def coord_to_nodes(i,j, W) -> tuple[int, int, int, int]:
    """Convert voxel coordinates to node indices"""
    return (i*(W+1)+j, i*(W+1)+j+1, (i+1)*(W+1)+j, (i+1)*(W+1)+j+1)

def nodes_to_coord(node_idx, W) -> tuple[int, int, int, int]:
    """Convert node indices to voxel coordinates"""
    i = node_idx // (W+1)
    j = node_idx % (W+1)
    return (i, j)

def global_stiffness_matrix(Ke: np.ndarray, voxels: np.ndarray) -> np.ndarray:
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
    This function assumes that the element stiffness matrix is symmetric, i.e., Ke = Ke.T.
    The function also assumes that the voxels are numbered in column-major order, i.e., the first column is 0, 1, 2, ... and the second column is Width, Width+1, Width+2, ...
    """
    dof_per_node = 2
    n_dofs = (voxels.shape[0]+1)*(voxels.shape[1]+1)*dof_per_node
    Width = voxels.shape[1]
    K = np.zeros((n_dofs, n_dofs))

    # Loop over all voxels
    for i in range(voxels.shape[0]):
        for j in range(voxels.shape[1]):
            if voxels[i, j] == 1:
                # Convert the voxel coordinates to node numbers
                my_nodes = coord_to_nodes(i, j, Width)

                # Define the connections for the square element
                connections = [
                    (0,1), 
                    (1,3),
                    (3,2),
                    (2,0)
                ]
                
                # Loop over all connections
                for connection in connections:
                    # Add all connections to global stiffness matrix
                    ii = connection[0]*2
                    jj = connection[1]*2
                    global_i = my_nodes[connection[0]]*2
                    global_j = my_nodes[connection[1]]*2

                    K[global_i:global_i+2, global_j:global_j+2] += Ke[ii:ii+2, jj:jj+2]
                    K[global_j:global_j+2, global_i:global_i+2] += Ke[jj:jj+2, ii:ii+2] # symmetry

                    K[global_i:global_i+2, global_i:global_i+2] += Ke[ii:ii+2, ii:ii+2] # diagonal (self)
    
    return K

def add_force_to_node(node_idx, F: np.ndarray, force: np.ndarray) -> np.ndarray:
    """
    Add a force to a node in the force vector F.

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

def fix_boundary_nodes(node_indices, K: np.ndarray, F: np.ndarray) -> np.ndarray:
    # Find boundary nodes, and remove them from the K and F arrays
    # Return the updated K and F arrays as well as the indicesn (of the dof) of the boundary nodes

    dof_indices = []
    
    for node_idx in node_indices:
        dof_indices.append(node_idx*2)
        dof_indices.append(node_idx*2+1)
    
    K = np.delete(K, dof_indices, axis=0)
    K = np.delete(K, dof_indices, axis=1)
    F = np.delete(F, dof_indices, axis=0)

    return K, F, dof_indices

def fix_null_nodes(K: np.ndarray, F: np.ndarray) -> np.ndarray:
    # Find null nodes, and remove them from the K and F arrays
    # Return the updated K and F arrays as well as the indices (of the dof) of the null nodes

    # Find null nodes
    null_nodes = []
    for i in range(K.shape[0]):
        if (K[i,i] == 0):
            null_nodes.append(i)
    
    # Remove null nodes
    K = np.delete(K, null_nodes, axis=0)
    K = np.delete(K, null_nodes, axis=1)
    F = np.delete(F, null_nodes, axis=0)

    return K, F, null_nodes


def solve(K: np.ndarray, F: np.ndarray, fixed_nodes: list) -> np.ndarray:
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
    """

    K_red, F_red, _ = fix_boundary_nodes(fixed_nodes, K, F) # Remove fixed nodes (zero displacements)
    K_red, F_red, null_nodes = fix_null_nodes(K_red, F_red) # Remove null nodes (unconnected nodes)
    #u_red = np.linalg.pinv(K_red) @ F_red
    u_red = np.linalg.solve(K_red, F_red)
    print(len(u_red))
    u = u_red
    for i in range(len(null_nodes)): # Add back null nodes
        u = np.insert(u, null_nodes[i], 0, axis=0)
    print(len(u))
    for i in range(len(fixed_nodes)): # Add back fixed nodes
        u = np.insert(u, fixed_nodes[i], 0, axis=0)
        u = np.insert(u, fixed_nodes[i], 0, axis=0)
    print(len(u))
    return u

def sub_divide(voxels: np.ndarray, factor: int) -> np.ndarray:
    new_voxels = np.zeros((factor*voxels.shape[0], factor*voxels.shape[1]))

    for i in range(voxels.shape[0]):
        for j in range(voxels.shape[1]):
            if (voxels[i,j] == 1):
                new_voxels[i*factor:(i+1)*factor, j*factor:(j+1)*factor] = 1
    
    return new_voxels

def node_vector_plot(vals: np.ndarray, voxels: np.ndarray, scale=10e9) -> None:
    plt.figure()
    plt.imshow(voxels, cmap="binary")
    print(vals.shape)
    for i in range(len(vals)//2):

        coord = nodes_to_coord(i, voxels.shape[1])
        plt.arrow(coord[1]-0.5, coord[0]-0.5, scale*vals[i*2][0], scale*vals[i*2+1][0], color="red")

def node_value_plot(vals: np.ndarray, voxels: np.ndarray, scale=1) -> None:
    plt.figure()
    norm = plt.Normalize(min(vals), max(vals))
    # Vectorize the function
    vectorized_compute_vector = np.vectorize(shape_function, signature='(),()->(n)')
    print(vals.shape)
    for i in range(voxels.shape[0]):
        for j in range(voxels.shape[1]):
            if (voxels[i, j] == 1):
                coord = coord_to_nodes(i, j, voxels.shape[1])
                z = [vals[coord[0]], vals[coord[1]], vals[coord[2]], vals[coord[3]]]
                x = np.linspace(-1, 1, 100)
                y = np.linspace(-1, 1, 100)
                X, Y = np.meshgrid(x, y)


                # Compute vectors for all points in the meshgrid
                vectors = vectorized_compute_vector(X, Y)

                # Compute the interpolated value
                result = np.dot(vectors, z)

                # Plot the result
                plt.pcolormesh(X*0.5+j, Y*0.5+i, result, cmap='viridis', alpha=0.5, norm=norm)
    plt.colorbar()

def test():
    # Simple test case for the voxel fem solver

    E = 200e9  # Young's modulus (Pa)
    nu = 0.3   # Poisson's ratio
    L = 0.01    # Side length (m)
    t = 0.1   # Thickness (m)

    Ke = element_stiffness_matrix(E, nu, L, t)

    voxels = np.array([[0,1],
                    [1,1]])
    
    voxels = sub_divide(voxels, 2)

    K = global_stiffness_matrix(Ke, voxels)

    n_dofs = K.shape[0]
    F = np.zeros((n_dofs, 1))
    F = add_force_to_node(3, F, np.array([1, 1]))

    print(K.shape)

    u = solve(K, F, [20, 21, 22, 23, 24])
    node_vector_plot(u, voxels)
    plt.figure()
    eps = get_element_strains(u, voxels, L)
    sigma = get_element_stresses(eps, E, nu)
    get_node_values(eps, voxels, L)
    n_sigma = get_node_values(sigma, voxels, L)
    moment = n_sigma[:, 2]
    node_value_plot(moment, voxels)

    von_mises = von_mises_stresses_node(n_sigma)
    node_value_plot(von_mises, voxels)
    #print(u)
    #print(eps)
    #print(sigma)
    plt.show()

if __name__ == "__main__":
    test()
