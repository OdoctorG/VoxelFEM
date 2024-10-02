import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from femsolver import nodes_to_coord, coord_to_nodes, shape_function

def plot_mesh(voxels: np.ndarray, new_figure: bool = False, offset: np.ndarray = np.array([0, 0]),
            flip_y: bool = False, z_order=2) -> plt.Figure:
    if new_figure:
        fig = plt.figure()
    for i in range(voxels.shape[0]):
        for j in range(voxels.shape[1]):
            if voxels[i, j] == 1:
                for k in [[0,1,0,0],[0,0,0,1],[1,1,0,1],[0,1,1,1]]:
                    if flip_y:
                        plt.plot([j+k[0]+offset[0], j+k[1]+offset[0]], [-i+k[2]+offset[1], -i+k[3]+offset[1]], color="black", zorder=z_order)
                    else:
                        plt.plot([j+k[0]+offset[0], j+k[1]+offset[0]], [i+k[2]+offset[1], i+k[3]+offset[1]], color="black", zorder=z_order)
    if new_figure:
        return fig
    else:
        return None

def plot_displaced_mesh(u: np.ndarray, voxels: np.ndarray, scale: float = 10e9, new_figure: bool = False, offset: np.ndarray = np.array([0, 0]),
            flip_y: bool = False, z_order=2) -> plt.Figure:
    if new_figure:
        fig = plt.figure()
    for i in range(voxels.shape[0]):
        for j in range(voxels.shape[1]):
            if voxels[i, j] == 1:
                nodes = coord_to_nodes(i, j, voxels.shape[1])
                node_displacements = []
                for node in nodes:
                    node_displacements.append((u[node*2]*scale, u[node*2+1]*scale))

                line1 = [[j+offset[0]+node_displacements[0][0], j+offset[0]+1+node_displacements[1][0]], 
                        [i+offset[1]+node_displacements[0][1], i+offset[1]+node_displacements[1][1]]]
                line2 = [[j+offset[0]+1+node_displacements[1][0], j+offset[0]+1+node_displacements[3][0]], 
                        [i+offset[1]+node_displacements[1][1], i+1+offset[1]+node_displacements[3][1]]]
                line3 = [[j+offset[0]+1+node_displacements[3][0], j+offset[0]+node_displacements[2][0]], 
                        [i+offset[1]+1+node_displacements[3][1], i+offset[1]+1+node_displacements[2][1]]]
                line4 = [[j+offset[0]+node_displacements[2][0], j+offset[0]+node_displacements[0][0]], 
                        [i+offset[1]+1+node_displacements[2][1], i+offset[1]+node_displacements[0][1]]]

                for line in [line1, line2, line3, line4]:
                    plt.plot(line[0], line[1], color="black", zorder=z_order)
    ax = fig.axes
    ax[0].invert_yaxis()
    if new_figure:
        return fig
    else:
        return None


def node_vector_plot(vals: np.ndarray, voxels: np.ndarray, scale=10e9) -> plt.Figure:
    """
    Plot a vector at each node in the mesh, given the values and the voxel representation of the geometry.

    Parameters
    ----------
    vals : np.ndarray
        2D vector field
    voxels : np.ndarray
        2D array of voxels, where 1 indicates a solid voxel and 0 indicates a void voxel.
    scale : float
        Scale factor for the vector length.

    Returns
    -------
    None
    """
    
    fig = plt.figure()
    
    print(vals.shape)
    norm = Normalize(vals.min(), vals.max())
    for i in range(len(vals)//2):
        coord = nodes_to_coord(i, voxels.shape[1])

        vector_length = np.sqrt(vals[i*2][0]**2 + vals[i*2+1][0]**2)
        color = cm.viridis(norm(vector_length))

        if vector_length > 0:
            plt.arrow(coord[1], coord[0], scale*vals[i*2][0], scale*vals[i*2+1][0], color=color, head_width=np.log(vector_length*scale*0.5+1)+0.01,
                    zorder=3)
    fig.axes[0].invert_yaxis()
    plot_mesh(voxels)
    return fig
def node_value_plot(vals: np.ndarray, voxels: np.ndarray, scale=1) -> plt.Figure:
    """
    Plot a 2D interpolation of the given values at each node in the mesh, given the voxel representation of the geometry.

    Parameters
    ----------
    vals : np.ndarray
        1D array of values at each node
    voxels : np.ndarray
        2D array of voxels, where 1 indicates a solid voxel and 0 indicates a void voxel.
    scale : float
        Scale factor for the plot.

    Returns
    -------
    None
    """

    fig = plt.figure()
    norm = plt.Normalize(0.9*min(vals), max(vals)*4)
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
                plt.pcolormesh(X*0.5+0.5+j, Y*(0.5)+0.5+i, result, cmap='viridis', norm=norm)
    plt.colorbar()
    fig.axes[0].invert_yaxis()
    plot_mesh(voxels)
    return fig