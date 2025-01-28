""" Geometry Optimization -- Main Module """

import math
import numpy as np
import matplotlib.pyplot as plt
from geometry import *

import femsolver, femplotter, time


def calculate_stress(voxels, Ke, L, E, nu, grid: Grid, objects: ObjectManager, cond_limit=None):
    # Calculate stresses based on voxel geometry

    K = femsolver.global_stiffness_matrix(Ke, voxels)
    
    n_dofs = K.shape[0]

    F = np.zeros((n_dofs, 1))
    u = np.zeros((n_dofs, 1))
    #F = femsolver.add_force_to_node(4, F, np.array([0, 0.5]))

    fixed_nodes = set()
    BoundaryVoxels = grid.get_boundary_voxels(objects)
    for b in BoundaryVoxels:
        nodes = femsolver.coord_to_nodes(b[1], b[0], voxels.shape[1])
        if b[0] >= voxels.shape[1] or b[1] >= voxels.shape[0]:
            continue
        elif b[0] < 0 or b[1] < 0:
            continue
        for node in nodes:
            fixed_nodes.add(node)
    
    ForceLines = []
    for line in objects.lines:
        if isinstance(line, ForceLine):
            ForceLines.append(line)
    for force_line in ForceLines:
        force = force_line.force_dir*force_line.force
        force_voxels = grid.get_voxels_intersecting_line(force_line)
        for voxel in force_voxels:
            F = femsolver.add_force_to_voxel(voxel[1], voxel[0], voxels.shape[1], F, force)
    
    u, components = femsolver.solve(K.tocsr(), F, fixed_nodes=list(fixed_nodes), debug=True, )
    if u is None:
        return None, None

    # Compute stresses and strains
    eps = femsolver.get_element_strains(u, voxels, L)
    sigma = femsolver.get_element_stresses(eps, E, nu)
    n_sigma = femsolver.get_node_values(sigma, voxels, L)
    
    von_mises = femsolver.von_mises_stresses_node(n_sigma)
    von_mises_v = femsolver.get_voxel_values(von_mises, voxels)
    State = {"u": u, "n_sigma": n_sigma, "von_mises": von_mises, "components": components}
    return von_mises_v, State



def forward_pass_B(objects: ObjectManager, grid: Grid, break_limit = 10_000_000, stepsize = 0.2, disconnect_counter = 1):
    # Geometry optimization based on fixed percentage steps of geometry removal
    
    E = 200e9  # Young's modulus (Pa)
    nu = 0.3   # Poisson's ratio
    L = 0.01    # Side length (m)
    t = 0.1   # Thickness (m)

    highest_stress = 0

    counter = 0
    Ke = femsolver.element_stiffness_matrix(E, nu, L, t)
    voxels = grid.getVoxels()
    new_voxels = np.copy(voxels)
    old_voxels = new_voxels
    old_highest_stress = highest_stress

    # Lock nodes with boundary conditions
    locked_voxels = grid.get_boundary_voxels(objects)
    locked_voxels.extend(grid.get_force_voxels(objects))

    def calc_and_plot(voxels):
        nonlocal old_voxels
        von_mises, old_state = calculate_stress(voxels, Ke, L, E, nu, grid, objects)
        if von_mises is None:
            print("Failed to calculate stress")
            return None
        #print("plot stresses")
        von_mises_figure = femplotter.node_value_plot(old_state["von_mises"], old_voxels)
        
        #femplotter.plot_mesh(old_voxels, new_figure=True, opacity=0.5, color="lightblue")
        #femplotter.plot_displaced_mesh(old_state["u"], old_voxels, scale=10e9)
        plt.show()
        return von_mises, old_state

    von_mises, old_state = calc_and_plot(old_voxels)
    n_disered_components = old_state["components"]

    while True:
        nnz = np.count_nonzero(new_voxels)
        condlimit = None if counter <= disconnect_counter else 1e12
        von_mises_v, state = calculate_stress(new_voxels, Ke, L, E, nu, grid, objects, condlimit)
        if von_mises_v is None:
            break
        locked_indices = [v[1]*voxels.shape[1] + v[0] for v in locked_voxels]
        sorted_indices = np.argsort(von_mises_v, axis=None)
        highest_stress = von_mises_v.flatten()[sorted_indices[-1]]

        if highest_stress > break_limit:
            print("Stress too high: ", highest_stress)
            break
        elif state["components"] != n_disered_components and counter > disconnect_counter:
            print("Mesh disconnected")
            break
        elif counter > 50:
            print("Too many iterations")
            break
        
        old_voxels = new_voxels.copy()
        old_highest_stress = highest_stress

        flat_voxels = new_voxels.flatten()
        i = 0
        j = 0
        #stepsize *= 0.85
        while i < math.ceil(stepsize*nnz) and j < len(sorted_indices):
            if flat_voxels[sorted_indices[j]] == 1 and sorted_indices[j] not in locked_indices:
                i += 1
                flat_voxels[sorted_indices[j]] = 0
            j += 1

        new_voxels = np.reshape(flat_voxels, (voxels.shape[0], voxels.shape[1]))
        #print(f"iteration {counter}, highest stress: {highest_stress}")

        counter += 1

    von_mises_v, old_state = calc_and_plot(old_voxels)
    sorted_indices = np.argsort(von_mises_v, axis=None)
    highest_stress = von_mises_v.flatten()[sorted_indices[-1]]
    print(f"Break limit: {break_limit}, highest stress: {highest_stress}")
    return old_voxels, old_state

def forward_pass_A(objects: ObjectManager, grid: Grid, break_limit = 10_000_000, steps = 3, low_limit = 0.0, high_limit = 1.0):
    # Geometry optimization based on interval bisection

    E = 200e9  # Young's modulus (Pa)
    nu = 0.3   # Poisson's ratio
    L = 0.01    # Side length (m)
    t = 0.1   # Thickness (m)

    highest_stress = 0

    counter = 0
    Ke = femsolver.element_stiffness_matrix(E, nu, L, t)
    voxels = grid.getVoxels()
    new_voxels = np.copy(voxels)
    old_voxels = new_voxels
    old_highest_stress = highest_stress

    # Lock nodes with boundary conditions
    locked_voxels = grid.get_boundary_voxels(objects)
    locked_voxels.extend(grid.get_force_voxels(objects))

    def calc_and_plot(voxels):
        #nonlocal old_voxels
        von_mises, old_state = calculate_stress(voxels, Ke, L, E, nu, grid, objects)
        if von_mises is None:
            print("Failed to calculate stress")
            return None
        #print("plot stresses")
        von_mises_figure = femplotter.node_value_plot(old_state["von_mises"], voxels)
        
        #femplotter.plot_mesh(old_voxels, new_figure=True, opacity=0.5, color="lightblue")
        #femplotter.plot_displaced_mesh(old_state["u"], old_voxels, scale=10e9)
        plt.show()
        return von_mises, old_state

    def select_voxels(voxels, von_mises_v, locked_indices, percentage: float = 0.5):
        # Select a subset percentage of the voxels based on the stress
        nnz = np.count_nonzero(voxels)
        sorted_indices = np.argsort(von_mises_v, axis=None)

        flat_voxels = voxels.flatten()
        i = 0
        j = 0

        while i < math.ceil((1-percentage)*nnz) and j < len(sorted_indices):
            if flat_voxels[sorted_indices[j]] == 1 and sorted_indices[j] not in locked_indices:
                i += 1
                flat_voxels[sorted_indices[j]] = 0
            j += 1

        new_voxels = np.reshape(flat_voxels, (voxels.shape[0], voxels.shape[1]))
        return new_voxels

    von_mises_v, state = calc_and_plot(old_voxels)
    OG_VON_MISES = von_mises_v

    desired_n_components = state["components"]
    components = desired_n_components
    limit = high_limit
    upper_limit = high_limit
    lower_limit = low_limit

    while counter < steps:
        sorted_indices = np.argsort(von_mises_v, axis=None)
        locked_indices = [v[1]*voxels.shape[1] + v[0] for v in locked_voxels]
        highest_stress = von_mises_v.flatten()[sorted_indices[-1]]
        disconnected = components != desired_n_components

        if (highest_stress > break_limit or disconnected) and limit == high_limit:
            print(f"Stress too high: {highest_stress}")
            break
        elif highest_stress >= break_limit or disconnected: # Interval splitting search
            lower_limit = limit
            limit += (upper_limit-limit)*0.5
        elif highest_stress < break_limit:
            OG_VON_MISES = von_mises_v
            old_voxels = new_voxels
            upper_limit = limit
            limit -= (limit-lower_limit)*0.5
            lower_limit *= 0.9

        print("LIMIT: ", limit)
        print(highest_stress, " / ",break_limit)
        new_voxels = select_voxels(voxels, OG_VON_MISES, locked_indices, limit)
        #calc_and_plot(new_voxels)

        von_mises_v, state = calculate_stress(new_voxels, Ke, L, E, nu, grid, objects)
        if von_mises_v is None:
            break
        components = state["components"]
        counter += 1

    von_mises_v, old_state = calc_and_plot(old_voxels)
    sorted_indices = np.argsort(von_mises_v, axis=None)
    highest_stress = von_mises_v.flatten()[sorted_indices[-1]]
    print(f"Break limit: {break_limit}, highest stress: {highest_stress}")
    return old_voxels, old_state, lower_limit, upper_limit

def opt(objects: ObjectManager):
    # Optimizatation of voxel geometry
    bbox = objects.get_bounding_box()
    # Define grid
    xdivs = 100
    ydivs = 10

    # Adjustment factor for break limit 
    # - the break limit approaches the true break limit with each iteration of i
    def adjust(x):
        return (x+1)/(x+1.5)
    
    # Setup grid and voxels
    grid = Grid(bbox[0], bbox[1], bbox[2], bbox[3], xdivs, ydivs)
    grid.add_voxels_inside(objects)
    voxels = grid.getVoxels()

    # 200 Mpa stress limit for testing
    break_limit = 200_000_000

    # Run optimization - each iteration you get finer resolution of the solution
    # No gaurantee of correctness or convergence 
    for i in range(4):
        print(np.sum(voxels == 1), " voxels")
        grid = Grid(bbox[0], bbox[1], bbox[2], bbox[3], xdivs, ydivs)
        grid.voxels = voxels

        objects.draw()
        grid.draw()

        plt.gcf().axes[0].invert_yaxis()
        plt.margins(0.1)
        plt.show()

        voxels, state, low, high = forward_pass_A(objects, grid, break_limit*adjust(i), 25, 0.0, 1.0)
        #grid.voxels = voxels
        #voxels, state = forward_pass_B(objects, grid, break_limit*adjust(i), stepsize=0.01, disconnect_counter=-1)

        voxels = femsolver.sub_divide(voxels, 2)
        xdivs *= 2
        ydivs *= 2

def test():
    # Test program

    # Construct geometry
    objects = ObjectManager()
    ground = Object2D(0, 0, 10, 1, 0)
    #platform = Object2D(6, 7, 5, 1, -45)
    line2 = ForceLine((0.01, 0), (0.01, 1), 1000, [0,1])
    line3 = FixedLine((10, 0), (10, 1))
    
    objects.add(ground)
    #objects.add(platform)
    objects.add_line(line2)
    objects.add_line(line3)
    plt.margins(y=1.5)
    opt(objects)
    
    plt.show()
    return

if __name__ == "__main__":
    test()
