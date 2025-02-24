""" Geometry Optimization -- Main Module """

import math
import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.sparse
from geometry import *

import femsolver, femplotter, time

PLOT = False

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
    
    u, components = femsolver.solve(K.tocsr(), F, fixed_nodes=list(fixed_nodes), debug=True)
    if u is None:
        return None, None

    # Compute stresses and strains
    eps = femsolver.get_element_strains_fast(u, voxels, L)
    sigma = femsolver.get_element_stresses_fast(eps, E, nu)
    n_sigma = femsolver.get_node_values_fast(sigma, voxels, L)
    
    von_mises = femsolver.von_mises_stresses_node(n_sigma)
    von_mises_v = femsolver.get_voxel_values_fast(von_mises, voxels)
    State = {"u": u, "n_sigma": n_sigma, "von_mises": von_mises, "components": components}

    K_red, F_red = femsolver.fix_boundary_nodes_fast(list(fixed_nodes), K, F)

    return von_mises_v, State, K_red, F_red, list(fixed_nodes)

def recalculate_stress(voxels, K, F, L, E, nu, fixed_nodes, solver: femsolver.Solver, delta: scipy.sparse.csr_matrix =None):

    #u, components = femsolver.quick_solve(K_red, F_red, debug=True)
    #prev_u = None
    u, components = None, None
    if delta is None:
        u, components = solver.solve(K.tocsr(), F, fixed_nodes,debug=True)
    else:
        u, components = solver.solve(K.tocsr()+delta.tocsr(), F, fixed_nodes, debug=True)
        #u, components = solver.solve_perturbed_with_projection(K.tocsr(), delta.tocsr(), F, fixed_nodes)
        #u, components = solver.update_solution(K.tocsr(), delta.tocsr(), F, fixed_nodes, debug=True)
    
    if u is None:
        return None, None

    # Compute stresses and strains
    eps = femsolver.get_element_strains_fast(u, voxels, L)
    sigma = femsolver.get_element_stresses_fast(eps, E, nu)
    n_sigma = femsolver.get_node_values_fast(sigma, voxels, L)
    von_mises = femsolver.von_mises_stresses_node(n_sigma)
    von_mises_v = femsolver.get_voxel_values_fast(von_mises, voxels)
    State = {"u": u, "n_sigma": n_sigma, "von_mises": von_mises, "components": components}
    return von_mises_v, State

def forward_pass_AAA(objects: ObjectManager, grid: Grid, break_limit = 10_000_000, steps = 3, low_limit = 0.0, high_limit = 1.0):
    # Geometry optimization based on interval bisection

    E = 200e9  # Young's modulus (Pa)
    nu = 0.3   # Poisson's ratio
    L = 0.01    # Side length (m)
    t = 0.1   # Thickness (m)

    highest_stress = 0
    counter = 0
    Ke = femsolver.element_stiffness_matrix(E, nu, L, t)
    voxels = grid.getVoxels()

    _, _, K, F, fixed_nodes = calculate_stress(voxels, Ke, L, E, nu, grid, objects)
    threshold = 1e-10 * np.max(np.abs(K.data))

    K_red, F_red = femsolver.fix_boundary_nodes_fast(list(fixed_nodes), K, F)
    solver = femsolver.Solver(K_red, F_red)

    new_voxels = np.copy(voxels)
    old_voxels = new_voxels
    newK = K.copy()
    # Lock nodes with boundary conditions
    locked_voxels = grid.get_boundary_voxels(objects)
    locked_voxels.extend(grid.get_force_voxels(objects))

    
    delta = scipy.sparse.csr_matrix((K_red.shape[0], K_red.shape[1]), dtype=float)
    delta_acc = delta

    def calc_and_plot(voxels, newK, delta = None):
        von_mises, old_state = recalculate_stress(voxels, newK, F, L, E, nu, fixed_nodes, solver=solver, delta=delta)

        if von_mises is None:
            print("Failed to recalculate stress")
            return None
        
        if PLOT:
            von_mises_figure = femplotter.fast_value_plot(old_state["von_mises"], voxels)
            plt.title("calc_and_plot")
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

    von_mises_v, state = calc_and_plot(voxels, newK)
    old_state = state
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
            K = newK
            K_red, F_red = femsolver.fix_boundary_nodes_fast(list(fixed_nodes), K, F)
            solver.refactor(K_red.tocsc(), True)
            delta_acc = scipy.sparse.csr_matrix((K_red.shape[0], K_red.shape[1]), dtype=float)

        print("LIMIT: ", limit)
        print(highest_stress, " / ",break_limit)
        new_voxels = select_voxels(voxels, OG_VON_MISES, locked_indices, limit)
        newK, delta = femsolver.update_global_stiffness_matrix(K, old_voxels, new_voxels, Ke, threshold)
        delta_acc += delta
        
        von_mises_v, state = recalculate_stress(new_voxels, K, F, L, E, nu, fixed_nodes, solver=solver, delta=delta_acc)
        old_state = state
        if von_mises_v is None:
            break
        components = state["components"]
        counter += 1

    finalK, delta = femsolver.update_global_stiffness_matrix(K, old_voxels, old_voxels, Ke, threshold)
    von_mises_v, old_state = calc_and_plot(old_voxels, finalK, None)
    sorted_indices = np.argsort(von_mises_v, axis=None)
    highest_stress = von_mises_v.flatten()[sorted_indices[-1]]
    print(f"Break limit: {break_limit}, highest stress: {highest_stress}")
    return old_voxels, old_state, lower_limit, upper_limit

def forward_pass_BBB(objects: ObjectManager, grid: Grid, break_limit = 10_000_000, steps = 75, step_size = 0.01):
    # Geometry optimization based on interval bisection

    E = 200e9  # Young's modulus (Pa)
    nu = 0.3   # Poisson's ratio
    L = 0.01    # Side length (m)
    t = 0.1   # Thickness (m)

    highest_stress = 0
    counter = 0
    Ke = femsolver.element_stiffness_matrix(E, nu, L, t)
    voxels = grid.getVoxels()

    _, _, K, F, fixed_nodes = calculate_stress(voxels, Ke, L, E, nu, grid, objects)
    threshold = 1e-10 * np.max(np.abs(K.data))

    K_red, F_red = femsolver.fix_boundary_nodes_fast(list(fixed_nodes), K, F)
    solver = femsolver.Solver(K_red, F_red)
    #solver.refactor(K_red)

    new_voxels = np.copy(voxels)
    old_voxels = new_voxels
    newK = K.copy()
    # Lock nodes with boundary conditions
    locked_voxels = grid.get_boundary_voxels(objects)
    locked_voxels.extend(grid.get_force_voxels(objects))

    delta = scipy.sparse.csr_matrix((K_red.shape[0], K_red.shape[1]), dtype=float)
    delta_acc = delta.copy()

    def calc_and_plot(voxels, newK, delta = None):
        von_mises, old_state = recalculate_stress(voxels, newK, F, L, E, nu, fixed_nodes, solver=solver, delta=delta)

        if von_mises is None:
            print("Failed to recalculate stress")
            return None
        
        if PLOT:
            von_mises_figure = femplotter.fast_value_plot(old_state["von_mises"], voxels)
            plt.title("calc_and_plot")
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

    von_mises_v, state = calc_and_plot(voxels, newK)
    old_state = state
    OG_VON_MISES = von_mises_v

    desired_n_components = state["components"]
    components = desired_n_components
    limit = 1.0

    while counter < steps:
        sorted_indices = np.argsort(von_mises_v, axis=None)
        locked_indices = [v[1]*voxels.shape[1] + v[0] for v in locked_voxels]
        highest_stress = von_mises_v.flatten()[sorted_indices[-1]]
        disconnected = components != desired_n_components
        limit -= step_size

        if (highest_stress > break_limit or disconnected):
            print(f"Stress too high / disconnected: {highest_stress}")
            break
        elif limit <= 0.0:
            break
        else:
            OG_VON_MISES = von_mises_v
            old_voxels = new_voxels
            K = newK
            K_red, F_red = femsolver.fix_boundary_nodes_fast(list(fixed_nodes), K, F)
            solver.refactor(K_red.tocsc(), True)
            delta_acc = scipy.sparse.csr_matrix((K_red.shape[0], K_red.shape[1]), dtype=float)

        print("LIMIT: ", limit)
        print(highest_stress, " / ",break_limit)
        new_voxels = select_voxels(voxels, OG_VON_MISES, locked_indices, limit)
        newK, delta = femsolver.update_global_stiffness_matrix(K, old_voxels, new_voxels, Ke, threshold)
        delta_acc += delta
        #prev_u = old_state["u"]
        von_mises_v, state = recalculate_stress(new_voxels, K, F, L, E, nu, fixed_nodes, delta=delta_acc, solver=solver)
        if von_mises_v is None:
            break
        components = state["components"]
        counter += 1

    finalK, delta = femsolver.update_global_stiffness_matrix(K, old_voxels, old_voxels, Ke, threshold)
    delta_acc += delta
    von_mises_v, old_state = calc_and_plot(old_voxels, finalK, None)
    sorted_indices = np.argsort(von_mises_v, axis=None)
    highest_stress = von_mises_v.flatten()[sorted_indices[-1]]
    print(f"Break limit: {break_limit}, highest stress: {highest_stress}")
    return old_voxels, old_state

def opt(objects: ObjectManager):
    # Optimizatation of voxel geometry
    bbox = objects.get_bounding_box()
    # Define grid
    xdivs = 200
    ydivs = 20

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

    if PLOT:
        objects.draw()
        grid.draw()

        plt.gcf().axes[0].invert_yaxis()
        plt.margins(0.1)
        plt.show()

    # Run optimization - each iteration you get finer resolution of the solution
    # No gaurantee of correctness or convergence 
    for i in range(3):
        print(np.sum(voxels == 1), " voxels")
        grid = Grid(bbox[0], bbox[1], bbox[2], bbox[3], xdivs, ydivs)
        grid.voxels = voxels

        #objects.draw()
        #grid.draw()

        #plt.gcf().axes[0].invert_yaxis()
        #plt.margins(0.1)
        #plt.show()

        t1 =time.perf_counter()
        stepsize = 0.005
        if i <= 1:
            voxels, state = forward_pass_BBB(objects, grid, break_limit*adjust(i), 75, stepsize)
            #grid.voxels = voxels
            #voxels, state, low, high = forward_pass_AAA(objects, grid, break_limit*adjust(i), 20, 0.0, 1.0)
        else:
            voxels, state, low, high = forward_pass_AAA(objects, grid, break_limit*adjust(i), 20, 0.0, 1.0)
        t2 =time.perf_counter()
        print(f"Forward pass took {t2-t1} seconds")
        #grid.voxels = voxels
        
        voxels = femsolver.sub_divide(voxels, 2)
        xdivs *= 2
        ydivs *= 2

def test():
    # Test program

    # Construct geometry
    objects = ObjectManager()
    ground = Object2D(0, 0, 10, 1, 0)
    #platform = Object2D(6, 7, 5, 1, -45)
    #line3 = ForceLine((9.99, 0.5), (9.99, 0), 100, [1,0])
    line3 = ForceLine((0.01, 1), (0.01, 0), 50, [-1,0.5])
    line4 = ForceLine((9.99, 1), (9.99, 0), 50, [1,-0.5])
    #line5 = ForceLine((5.99, 1), (5.99, 0.5), 100, [0,1])
    #line3 = FixedLine((10, 0), (10, 1))
    #line4 = FixedLine((0.01, 0), (0.01, 1))
    line6 = FixedLine((5, 0.4), (5, 0.6))
    line7 = FixedLine((4.9, 0.5), (5.1, 0.5))

    
    objects.add(ground)
    #objects.add(platform)
    objects.add_line(line3)
    objects.add_line(line4)
    #objects.add_line(line4)
    #bjects.add_line(line5)
    objects.add_line(line6)
    objects.add_line(line7)
    if PLOT: plt.margins(y=1.5)
    opt(objects)
    
    if PLOT: plt.show()
    return

from cProfile import Profile
from pstats import SortKey, Stats

if __name__ == "__main__":
    with Profile() as profile:
        test()
        (
            Stats(profile)
            .strip_dirs()
            .sort_stats(SortKey.TIME)
            .print_stats(50)
        )
