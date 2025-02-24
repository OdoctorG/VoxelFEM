import math
import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.sparse
from geometry import *

import femsolver, femplotter, time

class ForwardPass:
    
    def __init__(self, objects: ObjectManager, grid: Grid, break_limit = 10_000_000, E = 200e9, nu = 0.3, L = 0.01, t = 0.1, plot = True, debug=False):
        """Creates a new ForwardPass object for use in the forward pass of the optimization.
        
        Parameters
        ----------
        objects : ObjectManager
            The object manager for the simulation
        grid : Grid
            The grid for the simulation
        break_limit : int, optional
            Maximum von Mises stress allowed before breaking a voxel, by default 10M Pa
        E : float, optional
            Young's modulus (Pa), by default 200e9
        nu : float, optional
            Poisson's ratio, by default 0.3
        L : float, optional
            Side length (m), by default 0.01
        t : float, optional
            Thickness (m), by default 0.1
        plot: bool, optional
            Whether or not to plot the resulting von mises stresses, by default True
        debug: bool, optional
            Whether or not to print debug information, by default False
        """
        
        self.E = E  # Young's modulus (Pa)
        self.nu = nu  # Poisson's ratio
        self.L = L    # Side length (m)
        self.t = t   # Thickness (m)
        self.objects = objects
        self.grid = grid
        self.break_limit = break_limit
        self.PLOT = plot
        self.debug = debug
    
    def _calculate_stress(self, voxels, Ke):
        # Calculate stresses based on voxel geometry

        E = self.E  # Young's modulus (Pa)
        nu = self.nu   # Poisson's ratio
        L = self.L    # Side length (m)
        t = self.t   # Thickness (m)
        grid = self.grid
        objects = self.objects
        

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
        
        u, components = femsolver.solve(K.tocsr(), F, fixed_nodes=list(fixed_nodes), debug=self.debug)
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

    def _recalculate_stress(self, voxels, K, F, fixed_nodes, solver: femsolver.Solver, delta: scipy.sparse.csr_matrix = None):
        E = self.E  # Young's modulus (Pa)
        nu = self.nu   # Poisson's ratio
        L = self.L    # Side length (m)
        t = self.t   # Thickness (m)
        grid = self.grid
        objects = self.objects
        #u, components = femsolver.quick_solve(K_red, F_red, debug=True)
        #prev_u = None
        u, components = None, None
        if delta is None:
            u, components = solver.solve(K.tocsr(), F, fixed_nodes,debug=self.debug)
        else:
            u, components = solver.solve(K.tocsr()+delta.tocsr(), F, fixed_nodes, debug=self.debug)
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


    def _calc_and_plot(self, voxels, newK, F, fixed_nodes, solver: femsolver.Solver, delta: scipy.sparse.csr_matrix = None):
        von_mises, old_state = self._recalculate_stress(voxels, newK, F, fixed_nodes, solver=solver, delta=delta)

        if von_mises is None:
            if self.debug: print("Failed to recalculate stress")
            return None
        
        if self.PLOT:
            von_mises_figure = femplotter.fast_value_plot(old_state["von_mises"], voxels)
            plt.title("calc_and_plot")
            plt.show()
        
        return von_mises, old_state

    def select_voxels(self, voxels, von_mises_v, locked_indices, percentage: float = 0.5):
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
    
    def forward_pass_A(self, steps = 3, low_limit = 0.0, high_limit = 1.0):
        # Geometry optimization based on interval bisection

        E = self.E  # Young's modulus (Pa)
        nu = self.nu   # Poisson's ratio
        L = self.L    # Side length (m)
        t = self.t   # Thickness (m)

        highest_stress = 0
        counter = 0
        Ke = femsolver.element_stiffness_matrix(E, nu, L, t)
        voxels = self.grid.getVoxels()

        _, _, K, F, fixed_nodes = self._calculate_stress(voxels, Ke)
        threshold = 1e-8 * np.max(np.abs(K.data))

        K_red, F_red = femsolver.fix_boundary_nodes_fast(list(fixed_nodes), K, F)
        solver = femsolver.Solver(K_red, F_red)

        new_voxels = np.copy(voxels)
        old_voxels = new_voxels
        newK = K.copy()
        # Lock nodes with boundary conditions
        locked_voxels = self.grid.get_boundary_voxels(self.objects)
        locked_voxels.extend(self.grid.get_force_voxels(self.objects))

        
        delta = scipy.sparse.csr_matrix((K_red.shape[0], K_red.shape[1]), dtype=float)
        delta_acc = delta

        von_mises_v, state = self._calc_and_plot(voxels, newK, F, fixed_nodes, solver)
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

            if (highest_stress > self.break_limit or disconnected) and limit == high_limit:
                if self.debug: print(f"Stress too high: {highest_stress}")
                break
            elif highest_stress >= self.break_limit or disconnected: # Interval splitting search
                lower_limit = limit
                limit += (upper_limit-limit)*0.5
            elif highest_stress < self.break_limit:
                OG_VON_MISES = von_mises_v
                old_voxels = new_voxels
                upper_limit = limit
                limit -= (limit-lower_limit)*0.5
                lower_limit *= 0.9
                K = newK
                K_red, F_red = femsolver.fix_boundary_nodes_fast(list(fixed_nodes), K, F)
                solver.refactor(K_red.tocsc(), True)
                delta_acc = scipy.sparse.csr_matrix((K_red.shape[0], K_red.shape[1]), dtype=float)

            if self.debug: 
                print("LIMIT: ", limit)
                print(highest_stress, " / ", self.break_limit)
            
            new_voxels = self.select_voxels(voxels, OG_VON_MISES,locked_indices, limit)
            newK, delta = femsolver.update_global_stiffness_matrix(K, old_voxels, new_voxels, Ke, threshold)
            delta_acc += delta
            
            von_mises_v, state = self._recalculate_stress(new_voxels, K, F, fixed_nodes, solver=solver, delta=delta_acc)
            old_state = state
            if von_mises_v is None:
                break
            components = state["components"]
            counter += 1
        
        finalK, delta = femsolver.update_global_stiffness_matrix(K, old_voxels, old_voxels, Ke, threshold)
        von_mises_v, old_state = self._calc_and_plot(old_voxels, finalK, F, fixed_nodes, solver, None)
        sorted_indices = np.argsort(von_mises_v, axis=None)
        highest_stress = von_mises_v.flatten()[sorted_indices[-1]]

        if self.debug: 
            print(f"Break limit: {self.break_limit}, highest stress: {highest_stress}")
        
        return old_voxels, old_state, lower_limit, upper_limit

    def forward_pass_B(self, steps = 75, step_size = 0.01):
        # Geometry optimization based on interval bisection

        E = self.E  # Young's modulus (Pa)
        nu = self.nu   # Poisson's ratio
        L = self.L    # Side length (m)
        t = self.t   # Thickness (m)

        highest_stress = 0
        counter = 0
        Ke = femsolver.element_stiffness_matrix(E, nu, L, t)
        voxels = self.grid.getVoxels()

        _, _, K, F, fixed_nodes = self._calculate_stress(voxels, Ke)
        threshold = 1e-8 * np.max(np.abs(K.data))

        K_red, F_red = femsolver.fix_boundary_nodes_fast(list(fixed_nodes), K, F)
        solver = femsolver.Solver(K_red, F_red)

        new_voxels = np.copy(voxels)
        old_voxels = new_voxels
        newK = K.copy()
        # Lock nodes with boundary conditions
        locked_voxels = self.grid.get_boundary_voxels(self.objects)
        locked_voxels.extend(self.grid.get_force_voxels(self.objects))

        
        delta = scipy.sparse.csr_matrix((K_red.shape[0], K_red.shape[1]), dtype=float)
        delta_acc = delta.copy()

        von_mises_v, state = self._calc_and_plot(voxels, newK, F, fixed_nodes, solver)
        old_state = state
        OG_VON_MISES = von_mises_v

        desired_n_components = state["components"]
        components = desired_n_components

        desired_n_components = state["components"]
        components = desired_n_components
        limit = 1.0

        while counter < steps:
            sorted_indices = np.argsort(von_mises_v, axis=None)
            locked_indices = [v[1]*voxels.shape[1] + v[0] for v in locked_voxels]
            highest_stress = von_mises_v.flatten()[sorted_indices[-1]]
            disconnected = components != desired_n_components
            limit -= step_size

            if (highest_stress > self.break_limit or disconnected):
                if self.debug: 
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

            if self.debug: 
                print("LIMIT: ", limit)
                print(highest_stress, " / ", self.break_limit)
            
            new_voxels = self.select_voxels(voxels, OG_VON_MISES, locked_indices, limit)
            newK, delta = femsolver.update_global_stiffness_matrix(K, old_voxels, new_voxels, Ke, threshold)
            delta_acc += delta
            #prev_u = old_state["u"]
            von_mises_v, state = self._recalculate_stress(new_voxels, K, F, fixed_nodes, delta=delta_acc, solver=solver)
            if von_mises_v is None:
                break
            components = state["components"]
            counter += 1

        finalK, delta = femsolver.update_global_stiffness_matrix(K, old_voxels, old_voxels, Ke, threshold)
        delta_acc += delta
        von_mises_v, old_state = self._calc_and_plot(old_voxels, finalK, F, fixed_nodes, solver)
        sorted_indices = np.argsort(von_mises_v, axis=None)
        highest_stress = von_mises_v.flatten()[sorted_indices[-1]]
        
        if self.debug: 
            print(f"Break limit: {self.break_limit}, highest stress: {highest_stress}")
        
        return old_voxels, old_state

