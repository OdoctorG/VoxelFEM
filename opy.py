""" Geometry Optimization -- Main Module """

import numpy as np
import matplotlib.pyplot as plt
from geometry import *

import femsolver_cont as femsolver
import time
import forward_pass

PLOT = True

def opt(objects: ObjectManager):
    # Optimizatation of voxel geometry
    bbox = objects.get_bounding_box()
    # Define grid
    xdivs = 40
    ydivs = 20

    # Adjustment factor for break limit 
    # - the break limit approaches the true break limit with each iteration of i
    def adjust(x):
        #return 1.0
        return 1/(10*2**(-x)+1)
        #return (x+1)/(x+1.5)
    
    # Setup grid and voxels
    grid = Grid(bbox[0], bbox[1], bbox[2], bbox[3], xdivs, ydivs)
    grid.add_voxels_inside(objects)
    voxels = grid.getVoxels()

    # 200 Mpa stress limit for testing
    break_limit = 300_000_000

    if PLOT:
        objects.draw()
        grid.draw()

        plt.gcf().axes[0].invert_yaxis()
        plt.margins(0.1)
        plt.show()

    # Run optimization - each iteration you get finer resolution of the solution
    # No gaurantee of correctness or convergence 
    for i in range(5):
        print(np.sum(voxels == 1), " voxels")
        grid = Grid(bbox[0], bbox[1], bbox[2], bbox[3], xdivs, ydivs)
        grid.voxels = voxels

        # Create Forward Pass object
        FP = forward_pass.ForwardPass(objects, grid, break_limit*adjust(i), plot=PLOT, debug=True)
        t1 =time.perf_counter()

        # Run forward pass, first with only method A and then alternating between method A and B
        stepsize = 0.005
        if i <= 2:
            voxels, state = FP.forward_pass_B(75, stepsize)
        else:
            voxels, state, low, high = FP.forward_pass_A(10, 0.0, 1.0)
            grid.voxels = voxels
            FP = forward_pass.ForwardPass(objects, grid, break_limit*adjust(i), plot=PLOT, debug=True)
            voxels, state = FP.forward_pass_B(75, stepsize)
        t2 =time.perf_counter()
        print(f"Forward pass took {t2-t1} seconds")
        
        # Subdivide voxels to get finer solution
        voxels = femsolver.sub_divide(voxels, 2)
        xdivs *= 2
        ydivs *= 2

def bar_test():
    # Test program

    # Construct geometry
    objects = ObjectManager()
    ground = Object2D(0, 0, 10, 1, 0)
    line3 = ForceLine((0.01, 1), (0.01, 0), 50000, [-1,0.5])
    line4 = ForceLine((9.99, 1), (9.99, 0), 50000, [1,-0.5])
    line6 = FixedLine((5, 0.4), (5, 0.6))
    line7 = FixedLine((4.9, 0.5), (5.1, 0.5))

    
    objects.add(ground)
    objects.add_line(line3)
    objects.add_line(line4)
    objects.add_line(line6)
    objects.add_line(line7)

    if PLOT: 
        plt.margins(x=1.1, y=1.5)

    opt(objects)
    
    if PLOT:
        plt.show()
    return

def bridge_test():
    # Construct geometry
    objects = ObjectManager()
    base = Object2D(-0.5, -0.25, 11, 5.5, 0)
    line1 = ForceLine((0.0, 0), (10.0, 0), 1000, [0,1])
    line2 = FixedLine((0.01, 1.0), (0.01, 4.0))
    line3 = FixedLine((9.99, 1.0), (9.99, 4.0))

    
    objects.add(base)
    objects.add_line(line1)
    objects.add_line(line2)
    objects.add_line(line3)

    if PLOT: plt.margins(x=1.1, y=1.5)

    opt(objects)
    
    if PLOT: plt.show()
    return

from cProfile import Profile
from pstats import SortKey, Stats

if __name__ == "__main__":
    with Profile() as profile:
        bridge_test()
        (
            Stats(profile)
            .strip_dirs()
            .sort_stats(SortKey.TIME)
            .print_stats(50)
        )
