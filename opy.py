import math
import numpy as np
import matplotlib.pyplot as plt


ax = plt.axes()

class Object2D:
    x: float
    y: float
    w: float
    h: float

    rot: float
    def __init__(self, x: float, y: float, w: float, h: float, rot: float =0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.rot = rot
    
    def draw(self):
        rect = plt.Rectangle((self.x, self.y), self.w, self.h, angle=self.rot, fc='b', ec='k')
        plt.gca().add_patch(rect)
        plt.axis('scaled')
    
    def pointInside(self, x: float, y: float) -> bool:
        s = np.sin(np.radians(-self.rot))
        c = np.cos(np.radians(-self.rot))
        x_rot = (x - self.x) * c - (y - self.y) * s
        y_rot = (x - self.x) * s + (y - self.y) * c
        #print(x, y, " inside: ", x_rot >= 0 and x_rot <= self.w and y_rot >= 0 and y_rot <= self.h)
        return x_rot >= 0 and x_rot <= self.w and y_rot >= 0 and y_rot <= self.h
    
    def getCorners(self) -> list[tuple[float, float]]:
        s = np.sin(np.radians(self.rot))
        c = np.cos(np.radians(self.rot))
        corners = []
        corners.append((self.x, self.y))
        corners.append((self.x + self.w, self.y))
        corners.append((self.x + self.w, self.y + self.h))
        corners.append((self.x, self.y + self.h))
        for i in range(len(corners)):
            x_rot = (corners[i][0] - self.x) * c - (corners[i][1] - self.y) * s
            y_rot = (corners[i][0] - self.x) * s + (corners[i][1] - self.y) * c
            corners[i] = (x_rot+ self.x, y_rot+ self.y)
        return corners

class Line:
    p1: tuple[float, float]
    p2: tuple[float, float]
    def __init__(self, p1: tuple[float, float], p2: tuple[float, float]):
        self.p1 = p1
        self.p2 = p2
    
    def draw(self, color='k'):
        plt.plot([self.p1[0], self.p2[0]], [self.p1[1], self.p2[1]], color, linewidth=3)

class ForceLine(Line):
    force: float
    force_dir: np.ndarray
    def __init__(self, p1: tuple[float, float], p2: tuple[float, float], force: float, force_dir: np.ndarray = None, flip_force: bool = False):
        super().__init__(p1, p2)
        if force_dir is None:
            force_dir = np.array([self.p1[1] - self.p2[1], self.p2[0] - self.p1[0]])
        if flip_force:
            force_dir = -force_dir
        
        self.force = force
        self.force_dir = force_dir / np.linalg.norm(force_dir)
    
    def draw(self):
        super().draw('r')
        center = np.array([(self.p1[0] + self.p2[0]) / 2, (self.p1[1] + self.p2[1]) / 2])
        plt.arrow(center[0], center[1], self.force_dir[0], self.force_dir[1], color='r', head_width=0.2, head_length=0.2, width=0.05)

class FixedLine(Line):
    def __init__(self, p1: tuple[float, float], p2: tuple[float, float]):
        super().__init__(p1, p2)
    def draw(self):
        super().draw('g')

class ObjectManager:
    objects: list[Object2D]
    lines: list[Line]
    def __init__(self):
        self.objects = []
        self.lines = []
    
    def add(self, obj: Object2D):
        self.objects.append(obj)
    
    def addLine(self, line: Line):
        self.lines.append(line)
    
    def pointInside(self, x: float, y: float) -> int:
        i = 0
        for obj in self.objects:
            if obj.pointInside(x, y):
                return i
            i += 1
        return -1
    
    def draw(self):
        for obj in self.objects:
            obj.draw()
        for lines in self.lines:
            lines.draw()
    
    def clear(self):
        plt.clf()

    def getBoundingBox(self) -> tuple[float, float, float, float]:
        minx = 1000
        miny = 1000
        maxx = 0
        maxy = 0
        for obj in self.objects:
            if obj.x < minx:
                minx = obj.x
            if obj.y < miny:
                miny = obj.y
            if obj.x + obj.w > maxx:
                maxx = obj.x + obj.w
            if obj.y + obj.h > maxy:
                maxy = obj.y + obj.h
        for line in self.lines:
            if line.p1[0] < minx:
                minx = line.p1[0]
            if line.p1[1] < miny:
                miny = line.p1[1]
            if line.p2[0] > maxx:
                maxx = line.p2[0]
            if line.p2[1] > maxy:
                maxy = line.p2[1]
        return minx, miny, maxx, maxy

class Grid:
    LINE_WIDTH = 0.1
    minx: float = 0
    miny: float = 0
    maxx: float = 10
    maxy: float = 10
    xdivs: int = 2
    ydivs: int = 2
    voxels: np.ndarray

    def __init__(self, minx: float, miny: float, maxx: float, maxy: float, xdivs: int = 2, ydivs: int = 2): 
        self.minx = minx
        self.miny = miny
        self.maxx = maxx
        self.maxy = maxy
        self.xdivs = xdivs
        self.ydivs = ydivs
        self.voxels = np.zeros((ydivs, xdivs), dtype=np.uint8)
    
    def draw(self):
        for i in range(self.ydivs + 1):
            plt.plot(
                [self.minx, self.maxx], 
                [self.miny + i * (self.maxy - self.miny) / self.ydivs, self.miny + i * (self.maxy - self.miny) / self.ydivs], 
                'k',
                linewidth=self.LINE_WIDTH
            )
        for i in range(self.xdivs + 1):
            plt.plot(
                [self.minx + i * (self.maxx - self.minx) / self.xdivs, self.minx + i * (self.maxx - self.minx) / self.xdivs], 
                [self.miny, self.maxy], 
                'k', 
                linewidth=self.LINE_WIDTH)
        for i in range(self.xdivs):
            for j in range(self.ydivs):
                if self.voxels[j, i] == 1:
                    rect = plt.Rectangle(
                        [self.minx + i * (self.maxx - self.minx) / self.xdivs, self.miny + j * (self.maxy - self.miny) / self.ydivs],
                        (self.maxx - self.minx) / self.xdivs,
                        (self.maxy - self.miny) / self.ydivs
                    )
                    rect.set_facecolor((0,0,0,0.25))
                    plt.gca().add_patch(rect)

    # Check if point (x, y) is inside voxel[i, j]
    def pointInside(self, x: float, y: float, i: int, j: int) -> bool:
        return x >= self.minx + i * (self.maxx - self.minx) / self.xdivs and \
            x <= self.minx + (i + 1) * (self.maxx - self.minx) / self.xdivs and \
            y >= self.miny + j * (self.maxy - self.miny) / self.ydivs and \
            y <= self.miny + (j + 1) * (self.maxy - self.miny) / self.ydivs
    
    def getVoxelCenter(self, i: int, j: int) -> tuple[float, float]:
        voxel_minx = self.minx + i * (self.maxx - self.minx) / self.xdivs
        voxel_maxx = voxel_minx + (self.maxx - self.minx) / self.xdivs
        voxel_miny = self.miny + j * (self.maxy - self.miny) / self.ydivs
        voxel_maxy = voxel_miny + (self.maxy - self.miny) / self.ydivs

        voxel_center_x = (voxel_minx + voxel_maxx) / 2
        voxel_center_y = (voxel_miny + voxel_maxy) / 2

        return voxel_center_x, voxel_center_y

    def doLineSegmentsIntersect(self, p1, p2, p3, p4):
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    
    def doesVoxelIntersectLine(self, i: int, j: int, p1: tuple[float, float], p2: tuple[float, float]) -> bool:
        voxel_minx = self.minx + i * (self.maxx - self.minx) / self.xdivs
        voxel_maxx = voxel_minx + (self.maxx - self.minx) / self.xdivs
        voxel_miny = self.miny + j * (self.maxy - self.miny) / self.ydivs
        voxel_maxy = voxel_miny + (self.maxy - self.miny) / self.ydivs

        if self.pointInside(p1[0], p1[1], i, j) or self.pointInside(p2[0], p2[1], i, j):
            return True
        if self.doLineSegmentsIntersect((voxel_minx, voxel_miny), (voxel_maxx, voxel_miny), p1, p2) or \
            self.doLineSegmentsIntersect((voxel_maxx, voxel_miny), (voxel_maxx, voxel_maxy), p1, p2) or \
            self.doLineSegmentsIntersect((voxel_maxx, voxel_maxy), (voxel_minx, voxel_maxy), p1, p2) or \
            self.doLineSegmentsIntersect((voxel_minx, voxel_maxy), (voxel_minx, voxel_miny), p1, p2):
            return True
        return False
    
    # Function sets voxel[i, j]=1 if the voxel has any object inside of it
    def addVoxelsInside(self, objects: ObjectManager):
        for obj in objects.objects:
            corners = obj.getCorners()
            for i in range(self.xdivs):
                for j in range(self.ydivs):
                    voxel_minx = self.minx + i * (self.maxx - self.minx) / self.xdivs
                    voxel_maxx = voxel_minx + (self.maxx - self.minx) / self.xdivs
                    voxel_miny = self.miny + j * (self.maxy - self.miny) / self.ydivs
                    voxel_maxy = voxel_miny + (self.maxy - self.miny) / self.ydivs

                    voxel_center_x, voxel_center_y = self.getVoxelCenter(i, j)

                    # Check if voxel center is inside object
                    if obj.pointInside(voxel_center_x, voxel_center_y):
                        self.voxels[j, i] = 1
                        continue

                    # Check if voxel intersects with object edges
                    for k in range(len(corners)):
                        p1 = corners[k]
                        p2 = corners[(k+1) % len(corners)]  # wrap around to first corner after last corner
                        if self.doesVoxelIntersectLine(i, j, p1, p2) or \
                        self.pointInside(p1[0], p1[1], i, j) or self.pointInside(p2[0], p2[1], i, j):
                            self.voxels[j, i] = 1
                            break
        
        for lines in objects.lines:
            for i in range(self.xdivs):
                for j in range(self.ydivs):
                    if self.doesVoxelIntersectLine(i, j, lines.p1, lines.p2):
                        self.voxels[j, i] = 1

    def getVoxels(self) -> np.ndarray:
        return self.voxels
    
    def getVoxelsIntersectingLine(self, line: Line) -> list:
        # Get all voxels that intersect with the line
        res = []
        for i in range(self.xdivs):
            for j in range(self.ydivs):
                if self.doesVoxelIntersectLine(i, j, line.p1, line.p2):
                    res.append((i, j))
        return res
    
    def getForceVoxels(self, objects: ObjectManager) -> list:
        # Get all voxels that have a force applied to them
        res = []
        for line in objects.lines:
            if not isinstance(line, ForceLine):
                continue
            for i in range(self.xdivs):
                for j in range(self.ydivs):
                    if self.doesVoxelIntersectLine(i, j, line.p1, line.p2):
                        res.append((i, j))
        return res
    
    def getBoundaryVoxels(self, objects: ObjectManager) -> list:
        # Get all voxels that are fixed
        res = []
        for line in objects.lines:
            if not isinstance(line, FixedLine):
                continue
            for i in range(self.xdivs):
                for j in range(self.ydivs):
                    if self.doesVoxelIntersectLine(i, j, line.p1, line.p2):
                        res.append((i, j))
        return res

import femsolver, femplotter, time

def calculate_stress(voxels, Ke, L, E, nu, grid: Grid, objects: ObjectManager, cond_limit=None):
    K = femsolver.global_stiffness_matrix(Ke, voxels)
    n_dofs = K.shape[0]

    F = np.zeros((n_dofs, 1))
    u = np.zeros((n_dofs, 1))
    #F = femsolver.add_force_to_node(4, F, np.array([0, 0.5]))

    fixed_nodes = set()
    BoundaryVoxels = grid.getBoundaryVoxels(objects)
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
        force_voxels = grid.getVoxelsIntersectingLine(force_line)
        for voxel in force_voxels:
            F = femsolver.add_force_to_voxel(voxel[1], voxel[0], voxels.shape[1], F, force)
    
    u, cond = femsolver.solve(K.tocsr(), F, fixed_nodes=list(fixed_nodes), debug=True, max_cond=cond_limit)
    if u is None:
        return None, None

    # Compute stresses and strains
    eps = femsolver.get_element_strains(u, voxels, L)
    sigma = femsolver.get_element_stresses(eps, E, nu)
    n_sigma = femsolver.get_node_values(sigma, voxels, L)
    
    von_mises = femsolver.von_mises_stresses_node(n_sigma)
    von_mises_v = femsolver.get_voxel_values(von_mises, voxels)
    State = {"u": u, "n_sigma": n_sigma, "von_mises": von_mises, "cond": cond}
    return von_mises_v, State



def forward_pass(objects: ObjectManager, grid: Grid, break_limit = 10_000_000, stepsize = 0.2, disconnect_counter = 1):
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
    locked_voxels = grid.getBoundaryVoxels(objects)
    locked_voxels.extend(grid.getForceVoxels(objects))

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

    calc_and_plot(old_voxels)

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
        elif state["cond"] > 1e12 and counter > disconnect_counter:
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

    print(f"Break limit: {break_limit}, highest stress: {old_highest_stress}")
    von_mises, old_state = calc_and_plot(old_voxels)
    return old_voxels, old_state

def opt(objects: ObjectManager):
    bbox = objects.getBoundingBox()
    xdivs = 200
    ydivs = 20

    def adjust(x):
        return (x+1)/(x+1.5)
    
    grid = Grid(bbox[0], bbox[1], bbox[2], bbox[3], xdivs, ydivs)
    grid.addVoxelsInside(objects)
    voxels = grid.getVoxels()
    break_limit = 200_000_000

    for i in range(3):
        print(np.sum(voxels == 1), " voxels")
        grid = Grid(bbox[0], bbox[1], bbox[2], bbox[3], xdivs, ydivs)
        grid.voxels = voxels

        objects.draw()
        grid.draw()

        plt.gcf().axes[0].invert_yaxis()
        plt.margins(0.1)
        plt.show()
        if i == 0:
            voxels, state = forward_pass(objects, grid, break_limit*adjust(i), stepsize=0.05, disconnect_counter=-1)
        else:
            voxels, state = forward_pass(objects, grid, break_limit*adjust(i), stepsize=0.01, disconnect_counter=-1)
        #print(voxels)
        voxels = femsolver.sub_divide(voxels, 2)
        xdivs *= 2
        ydivs *= 2


def test():
    # Construct geometry
    objects = ObjectManager()
    ground = Object2D(0, 0, 10, 1, 0)
    #platform = Object2D(6, 7, 5, 1, -45)
    line2 = ForceLine((0.01, 0), (0.01, 1), 1000, [1,0])
    line3 = FixedLine((8, 0), (10, 0))
    
    objects.add(ground)
    #objects.add(platform)
    objects.addLine(line2)
    objects.addLine(line3)
    plt.margins(y=1.5)
    opt(objects)
    
    plt.show()
    return

if __name__ == "__main__":
    test()
