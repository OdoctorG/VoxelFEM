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

class ObjectManager:
    objects: list[Object2D]
    def __init__(self):
        self.objects = []
    
    def add(self, obj: Object2D):
        self.objects.append(obj)
    
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
    
    

    def doLineSegmentsIntersect(self, p1, p2, p3, p4):
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
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

                    voxel_center_x = (voxel_minx + voxel_maxx) / 2
                    voxel_center_y = (voxel_miny + voxel_maxy) / 2

                    # Check if voxel center is inside object
                    if obj.pointInside(voxel_center_x, voxel_center_y):
                        self.voxels[j, i] = 1
                        continue

                    # Check if voxel intersects with object edges
                    for k in range(len(corners)):
                        p1 = corners[k]
                        p2 = corners[(k+1) % len(corners)]  # wrap around to first corner after last corner
                        if self.doLineSegmentsIntersect(p1, p2, (voxel_minx, voxel_miny), (voxel_maxx, voxel_miny)) or \
                        self.doLineSegmentsIntersect(p1, p2, (voxel_maxx, voxel_miny), (voxel_maxx, voxel_maxy)) or \
                        self.doLineSegmentsIntersect(p1, p2, (voxel_maxx, voxel_maxy), (voxel_minx, voxel_maxy)) or \
                        self.doLineSegmentsIntersect(p1, p2, (voxel_minx, voxel_maxy), (voxel_minx, voxel_miny)) or \
                        self.pointInside(p1[0], p1[1], i, j) or self.pointInside(p2[0], p2[1], i, j):
                            self.voxels[j, i] = 1
                            break

    def getVoxels(self) -> np.ndarray:
        return self.voxels

def test():
    # Construct geometry
    objects = ObjectManager()
    ground = Object2D(0, 0, 10, 1, 0)
    platform = Object2D(6, 7, 5, 1, -45)
    objects.add(ground)
    objects.add(platform)
    objects.draw()

    bbox = objects.getBoundingBox()
    grid = Grid(bbox[0], bbox[1], bbox[2], bbox[3], 100, 50)
    grid.addVoxelsInside(objects)
    grid.draw()

    # Solve FEM
    voxels = grid.getVoxels()
    import femsolver, femplotter, time

    E = 200e9  # Young's modulus (Pa)
    nu = 0.3   # Poisson's ratio
    L = 0.01    # Side length (m)
    t = 0.1   # Thickness (m)

    print("Setting up")

    t1 = time.perf_counter()
    Ke = femsolver.element_stiffness_matrix(E, nu, L, t)
    t2 = time.perf_counter()
    K = femsolver.global_stiffness_matrix(Ke, voxels)
    t3 = time.perf_counter()

    print(f"Element stiffeness took {t2-t1} seconds, global stiffnes matrix took {t3-t2} seconds")

    n_dofs = K.shape[0]

    F = np.zeros((n_dofs, 1))
    F = femsolver.add_force_to_node(4, F, np.array([0, 0.5]))

    print("Solving")
    u = femsolver.solve(K.tocsr(), F, [20, 21, 22, 23, 24])
    
    # Compute stresses and strains
    eps = femsolver.get_element_strains(u, voxels, L)
    sigma = femsolver.get_element_stresses(eps, E, nu)
    n_sigma = femsolver.get_node_values(sigma, voxels, L)

    print("Plot displaced mesh")
    # Plot the displacements
    femplotter.plot_displaced_mesh(u, voxels, new_figure=True)

    # Plot the von_mises stresses
    print("Get stresses in nodes")
    von_mises = femsolver.von_mises_stresses_node(n_sigma)
    print("plot stresses")
    von_mises_figure = femplotter.node_value_plot(von_mises, voxels)
    von_mises_figure.suptitle("von Mises stresses")
    print("plot show (render)")
    plt.show()

if __name__ == "__main__":
    test()
