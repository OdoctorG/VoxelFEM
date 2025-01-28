""" Module for handling problem geometry, grid and voxelization """

import numpy as np
import matplotlib.pyplot as plt


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
    
    def point_inside(self, x: float, y: float) -> bool:
        s = np.sin(np.radians(-self.rot))
        c = np.cos(np.radians(-self.rot))
        x_rot = (x - self.x) * c - (y - self.y) * s
        y_rot = (x - self.x) * s + (y - self.y) * c
        #print(x, y, " inside: ", x_rot >= 0 and x_rot <= self.w and y_rot >= 0 and y_rot <= self.h)
        return x_rot >= 0 and x_rot <= self.w and y_rot >= 0 and y_rot <= self.h
    
    def get_corners(self) -> list[tuple[float, float]]:
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
    
    def add_line(self, line: Line):
        self.lines.append(line)
    
    def point_inside(self, x: float, y: float) -> int:
        i = 0
        for obj in self.objects:
            if obj.point_inside(x, y):
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

    def get_bounding_box(self) -> tuple[float, float, float, float]:
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
    def point_inside(self, x: float, y: float, i: int, j: int) -> bool:
        return x >= self.minx + i * (self.maxx - self.minx) / self.xdivs and \
            x <= self.minx + (i + 1) * (self.maxx - self.minx) / self.xdivs and \
            y >= self.miny + j * (self.maxy - self.miny) / self.ydivs and \
            y <= self.miny + (j + 1) * (self.maxy - self.miny) / self.ydivs
    
    def get_voxel_center(self, i: int, j: int) -> tuple[float, float]:
        voxel_minx = self.minx + i * (self.maxx - self.minx) / self.xdivs
        voxel_maxx = voxel_minx + (self.maxx - self.minx) / self.xdivs
        voxel_miny = self.miny + j * (self.maxy - self.miny) / self.ydivs
        voxel_maxy = voxel_miny + (self.maxy - self.miny) / self.ydivs

        voxel_center_x = (voxel_minx + voxel_maxx) / 2
        voxel_center_y = (voxel_miny + voxel_maxy) / 2

        return voxel_center_x, voxel_center_y

    def do_line_segments_intersect(self, p1, p2, p3, p4):
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    
    def does_voxel_intersect_line(self, i: int, j: int, p1: tuple[float, float], p2: tuple[float, float]) -> bool:
        voxel_minx = self.minx + i * (self.maxx - self.minx) / self.xdivs
        voxel_maxx = voxel_minx + (self.maxx - self.minx) / self.xdivs
        voxel_miny = self.miny + j * (self.maxy - self.miny) / self.ydivs
        voxel_maxy = voxel_miny + (self.maxy - self.miny) / self.ydivs

        if self.point_inside(p1[0], p1[1], i, j) or self.point_inside(p2[0], p2[1], i, j):
            return True
        if self.do_line_segments_intersect((voxel_minx, voxel_miny), (voxel_maxx, voxel_miny), p1, p2) or \
            self.do_line_segments_intersect((voxel_maxx, voxel_miny), (voxel_maxx, voxel_maxy), p1, p2) or \
            self.do_line_segments_intersect((voxel_maxx, voxel_maxy), (voxel_minx, voxel_maxy), p1, p2) or \
            self.do_line_segments_intersect((voxel_minx, voxel_maxy), (voxel_minx, voxel_miny), p1, p2):
            return True
        return False
    
    # Function sets voxel[i, j]=1 if the voxel has any object inside of it
    def add_voxels_inside(self, objects: ObjectManager):
        for obj in objects.objects:
            corners = obj.get_corners()
            for i in range(self.xdivs):
                for j in range(self.ydivs):
                    voxel_minx = self.minx + i * (self.maxx - self.minx) / self.xdivs
                    voxel_maxx = voxel_minx + (self.maxx - self.minx) / self.xdivs
                    voxel_miny = self.miny + j * (self.maxy - self.miny) / self.ydivs
                    voxel_maxy = voxel_miny + (self.maxy - self.miny) / self.ydivs

                    voxel_center_x, voxel_center_y = self.get_voxel_center(i, j)

                    # Check if voxel center is inside object
                    if obj.point_inside(voxel_center_x, voxel_center_y):
                        self.voxels[j, i] = 1
                        continue

                    # Check if voxel intersects with object edges
                    for k in range(len(corners)):
                        p1 = corners[k]
                        p2 = corners[(k+1) % len(corners)]  # wrap around to first corner after last corner
                        if self.does_voxel_intersect_line(i, j, p1, p2) or \
                        self.point_inside(p1[0], p1[1], i, j) or self.point_inside(p2[0], p2[1], i, j):
                            self.voxels[j, i] = 1
                            break
        
        for lines in objects.lines:
            for i in range(self.xdivs):
                for j in range(self.ydivs):
                    if self.does_voxel_intersect_line(i, j, lines.p1, lines.p2):
                        self.voxels[j, i] = 1

    def getVoxels(self) -> np.ndarray:
        return self.voxels
    
    def get_voxels_intersecting_line(self, line: Line) -> list:
        # Get all voxels that intersect with the line
        res = []
        for i in range(self.xdivs):
            for j in range(self.ydivs):
                if self.does_voxel_intersect_line(i, j, line.p1, line.p2):
                    res.append((i, j))
        return res
    
    def get_force_voxels(self, objects: ObjectManager) -> list:
        # Get all voxels that have a force applied to them
        res = []
        for line in objects.lines:
            if not isinstance(line, ForceLine):
                continue
            for i in range(self.xdivs):
                for j in range(self.ydivs):
                    if self.does_voxel_intersect_line(i, j, line.p1, line.p2):
                        res.append((i, j))
        return res
    
    def get_boundary_voxels(self, objects: ObjectManager) -> list:
        # Get all voxels that are fixed
        res = []
        for line in objects.lines:
            if not isinstance(line, FixedLine):
                continue
            for i in range(self.xdivs):
                for j in range(self.ydivs):
                    if self.does_voxel_intersect_line(i, j, line.p1, line.p2):
                        res.append((i, j))
        return res
