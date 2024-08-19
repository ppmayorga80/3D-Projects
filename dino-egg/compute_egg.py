import os

import cv2
import numpy as np
import logging
from stl import mesh
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


class ComputeEgg:
    CONTOUR_PATH = os.path.join(os.path.dirname(__file__), "path5.jpg")
    OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "egg.stl")
    POW = 2.5
    POINTS = 16
    EGG_X_SIZE = 48.7
    EGG_Y_SIZE = 45.3
    EGG_Z_SIZE = 51.8

    def __init__(self, contour_image_bw_path=CONTOUR_PATH,
                 exponent: float = POW,
                 number_of_points_per_quadrant: int = POINTS,
                 output_path=OUTPUT_PATH):
        """read the input path to calculate the contour (should be a BW image of the YZ contour - around X axis)
        then use the equation $(x/a)^n+(y/b)^n=1$ to compute the vertices (x,y,z)
        of the egg.
        """
        # for some reason every time we process this file, the input image becomes corrupted
        # os.system("cp path5-copy.jpg path5.jpg")

        self.input_contour_image_path = contour_image_bw_path
        self.exponent = exponent
        self.m = number_of_points_per_quadrant
        self.output_path = output_path

        self.binary_image: np.ndarray = np.zeros((1, 1))
        self.z_a: np.ndarray = np.zeros((1, 1))
        self.z: np.ndarray = np.zeros((1, 1))
        self.x: np.ndarray = np.zeros((1, 1))
        self.y: np.ndarray = np.zeros((1, 1))

        self.vertices: np.ndarray = np.zeros((1, 2))
        self.faces: np.ndarray = np.zeros((1, 2))
        self.egg: mesh.Mesh

    def run(self):
        self.read_image(path=self.input_contour_image_path)
        self.image_to_z_a()
        self.compute_xyz()
        self.build_mesh()

    def read_image(self, path: str) -> np.ndarray:
        image = cv2.imread(path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, bw_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
        self.binary_image = bw_image
        return self.binary_image

    def image_to_z_a(self) -> np.ndarray:
        rows, columns = self.binary_image.shape

        za = []
        for r in range(rows):
            z_r = rows - r - 1
            a_r = max([c for b_r, c in zip(self.binary_image[r, :], range(columns)) if b_r == 0])

            if a_r == 0:
                logging.warning(f"have x-radius equals 0 at image-row index:{r}")

            za.append((z_r, a_r))

        self.z_a = np.array(za)
        return self.z_a

    def compute_xyz(self):

        a_zz, a_xx, a_yy = [], [], []
        for z, a in tqdm(self.z_a, desc="Computing x,y,z"):
            a_zz.append(z)
            b = a / self.EGG_X_SIZE * self.EGG_Y_SIZE
            x = np.linspace(start=0, stop=a, num=self.m)
            y = b * np.power(1.0 - np.power(x / a, self.exponent), 1.0 / self.exponent)
            x0, xm = np.array([x[0]]), np.array([x[-1]])
            x1m = x[1:-1]
            xm1 = x1m[::-1]
            y0, ym = np.array([y[0]]), np.array([y[-1]])
            y1m = y[1:-1]
            ym1 = y1m[::-1]
            xx = np.concatenate((-xm, -xm1, x0, x1m, xm, xm1, x0, -x1m, -xm))
            yy = np.concatenate((ym, ym1, y0, y1m, ym, -ym1, -y0, -y1m, -ym))
            a_xx.append(xx)
            a_yy.append(yy)

        self.x = np.array(a_xx)
        self.y = np.array(a_yy)
        self.z = np.array(a_zz)

        # scale to real egg-size
        self.x = self.EGG_X_SIZE * self.x / (np.max(self.x) - np.min(self.x))
        self.y = self.EGG_Y_SIZE * self.y / (np.max(self.y) - np.min(self.y))
        self.z = self.EGG_Z_SIZE * self.z / (np.max(self.z) - np.min(self.z))

        return self.x, self.y, self.z

    def build_mesh(self):
        k = 0
        vertices = []
        faces = []
        max_i = self.z.shape[0]
        for i in tqdm(range(max_i), desc="Building Mesh"):
            # 1. add the vertices
            mi = self.x[i, :].shape[0]
            for j in range(mi):
                vertices.append((self.z[i], self.x[i, j], self.y[i, j]))
                k += 1

            # 2. build the faces with previously computed vertices
            if 0 < i < max_i - 1:
                for j in range(mi):
                    i1_j0 = k - mi + j
                    i1_j1 = k - mi + (j + 1) % mi
                    i0_j0 = k - mi * 2 + j
                    i0_j1 = k - mi * 2 + (j + 1) % mi

                    faces.append((i1_j0, i1_j1, i0_j0))
                    faces.append((i0_j0, i0_j1, i1_j1))
            elif (i == 0) or (i == max_i - 1):
                for j in range(0, mi, 2):
                    i1_j0 = k - mi + j
                    i1_j1 = k - mi + (j + 1) % mi
                    i1_j2 = k - mi + (j + 2) % mi
                    faces.append((i1_j0, i1_j1, i1_j2))

        self.vertices = np.array(vertices)
        self.faces = np.array(faces)

        self.egg = mesh.Mesh(np.zeros(self.faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                self.egg.vectors[i][j] = self.vertices[f[j], :]
        # Write the mesh to file "cube.stl"
        self.egg.save(filename=self.output_path)


if __name__ == "__main__":
    egg = ComputeEgg()
    egg.run()
