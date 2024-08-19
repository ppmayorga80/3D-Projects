import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class ComputeEgg:
    PATH = "path5.jpg"
    POW = 3.14
    POINTS = 16

    def __init__(self, path=PATH, n: float = POW, m: int = POINTS):
        self.path = path
        self.n = n
        self.m = m

        self.binary_image: np.ndarray = np.zeros((1, 1))
        self.z_a: np.ndarray = np.zeros((1, 1))
        self.z: np.ndarray = np.zeros((1, 1))
        self.xy: np.ndarray = np.zeros((1, 1))

    def run(self):
        self.binary_image = self.read_image()
        self.z_a = self.image_to_z_a()
        self.compute_xyz()

    def read_image(self) -> np.ndarray:
        image = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        _, bw_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
        return bw_image

    def image_to_z_a(self) -> np.ndarray:
        rows, columns = self.binary_image.shape

        za = [(rows, 0)]
        for r in range(rows):
            z_r = rows - r - 1
            a_r = max([c for b_r, c in zip(self.binary_image[r, :], range(columns)) if b_r == 0])

            za.append((z_r, a_r))

        za = np.array(za)
        return za

    def compute_xyz(self):

        self.z, self.xy = [], []
        for z, a in tqdm(self.z_a, desc="Computing x,y,z"):
            self.z.append(z)
            if a == 0:
                total_size = 4 * self.m - 3
                xx, yy = np.zeros((total_size,)), np.zeros((total_size,))
                self.xy.append((xx, yy))
            else:
                b = a / 24.1 * 21.5
                x = np.linspace(start=0, stop=a, num=self.m)
                y = b * np.power(1.0 - np.power(x / a, self.n), 1.0 / self.n)

                x0, xm = np.array([x[0]]), np.array([x[-1]])
                x1m = x[1:-1]
                xm1 = x1m[::-1]

                y0, ym = np.array([y[0]]), np.array([y[-1]])
                y1m = y[1:-1]
                ym1 = y1m[::-1]

                xx = np.concatenate((-xm, -xm1, x0, x1m, xm, xm1, x0, -x1m, -xm))
                yy = np.concatenate((ym, ym1, y0, y1m, ym, -ym1, -y0, -y1m, -ym))

                self.xy.append((xx, yy))

        return self.z, self.xy


if __name__ == "__main__":
    egg = ComputeEgg()
    egg.run()
