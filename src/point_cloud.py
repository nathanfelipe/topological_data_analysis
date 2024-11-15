# point_cloud.py
import numpy as np


class PointCloud:
    def __init__(self, points=None):
        self.points = points

    @staticmethod
    def generate_circle(n_points):
        t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        x = np.cos(t)
        y = np.sin(t)
        return PointCloud(np.column_stack((x, y)))

    @staticmethod
    def generate_sphere(n_points):
        X = np.random.randn(n_points, 3)
        X /= np.linalg.norm(X, axis=1)[:, np.newaxis]
        return PointCloud(X)

    @staticmethod
    def generate_torus(n_points, R, r):
        theta = np.linspace(0, 2 * np.pi, n_points)
        phi = theta * 2
        x = (R + r * np.cos(theta)) * np.cos(phi)
        y = (R + r * np.cos(theta)) * np.sin(phi)
        z = r * np.sin(theta)
        return PointCloud(np.column_stack((x, y, z)))

    def add_noise(self, noise_level):
        noise = noise_level * np.random.randn(*self.points.shape)
        self.points += noise
