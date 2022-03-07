import numpy as np

import numpy as np

from albumentations.core.transforms_interface import ImageOnlyTransform


class Backlit(ImageOnlyTransform):
    """Simulates sun backlit imges

    Args:
        k_range (float): range for the k coefficient to boost yellow intensity.
        c_0_range (float): Maximum contrast reduction range (top).
        c_end_range (float): Minimum contrast reduction range (bottom).
        b_0_range (float): Maximum brightness reduction range (top).
        b_end_range (float): Minimum brightness reduction range (bottom).
    Targets:
        image
    Image types:
        uint8, float32
    """
    def __init__(self, k_range=(1.3, 1.7), c_0_range=(0.1, 0.3), c_end_range=(0.4, 0.7), b_0_range=(190, 230), b_end_range=(30, 70), always_apply=False, p=1.0):
        super(Backlit, self).__init__(always_apply, p)

        self.k_range = k_range
        self.c_0_range = c_0_range
        self.c_end_range = c_end_range
        self.b_0_range = b_0_range
        self.b_end_range = b_end_range

    def apply(self, img, **params):

        # coefficient to increase the yellow in the image

        k = np.random.uniform(self.k_range[0], self.k_range[1])

        # coefficinet to modify brightness and contrast

        c_0 = np.random.uniform(self.c_0_range[0], self.c_0_range[1])

        c_end = np.random.uniform(self.c_end_range[0], self.c_end_range[1])

        b_0 = np.random.uniform(self.b_0_range[0], self.b_0_range[1])

        b_end = np.random.uniform(self.b_end_range[0], self.b_end_range[1])

        (h, w, _) = img.shape

        R, G, B = img[:, :, 0].copy(), img[:, :, 1].copy(), img[:, :, 2].copy()
        R = np.clip(R * k, 0, 255, R)
        G = np.clip(G * k, 0, 255, G)

        for r in range(h):
            b = (b_0 - b_end)/(h-1) * (0-r) + b_0
            c = (c_0 - c_end)/(h-1) * (0-r) + c_0
            R[r, :] = np.clip(R[r, :]*c+b, 0, 255, R[r, :])
            G[r, :] = np.clip(G[r, :]*c+b, 0, 255, G[r, :])
            B[r, :] = np.clip(B[r, :]*c+b, 0, 255, B[r, :])

        res = np.dstack((R, G, B))#.astype(np.uint8)
        return res
