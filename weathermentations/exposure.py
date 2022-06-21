import cv2
import skimage.draw as draw
from scipy.ndimage.morphology import distance_transform_edt as dt
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

        res = np.dstack((R, G, B))  # .astype(np.uint8)
        return res


def add_sharpness(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


def blur(image):
    return cv2.blur(image, (5, 5))


def sharp_blur(image):
    return blur(image) if np.random.random() > 0.5 else add_sharpness(image)


def saturation(image):
    hsvImg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsvImg[..., 1] = hsvImg[..., 1] * np.random.random()*2
    return cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)


def contrast(img):
    factor = max(0.2, np.random.random()*2.)
    return np.clip(128 + factor * img - factor * 128, 0, 255).astype(np.uint8)


def get_mask(img, minmax=(120, 160)):
    mask = np.zeros_like(img[..., 0])
    x, y = mask.shape
    min_dim = min(x, y)
    if np.random.random() > 0.5:  # Circle-shaped masks
        random_r = np.random.randint(int(min_dim / 5), int(min_dim / 2))
        random_r = int(random_r / 2)
        random_x = np.random.randint(random_r, x - random_r)
        random_y = np.random.randint(random_r, y - random_r)
        rr, cc = draw.circle(random_x, random_y, random_r)
    else:  # Ellipse-shaped masks
        random_r = np.random.randint(int(min_dim / 5), int(min_dim / 1.5))
        random_r = int(random_r / 2)
        random_x = np.random.randint(random_r, x - random_r)
        random_y = np.random.randint(random_r, y - random_r)
        rr, cc = draw.ellipse(random_x, random_y, random_r, random_r*np.random.uniform(low=0.3, high=0.8, size=1)[0],
                              shape=(x, y), rotation=np.random.random()*np.pi*2-np.pi)
    mask[rr, cc] = 1
    mask = dt(mask)
    rv = np.random.randint(minmax[0], minmax[1])
    mask = mask / np.max(mask) * rv
    return mask, rv


def illumination_augmenter(img, global_mask=(40, 80), local_mask=(120, 160)):
    img = np.squeeze(img)

    if np.random.random() < 0.33:
        img = saturation(img)
    elif np.random.random() < 0.66:
        img = sharp_blur(img)
    else:
        img = contrast(img)
        local_mask = (80, 120)
        global_mask = (30, 60)

    # Only local changes
    if any(x > 0 for x in local_mask):
        mask, ch = get_mask(img, local_mask)
        mask = np.stack((mask,) * 3, axis=-1)
        sign = '-'
        if np.random.random() > 0.5:
            sign = '+'
            img = img + mask
        else:
            img = img - mask

        # Local and global changes
        if any(x > 0 for x in global_mask):
            if np.random.random() > 0.5:
                sign += '+'
            else:
                sign += '-'
            if sign == '--' or sign == '++':
                global_max = global_mask[1]
                global_min = global_mask[0]
            else:
                global_max = global_mask[1] + ch
                global_min = global_mask[0] + ch

            if sign[1] == '+':
                img = img + np.ones_like(img) * \
                    np.random.randint(global_min, global_max)
            elif sign[1] == '-':
                img = img + \
                    np.ones_like(img) * \
                    np.random.randint(global_min, global_max) * -1

    # Only global changes
    elif any(x > 0 for x in global_mask):
        global_min, global_max = global_mask
        sign = [-1, 1]
        img = img + np.ones_like(img) * np.random.randint(global_min,
                                                          global_max) * sign[np.random.randint(0, 2)]
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype(np.uint8)
    return img


class DuskDawn(ImageOnlyTransform):
    """ Illumination augmentation based on "Illumination-Based Data Augmentation for Robust Background Subtraction"
        https://github.com/dksakkos/illumination_augmentation

    """
    def __init__(self, always_apply=False, p=1.0, **cgf_override):
        super().__init__(always_apply, p)
        self.augmenting_prob = 1.0
        self.local_mask = (1, 80)
        self.global_mask = (1, 40)
        self.augment_illumination = any(x > 0 for x in list(
            self.local_mask) + list(self.global_mask))

    def apply(self, img, **params):
        if self.augment_illumination and np.random.random() < self.augmenting_prob:
            img = illumination_augmenter(
                img, self.global_mask, self.local_mask)
        return img
