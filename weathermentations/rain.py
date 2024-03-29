from .raindrop.dropgenerator import generateDrops, generate_label
from .raindrop.config import cfg

from PIL import Image
import numpy as np
from .raindrop.raindrop import Raindrop


from random import randint

from albumentations.core.transforms_interface import ImageOnlyTransform


class DropsOnLens(ImageOnlyTransform):
    """Simulates Water Drops on lenses
    Based on https://github.com/ricky40403/ROLE

    Args:
        maxR (float): Maximum drop radius.
        minR (float): Maximum drop radius.
        maxDrops (float): Maximum num of drops.
        minDrops (float): Minimum num of drops.
        edge_darkratio (float): Edge Dark Ratio (see original repo and doc).
        label_thres (float): Label threshold (see original repo and doc).

    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(self, always_apply=False, p=1.0, **cgf_override):
        super().__init__(always_apply, p)

        self.cfg_override = cgf_override

    def apply(self, img, **params):
        (h, w, _) = img.shape
        cfg['maxR'] = self.cfg_override.get('maxR', 54)
        cfg['minR'] = self.cfg_override.get('minR', 32)

        cfg['maxDrops'] = self.cfg_override.get('maxDrops', 12)
        cfg['minDrops'] = self.cfg_override.get('minDrops', 2)

        cfg['edge_darkratio'] = self.cfg_override.get('edge_darkratio', 0.3)
        cfg['label_thres'] = self.cfg_override.get('label_thres', 128)

        drops_list, label_map = generate_label(h, w, cfg)
        pil_img = Image.fromarray(img)
        output_image, output_label, mask = generateDrops(
            pil_img, cfg, drops_list, label_map)
        res = np.array(output_image)
        return res
