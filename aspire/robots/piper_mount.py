"""Piper 落地立柱台架 (MountModel)——见 pedestal.xml 注释。"""

from __future__ import annotations

import os

import numpy as np
from robosuite.models.bases.mount_model import MountModel

_ASSETS = os.path.join(os.path.dirname(__file__), "assets", "piper")


class PiperPedestal(MountModel):
    """25cm 落地立柱 (Arm base z=0.25, 桌面相对高度 0.55)。"""

    def __init__(self, idn=0):
        super().__init__(os.path.join(_ASSETS, "pedestal.xml"), idn=idn)

    @property
    def top_offset(self):
        return np.array((0, 0, 0))

    @property
    def horizontal_radius(self):
        return 0.12
