"""Pyroki IK client（主环境调用，HTTP 协议与 cap-x server 兼容）。

对应 cap-x `integrations/motion/pyroki.py:15-52`，只保留 IK client，
去掉了 trajectory planning（D4 简化清单已砍掉）。
"""
from __future__ import annotations

import numpy as np
import urllib.request
import urllib.error

_DEFAULT_URL = "http://127.0.0.1:8116"


def _post(path: str, payload: dict, timeout: float = 15.0) -> dict:
    import json
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{_DEFAULT_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        raise RuntimeError(f"pyroki server {path} error {e.code}: {body}")


def ik_pyroki(pos_world: np.ndarray, quat_xyzw: np.ndarray, prev_cfg: np.ndarray | None = None) -> np.ndarray:
    """调用 pyroki server 求解 IK。

    Args:
        pos_world: (3,) 目标位置（世界/基座系，米）
        quat_xyzw: (4,) 目标姿态 [x,y,z,w]（世界/基座系）
        prev_cfg: (6,) 可选的前一构型（velocity-cost 热启动）

    Returns:
        (6,) 臂关节角 (rad)。server 失败抛 RuntimeError。
    """
    wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)
    pose = np.concatenate([wxyz, np.asarray(pos_world, dtype=np.float64).reshape(3)])
    payload = {"target_pose_wxyz_xyz": pose.tolist()}
    if prev_cfg is not None:
        # pyroki server 的 URDF 含 gripper 共 8 个 actuated joints，
        # prev_cfg 只给 arm 6 轴时补 0.0 使其长度匹配。
        prev = np.asarray(prev_cfg, dtype=np.float64).reshape(-1).tolist()
        if len(prev) == 6:
            prev = prev + [0.0, 0.0]
        payload["prev_cfg"] = prev
    out = _post("/ik", payload)
    return np.asarray(out["joint_positions"], dtype=np.float64)
