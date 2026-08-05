#!/usr/bin/env python3
"""从 Piper MJCF 生成配套简化 URDF，供 pyroki/cuRobo 做 IK。

策略：
1. 用 MuJoCo 解析 MJCF（含 robosuite 前缀后的完整模型）。
2. 取 robot0_base_link 为根，沿 body 树提取 link/joint。
3. 每个 URDF joint origin = child body 在 parent body 系中的 pos/quat（MuJoCo body frame）。
4. joint axis = MuJoCo joint axis（body 局部系）。
5. 无 mesh、无碰撞几何——只保留 frame + 限位，足够 IK/FK。
6. 输出到 external/piper_description/piper_mjcf/urdf/piper_mjcf.urdf。
"""
from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import (
    load_composite_controller_config,
)

import aspire.robots  # noqa
from aspire.robots.piper_robot import _ASSETS


def quat_to_rpy(quat_wxyz: np.ndarray) -> tuple[float, float, float]:
    """wxyz 四元数 → URDF 的 rpy (roll, pitch, yaw)，使用标准欧拉角 ZYX。"""
    w, x, y, z = quat_wxyz
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return float(roll), float(pitch), float(yaw)


def build_urdf():
    env = suite.make(
        env_name="Stack",
        robots="Piper",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        controller_configs=load_composite_controller_config(
            controller=os.path.join(_ASSETS, "default_piper.json")
        ),
    )
    model = env.sim.model._model

    # 找到 root body（base_link，带 robosuite 前缀 robot0_base_link）
    root_name = "robot0_base_link"
    root_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)
    if root_bid < 0:
        raise RuntimeError(f"root body {root_name} not found")

    # body 名去掉前缀
    prefix = "robot0_"

    def clean(name: str) -> str:
        return name[len(prefix):] if name.startswith(prefix) else name

    # 构建父子关系：遍历所有 body，找到 parent
    nbody = model.nbody
    body_children: dict[int, list[int]] = {i: [] for i in range(nbody)}
    for bid in range(nbody):
        pid = int(model.body_parentid[bid])
        if pid != bid and pid >= 0:
            body_children[pid].append(bid)

    # 收集 joint 信息
    joint_info = {}
    for jid in range(model.njnt):
        bid = int(model.jnt_bodyid[jid])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        axis = np.asarray(model.jnt_axis[jid], dtype=np.float64)
        pos = np.asarray(model.jnt_pos[jid], dtype=np.float64)
        lo, hi = model.jnt_range[jid]
        joint_info[bid] = {
            "name": clean(name),
            "axis": axis,
            "pos": pos,
            "range": (float(lo), float(hi)),
            "type": "revolute",
        }

    # 递归生成 URDF XML
    robot = ET.Element("robot", {"name": "piper_mjcf"})

    def add_link(bid: int, parent_bid: int | None):
        name = clean(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid))
        # link 元素
        ET.SubElement(robot, "link", {"name": name})

        # 如果有 joint 连接 parent
        if parent_bid is not None:
            parent_name = clean(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_bid))
            jnt = joint_info.get(bid)
            jnt_elem = ET.SubElement(
                robot, "joint",
                {
                    "name": jnt["name"] if jnt is not None else f"{parent_name}_to_{name}",
                    "type": jnt["type"] if jnt is not None else "fixed",
                }
            )
            ET.SubElement(jnt_elem, "parent", {"link": parent_name})
            ET.SubElement(jnt_elem, "child", {"link": name})
            if jnt is not None:
                # origin: child body 相对于 parent body 的 transform
                # MuJoCo model.body_pos/quat 已经是相对 parent 的
                pos = np.asarray(model.body_pos[bid], dtype=np.float64)
                quat_wxyz = np.asarray(model.body_quat[bid], dtype=np.float64)
                rpy = quat_to_rpy(quat_wxyz)
                ET.SubElement(jnt_elem, "origin", {
                    "xyz": f"{pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}",
                    "rpy": f"{rpy[0]:.8f} {rpy[1]:.8f} {rpy[2]:.8f}",
                })
                # axis 在 joint local frame 中；joint 位于 child body 初始 frame 原点
                axis = jnt["axis"]
                ET.SubElement(jnt_elem, "axis", {"xyz": f"{axis[0]:.8f} {axis[1]:.8f} {axis[2]:.8f}"})
                lo, hi = jnt["range"]
                ET.SubElement(jnt_elem, "limit", {"lower": f"{lo:.8f}", "upper": f"{hi:.8f}",
                                                   "effort": "100", "velocity": "10"})

            # 在 link6 上挂载 TCP frame（与 grip_site 同位姿）
            if name == "link6":
                ET.SubElement(robot, "link", {"name": "tcp"})
                tcp_joint = ET.SubElement(robot, "joint", {"name": "link6_to_tcp", "type": "fixed"})
                ET.SubElement(tcp_joint, "parent", {"link": "link6"})
                ET.SubElement(tcp_joint, "child", {"link": "tcp"})
                tcp_rpy = quat_to_rpy(np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64))  # wxyz, 绕 y 180°
                ET.SubElement(tcp_joint, "origin", {
                    "xyz": "0.00000000 0.00000000 0.09003000",
                    "rpy": f"{tcp_rpy[0]:.8f} {tcp_rpy[1]:.8f} {tcp_rpy[2]:.8f}",
                })

        for cid in sorted(body_children[bid]):
            add_link(cid, bid)

    add_link(root_bid, None)

    # 格式化输出
    ET.indent(robot, space="  ")
    urdf_str = ET.tostring(robot, encoding="unicode")
    urdf_str = '<?xml version="1.0"?>\n' + urdf_str

    out_dir = Path("external/piper_description/piper_mjcf/urdf")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "piper_mjcf.urdf"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(urdf_str)
    print(f"Generated URDF: {out_path}")
    print(f"  bodies: {nbody}, joints: {model.njnt}")

    env.close()


if __name__ == "__main__":
    build_urdf()
