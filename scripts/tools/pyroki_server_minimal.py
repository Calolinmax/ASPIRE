#!/usr/bin/env python3
"""最小化 pyroki IK 服务（Piper URDF）。

协议与 cap-x `launch_pyroki_server.py:317 /ik` 兼容，
独立 venv 启动，零 cap-x 依赖。

启动:
    PYTHONPATH=/home/stouching/Desktop/ASPIRE/external/cap-x \
    ~/venvs/pyroki/bin/python scripts/tools/pyroki_server_minimal.py \
    --urdf external/piper_description/piper/urdf/piper_description.urdf \
    --target-link link6 --port 8116
"""
from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import pyroki as pk

# ------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------
app = FastAPI()
_ROBOT: pk.Robot | None = None
_TARGET_LINK: str | None = None
_TARGET_LINK_INDEX: int | None = None


class IkRequest(BaseModel):
    target_pose_wxyz_xyz: list[float]  # length 7 (wxyz + xyz)
    prev_cfg: list[float] | None = None


class IkResponse(BaseModel):
    joint_positions: list[float]


# ------------------------------------------------------------------
# IK solve (sync, CPU-bound)
# ------------------------------------------------------------------
def _solve_ik(target_pose_wxyz_xyz: np.ndarray, prev_cfg: np.ndarray | None) -> list[float]:
    assert _ROBOT is not None and _TARGET_LINK_INDEX is not None
    target_wxyz = np.array(target_pose_wxyz_xyz[:4], dtype=np.float64)
    target_position = np.array(target_pose_wxyz_xyz[4:], dtype=np.float64)

    import jax
    import jax.numpy as jnp
    import jax_dataclasses as jdc
    import jaxlie
    import jaxls

    @jdc.jit
    def _ik_jax(robot: pk.Robot, link_idx, wxyz, pos, prev):
        joint_var = robot.joint_var_cls(0)
        factors = [
            pk.costs.pose_cost_analytic_jac(
                robot, joint_var,
                jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(wxyz), pos),
                link_idx, pos_weight=50.0, ori_weight=10.0,
            ),
            pk.costs.limit_cost(robot, joint_var, weight=100.0),
        ]
        if prev is not None:
            factors.append(
                # 【2026-08-03 修复】weight 10.0 → 0.01: 原值把解拉离目标
                # 140-284mm（rest 项主导盆地形貌, 零位热启动从不收敛,
                # solve_ik 的 FK 后验全拒 → 实际一直在用 DLS 兜底）。
                # 实测(entry100/50000/150000 精确可达位姿): 10.0→284mm,
                # 0.1→0.1mm, 0.01→0.0mm（零位启动）。保留微小值仅作正则。
                pk.costs.rest_cost(joint_var, rest_pose=prev, weight=0.01)
            )
        sol = (
            jaxls.LeastSquaresProblem(factors, [joint_var])
            .analyze()
            .solve(verbose=False, linear_solver="dense_cholesky",
                   trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0))
        )
        return sol[joint_var]

    prev_j = jnp.array(prev_cfg, dtype=jnp.float64) if prev_cfg is not None else None
    q = _ik_jax(_ROBOT, jnp.array(_TARGET_LINK_INDEX), jnp.array(target_wxyz), jnp.array(target_position), prev_j)
    return list(map(float, np.array(q)))


@app.post("/ik", response_model=IkResponse)
async def solve_ik(req: IkRequest):
    if _ROBOT is None:
        raise HTTPException(503, "Pyroki not initialized")
    target_pose = np.array(req.target_pose_wxyz_xyz, dtype=np.float64)
    prev_cfg = np.array(req.prev_cfg, dtype=np.float64) if req.prev_cfg is not None else None
    try:
        joints = await _run_in_thread(_solve_ik, target_pose, prev_cfg)
    except Exception as e:
        raise HTTPException(500, f"IK solve failed: {e}")
    return IkResponse(joint_positions=joints)


async def _run_in_thread(fn, *args, **kwargs):
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))


class FkRequest(BaseModel):
    joint_positions: list[float]


class FkResponse(BaseModel):
    position: list[float]
    quat_wxyz: list[float]


# ------------------------------------------------------------------
# FK (for TCP calibration)
# ------------------------------------------------------------------
@app.post("/fk", response_model=FkResponse)
async def forward_kinematics(req: FkRequest):
    if _ROBOT is None:
        raise HTTPException(503, "Pyroki not initialized")
    try:
        import jax.numpy as jnp
        q = jnp.array(req.joint_positions, dtype=jnp.float64)
        T = _ROBOT.forward_kinematics(q)  # shape (link_count, 7) = wxyz_xyz
        pose = np.array(T[_TARGET_LINK_INDEX])
        return FkResponse(position=pose[4:].tolist(), quat_wxyz=pose[:4].tolist())
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"FK failed: {e}")
# ------------------------------------------------------------------
def init_server(urdf_path: str, target_link: str):
    global _ROBOT, _TARGET_LINK, _TARGET_LINK_INDEX
    print(f"[pyroki_server] Loading URDF: {urdf_path}")
    import yourdfpy
    urdf = yourdfpy.URDF.load(urdf_path)
    _ROBOT = pk.Robot.from_urdf(urdf)
    _TARGET_LINK = target_link
    _TARGET_LINK_INDEX = _ROBOT.links.names.index(target_link)
    print(f"[pyroki_server] Ready! target_link={target_link} index={_TARGET_LINK_INDEX} "
          f"joints={_ROBOT.joints.num_actuated_joints}")
    # Warm-up: trigger JIT compilation on main thread before serving
    print("[pyroki_server] JIT warm-up ...")
    _solve_ik(np.array([1.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.5], dtype=np.float64), None)
    print("[pyroki_server] Warm-up done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--target-link", default="link6")
    parser.add_argument("--port", type=int, default=8116)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    init_server(args.urdf, args.target_link)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
