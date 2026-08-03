"""预计算 Piper IK 种子库（离线 FK 稠密采样）。

背景（本仓库实测）: Piper 短臂 + 腕限位 (j5±70°) 使"俯身抓桌面"的可达构型
流形极薄（随机种子的 DLS 重启基本打不中），但一旦落在盆地内，DLS 在
±0.8 rad 噪声下 4/4 收敛。因此 solve_ik 采用"库检索近邻种子 + DLS 精修"。

库内容: 原始 piper.xml 上均匀采样关节, 仅保留
  - 开口方向 (-link6_z) 距竖直向下 < 55°, 且
  - 指尖(site)位置在工作盒 x∈[0.15,0.60], |y|<0.25, z∈[0.35,0.90] (基座系,
    含 25cm 立柱安装后的桌面/悬停/放置高度带)
另按 1/25 比例保留少量全域构型做兜底。

产物: aspire/robots/assets/piper/ik_library.npz  (Q,P,Z,Y: float32)

用法:
    /home/stouching/anaconda3/envs/ASPIRE/bin/python scripts/build_piper_ik_library.py [--n 600000]
"""

import argparse
import os
import time

import mujoco
import numpy as np

RAW_XML = "/home/stouching/Desktop/ASPIRE/external/agilex_arm_mujoco/agilex_arm/agilex_piper/piper.xml"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "aspire", "robots", "assets", "piper", "ik_library.npz")

# site (grip_site) 相对 link6: eef 在 piper_hand(root z=-0.045, R_y(π) 翻转)
# → site pos = link6 + 0.09 * z_root; site z (开口) = -z_root; site y = -y_root
RY_PI = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])

BOX = ((0.15, 0.45), (-0.25, 0.25), (-0.05, 0.45))
# BOX 修正 (2026-08-03): 旧值 z∈[0.35,0.90] 是 25cm 立柱+低桌(0.55) 时代的
# 高度带; 现安装 (基座 z=0.80=桌面) 下抓取位姿基座系 z≈0.02、预抓取 ≈0.12、
# 悬停 ≤0.45 —— 旧盒对 descend 高度【零覆盖】, 是 DLS 兜底失败的结构性原因。
# x 上界 0.60→0.45 对齐 CUBE 工作区注释 (可达带 x∈[0.28,0.40]+余量)。
OPENING_MAX_DEG = 55.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600_000)
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(RAW_XML)
    d = mujoco.MjData(m)
    l6 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link6")
    rng = np.random.default_rng(42)
    q_lo, q_hi = m.jnt_range[:6, 0], m.jnt_range[:6, 1]

    cos_max = np.cos(np.deg2rad(OPENING_MAX_DEG))
    Q, P, Z, Y = [], [], [], []
    t0 = time.time()
    kept = 0
    for i in range(args.n):
        q = rng.uniform(q_lo, q_hi)
        d.qpos[:6] = q
        mujoco.mj_forward(m, d)
        R6 = d.xmat[l6].reshape(3, 3)
        z_root = R6[:, 2]
        p = d.xpos[l6] + 0.09 * z_root
        opening = -z_root
        in_box = (BOX[0][0] < p[0] < BOX[0][1] and BOX[1][0] < p[1] < BOX[1][1]
                  and BOX[2][0] < p[2] < BOX[2][1])
        steep = opening[2] < -cos_max
        if steep and in_box:
            Q.append(q)
            P.append(p)
            Z.append(opening)
            Y.append(-R6[:, 1])
            kept += 1
        elif i % 25 == 0:
            Q.append(q)
            P.append(p)
            Z.append(opening)
            Y.append(-R6[:, 1])
        if (i + 1) % 100_000 == 0:
            print(f"  {i + 1}/{args.n} sampled, kept(steep+box)={kept}, "
                  f"{time.time() - t0:.0f}s", flush=True)

    np.savez_compressed(
        OUT,
        Q=np.asarray(Q, dtype=np.float32),
        P=np.asarray(P, dtype=np.float32),
        Z=np.asarray(Z, dtype=np.float32),
        Y=np.asarray(Y, dtype=np.float32),
    )
    print(f"库规模 {len(Q)} (俯身工作区 {kept}), 保存到 {OUT} "
          f"({os.path.getsize(OUT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
