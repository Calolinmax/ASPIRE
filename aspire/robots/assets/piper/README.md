# aspire/robots/assets/piper/ —— Piper 模型资产 🔒

| 文件 | 说明 |
|---|---|
| `robot.xml` | 臂体 MJCF：joint1-6（j5 腕限位 ±70°）、robotview 相机 pos/quat 🔏锁死（同步项 engine_capx.ROBOTVIEW_CAM_WORLD）、eye_in_hand 腕部相机、6 个 position 执行器（j1-j3 kp80/kv5、j4 kp40、j5-j6 kp10）、碰撞 geom 自动命名 `robot0_g{i}_col`（全管线碰撞判对命名面来源）。 |
| `gripper.xml` | 平行夹爪 MJCF：eef body（绕 y 180° 翻转**不可去**）承载 grip_site（TCP）；执行器 kp=120（08-04 用户批准增强）。 |
| `wiper_gripper.xml` | **现役擦头**（wipe 用）：Franka 12×5×3cm 海绵板，38° 补偿安装 + solref 软化（0.05→0.2）；site→板面心 Δ=0.090（=任务代码 WIPER_FACE_L，改模型须同步重测）。 |
| `wiper_gripper_ball.xml` | 存档：首版球形擦头（r=12mm，L=0.074），回退备用。 |
| `pedestal.xml` | 25cm 圆柱立柱台架（现作桌面增高台）。 |
| `default_piper.json` | composite 控制器配置（JOINT_POSITION kp200/kd40/kv20 + 夹爪 GRIP）。⚠️ 与 robot.xml 执行器层 kp/kv 是两套参数，勿混。 |
| `ik_library.npz`（~11MB） | **IK 种子库**：Q/P/Z/Y 四键各 201,182 条 float32（2026-08-03 重建，rng=42）。几何/安装变更后必须用 `scripts/tools/build_piper_ik_library.py` 重建。 |
| 网格 84 个（72 OBJ + 12 STL） | **双轨制**：OBJ=纯视觉件（group=1）；STL=碰撞体为主（group=0，robot0_g*_col 网格来源）。link2_gray/link2_red/linke2_dark_gray 为官方导出残留未引用件。 |

🔒 2026-08-07 用户指令加锁（含本目录全部文件）。
