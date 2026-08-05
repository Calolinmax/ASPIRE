"""ASPIRE 执行引擎（Execution Engine）。

职责（AGENTS.md 第 3 节）：执行 coding agent 生成的任务代码，并记录每一次
primitive 调用的【观测、输入、输出、视觉证据】。

每次执行生成独立 trace 目录（MMDD_HHMM_任务名）：trace.json + images/
（top/wrist 帧流 + depth 边界帧 + 各视觉算法标注文件夹）。任务代码由
agent harness 管理，trace.json 的 code_ref 仅记录其路径引用。

用法：
    engine = ExecutionEngine(task="Lift", seed=0)
    result = engine.run(code_string)

CLI（子进程隔离执行，供 harness 调用）：
    MUJOCO_GL=egl python -m aspire.engine --code task_code.py --task Lift --seed 0
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import time
import traceback

import cv2
import numpy as np
import robosuite as suite
from robosuite.utils import camera_utils as CU

from .primitives import build_namespace
from .trace import Tracer

ENV_DEFAULTS = dict(
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names=["agentview", "robot0_eye_in_hand"],  # 双相机：顶部 + 腕部
    camera_heights=256,
    camera_widths=256,
    camera_depths=True,
    control_freq=20,
    horizon=4000,  # 容纳伺服闭环+修复重放（move_to_pose 单次可达 160 步，超出后 step 静默忽略）
)

# 相机名 → trace 帧流别名（top 即 agentview 顶部相机，wrist 即 robot0_eye_in_hand）
CAMERA_ALIASES = {"agentview": "top", "robot0_eye_in_hand": "wrist"}

FRAME_INTERVAL = 5          # 帧流固定间隔（仿真步 @20Hz → 4fps）
DEPTH_CLIP = (0.5, 2.5)     # depth colormap 固定量程（米），帧间可比


class ExecutionEngine:
    # 相机名 → trace 帧流别名（子类可覆盖）
    CAMERA_ALIASES = CAMERA_ALIASES

    def __init__(self, task: str = "Lift", seed: int = 0, trace_root: str = "traces",
                 render: bool = False, render_slowdown: float = 1.0):
        self.task = task
        self.seed = seed
        self.render = render
        self.render_slowdown = render_slowdown  # 演示减速倍率（>1 更慢）
        np.random.seed(seed)
        kwargs = self._env_kwargs()
        if render:
            kwargs["has_renderer"] = True
            kwargs["has_offscreen_renderer"] = True  # 视觉证据仍走离屏
        self.env = self._make_env(kwargs)
        self.obs = self.env.reset()
        # (别名, 真实相机名) 列表，供 trace 双相机存图
        self.cameras = [(self.CAMERA_ALIASES.get(c, c), c) for c in kwargs["camera_names"]]
        self.sim_step = 0
        self.gripper_cmd = -1.0  # 初始张开
        self.terminated = False
        # trace 目录：MMDD_HHMM_任务名（同时刻冲突加 _2/_3 后缀）
        base = f"{time.strftime('%m%d_%H%M')}_{task}"
        name, i = base, 2
        while os.path.exists(os.path.join(trace_root, name)):
            name = f"{base}_{i}"
            i += 1
        self.run_name = name
        self.trace_dir = os.path.join(trace_root, name)
        self.tracer = Tracer(self.trace_dir, task=task, seed=seed, frame_interval=FRAME_INTERVAL)
        self.renderer_recoveries = 0
        self._post_reset()
        self._renderer_healthcheck()
        self.capture_boundary(0)  # 初始帧：帧流起点

    # ------------------------------------------------------------------
    # 子类钩子：环境构造与 reset 后处理（cap-x/Piper 变体覆盖）
    # ------------------------------------------------------------------
    def _env_kwargs(self) -> dict:
        return dict(ENV_DEFAULTS)

    def _make_env(self, kwargs: dict):
        return suite.make(self.task, **kwargs)

    def _post_reset(self):
        """reset 完成后的子类钩子（默认空）。"""

    def _build_namespace(self) -> dict:
        """注入任务代码的命名空间（子类可换 API 集）。"""
        return build_namespace(self)

    @staticmethod
    def _img_corrupt(img: np.ndarray) -> bool:
        """检测损坏的渲染帧（EGL offscreen 偶发 wedge 后的条纹/黑帧）。

        正常 robosuite 渲染是平滑图像：水平相邻像素差的 p95 ≤ ~10（uint8 标度）。
        损坏帧呈竖条纹/大面积死黑：p95 ≥ ~60 或近零像素 > 30%。两者都有充足间隔。
        """
        if img is None or img.ndim != 3 or img.shape[1] < 8:
            return True
        dx = np.abs(np.diff(img.astype(np.float32), axis=1)).mean(axis=-1)
        if float(np.percentile(dx, 95)) > 25.0:
            return True
        return bool((img.sum(axis=-1) < 10).mean() > 0.30)

    def recover_renderer(self):
        """重建渲染管线并刷新观测（恢复 wedged 渲染器），逐级升级：

        L1 轻量：update_offscreen_size 抖动，迫使 MjrContext free+重建（buffer 对齐）。
        L2 重量：整个 MjRenderContextOffscreen 重建 + 尺寸对齐。
           注意直接重建会在 init 中途 del 旧 context、其析构解除 GL current 绑定，
           导致新 MjrContext 创建失败（"Default framebuffer is not complete"）；
           且新 context 的 EGL surface 与 buffer 尺寸错位会读出噪声帧，
           故需拦截 add_render_context + 事后恢复 current + 尺寸对齐。
        两级后仍损坏则抛错（不再让垃圾帧流入视觉）。
        """
        sim = self.env.sim
        ch = getattr(self.env, "camera_heights", 256)
        cw = getattr(self.env, "camera_widths", 256)
        cam_h = ch[0] if isinstance(ch, (list, tuple)) else ch
        cam_w = cw[0] if isinstance(cw, (list, tuple)) else cw
        keys = [f"{cam}_image" for _, cam in self.cameras]

        def _refresh_ok() -> bool:
            self.obs = self.env._get_observations(force_update=True)
            imgs = [self.obs[k] for k in keys if k in self.obs]
            if any(self._img_corrupt(im) for im in imgs):
                return False
            if len(imgs) >= 2:
                a, b = imgs[0], imgs[1]
                if a.shape == b.shape and bool((a == b).all()):
                    return False
            return True

        self.renderer_recoveries += 1
        print(f"[engine] ⚠ 渲染帧损坏，恢复渲染管线 (第{self.renderer_recoveries}次)...", flush=True)

        # L1: 轻量 con 重建
        ctx = sim._render_context_offscreen
        if ctx is not None:
            try:
                ctx.update_offscreen_size(cam_w + 64, cam_h + 64)
                ctx.update_offscreen_size(cam_w, cam_h)
                if _refresh_ok():
                    print("[engine] ✓ 渲染已恢复 (L1 con 重建)", flush=True)
                    return
            except Exception as e:
                print(f"[engine] L1 失败: {e}", flush=True)

        # L2: 整体 context 重建
        try:
            from robosuite.utils.binding_utils import MjRenderContextOffscreen
            device_id = getattr(self.env, "render_gpu_device_id", -1)
            orig_add = sim.add_render_context
            sim.add_render_context = lambda rc: None  # init 内不替换，保住旧 context
            try:
                new_ctx = MjRenderContextOffscreen(sim, device_id=device_id)
            finally:
                sim.add_render_context = orig_add
            orig_add(new_ctx)               # 现在才替换（del 旧 context 可能解绑 current）
            new_ctx.gl_ctx.make_current()   # 恢复 GL current 绑定
            new_ctx.update_offscreen_size(cam_w + 64, cam_h + 64)  # 强制 buffer 尺寸对齐
            new_ctx.update_offscreen_size(cam_w, cam_h)
            if _refresh_ok():
                print("[engine] ✓ 渲染已恢复 (L2 context 重建)", flush=True)
                return
        except Exception as e:
            print(f"[engine] L2 失败: {e}", flush=True)

        raise RuntimeError("渲染管线两级恢复后帧仍损坏")

    def _renderer_healthcheck(self):
        """每步调用：检查相机帧，损坏则重建 EGL offscreen context 并刷新观测。

        背景：新环境（py3.12）中 EGL offscreen context 偶发进入 wedge 状态，
        之后所有 readback 返回固定垃圾（竖条纹/黑帧），且不会自愈。
        重建 MjRenderContextOffscreen（init 内会 add_render_context 替换旧 context）
        可恢复。损坏帧若流入视觉模块会导致检测全灭，必须在观测入口处拦截。

        第二种损坏模式（cap-x/Piper 512px 高频渲染实测）: context 错乱后两个
        相机的 readback 返回**同一块 buffer**（top==wrist 像素级一致, 内容甚至是
        segmentation colormap）——统计指标完全正常, 只有跨相机一致性检查能抓到。
        """
        keys = [f"{cam}_image" for _, cam in self.cameras]
        imgs = [self.obs[k] for k in keys if k in self.obs]
        if any(self._img_corrupt(im) for im in imgs):
            self.recover_renderer()
            return
        if len(imgs) >= 2:
            a, b = imgs[0], imgs[1]
            if a.shape == b.shape and bool((a == b).all()):
                print("[engine] ⚠ 双相机帧完全一致（buffer 错乱），恢复渲染管线...", flush=True)
                self.recover_renderer()

    def step(self, action: np.ndarray):
        """primitives 的唯一仿真入口：步进并更新观测。"""
        if self.terminated:
            return  # episode 已终止（如 horizon 用尽），静默忽略后续 step
        self.obs, _, done, _ = self.env.step(action)
        self.sim_step += 1
        self._renderer_healthcheck()
        if self.sim_step % FRAME_INTERVAL == 0:
            # 帧流固定帧率：仅 RGB 双视角（depth 只在感知调用时被消费，边界才存）
            self.capture_boundary(self.sim_step, include_depth=False)
        if self.render:
            self.env.render()
            if self.render_slowdown > 1.0:
                import time as _time
                _time.sleep(0.05 * (self.render_slowdown - 1.0) / 20 * 20)
        if done:
            self.terminated = True

    def current_rgb(self, camera: str = "agentview") -> np.ndarray | None:
        img = self.obs.get(f"{camera}_image")
        return img.copy() if img is not None else None

    # ------------------------------------------------------------------
    # 帧流捕获（top/wrist/depth 原始帧，Tracer 按仿真步去重）
    # ------------------------------------------------------------------
    def capture_boundary(self, sim_step: int, include_depth: bool = True):
        """往帧流补一帧：API 调用边界（含 depth）与固定间隔（仅 RGB）都会调用。"""
        frames = {alias: self.current_rgb(real) for alias, real in self.cameras}
        if include_depth:
            frames["depth"] = self._depth_colormap()
        self.tracer.capture_frame(sim_step, frames)

    def _depth_colormap(self) -> np.ndarray | None:
        """顶部相机深度 → 固定量程 colormap（RGB，帧间亮度可比）。"""
        d = self.obs.get("agentview_depth")
        if d is None:
            return None
        if d.ndim == 3:
            d = d.squeeze(-1)
        d = np.clip(np.nan_to_num(np.asarray(d, dtype=np.float64), nan=1.0), 0.0, 1.0)
        real = CU.get_real_depth_map(self.env.sim, d)
        lo, hi = DEPTH_CLIP
        norm = (np.clip(real, lo, hi) - lo) / (hi - lo)
        cm = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        return cv2.cvtColor(cm, cv2.COLOR_BGR2RGB)

    def run(self, code: str, code_ref: str | None = None) -> dict:
        """执行任务代码，返回结果摘要并落盘 trace。

        code_ref: 任务代码路径（由 agent harness 传入），仅作引用记入 trace.json，
        代码本身不进 trace 目录——代码的生成/版本管理是 harness 的职责。
        """
        self.tracer.trace.code_ref = code_ref
        ns = self._build_namespace()
        buf = io.StringIO()
        error = None
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(code, "<task_code>", "exec"), ns)
        except Exception:
            error = traceback.format_exc()
        stdout_text = buf.getvalue()
        if error is None:
            try:
                success = bool(self.env._check_success())
            except Exception:
                success = False
        else:
            success = False
        self.capture_boundary(self.sim_step)  # 末帧：收尾状态入帧流
        self.tracer.log_stdout(stdout_text)
        trace_path = self.tracer.finalize(success, error, self.sim_step)
        return {
            "success": success,
            "error": error,
            "stdout": stdout_text,
            "sim_steps": self.sim_step,
            "trace_dir": self.trace_dir,
            "trace_path": trace_path,
            "n_primitive_calls": len(self.tracer.trace.records),
        }

    def close(self):
        self.env.close()


def main():
    parser = argparse.ArgumentParser(description="ASPIRE Execution Engine")
    parser.add_argument("--code", required=True, help="任务代码 .py 文件路径")
    parser.add_argument("--task", default="Lift")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trace-root", default="traces")
    parser.add_argument("--render", action="store_true", help="打开可视化仿真器窗口实时演示")
    parser.add_argument("--slow", type=float, default=1.0, help="演示减速倍率（如 2 = 半速）")
    args = parser.parse_args()

    with open(args.code, encoding="utf-8") as f:
        code = f.read()

    engine = ExecutionEngine(task=args.task, seed=args.seed, trace_root=args.trace_root,
                             render=args.render, render_slowdown=args.slow)
    try:
        result = engine.run(code, code_ref=args.code)
    finally:
        engine.close()

    print("=== Execution Result ===")
    print(f"success        : {result['success']}")
    print(f"sim_steps      : {result['sim_steps']}")
    print(f"primitive_calls: {result['n_primitive_calls']}")
    print(f"trace          : {result['trace_path']}")
    if result["stdout"]:
        print("--- stdout ---")
        print(result["stdout"])
    if result["error"]:
        print("--- error ---")
        print(result["error"])
    if args.render:
        import time as _time
        print("（窗口保持 5 秒后关闭）")
        try:
            for _ in range(250):  # 保持窗口响应 ~5s
                if engine.env.viewer is None:
                    break
                engine.env.render()
                _time.sleep(0.02)
        except (AttributeError, Exception):
            pass  # viewer 已销毁（如用户手动关窗）
    # 供 harness 解析的退出码
    raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
