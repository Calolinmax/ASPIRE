"""ASPIRE 复现包：执行引擎、Primitive API、trace 记录。

子模块按需显式导入（e.g. `from aspire.engine.engine import ExecutionEngine`），
避免 vision_server 等独立进程被拖入 robosuite/mujoco/EGL 初始化。
"""

__all__ = []
