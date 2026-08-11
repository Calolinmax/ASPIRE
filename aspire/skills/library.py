"""SkillLibrary —— 文件系统技能库（论文 §2.2 / E.2：文件系统即索引，in-context 注入）。

设计对应关系：
- 论文版 = `.claude/skills/<name>/SKILL.md` 自动发现 + CLAUDE.md 索引表；
  本库 = `skill_library/<category>/<name>.md` + `index.json`（git 可跟踪、跨进程共享）。
- 论文**无向量检索**（Limitations 明示未解决长期记忆管理）；本库默认**全量注入**
  （roadmap 既定：>20 条后启用 `retrieve()` 的 category/关键词打分检索）。
- coordinator **串行入库**（E.1："serializes skill admission to avoid conflicting
  library writes"）——`admit()` 走文件锁，actor 永不直接写库。
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from contextlib import contextmanager

from .schema import SkillEntry

# 进程内锁（同进程多线程）+ 文件锁（跨进程串行化）
_PROCESS_LOCK = threading.Lock()


@contextmanager
def _file_lock(lock_path: str, timeout: float = 30.0):
    """跨进程互斥：O_CREAT|O_EXCL 原子创建锁文件（POSIX 语义，NFS 外可靠）。"""
    fd = None
    t0 = time.time()
    while fd is None:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.time() - t0 > timeout:
                raise RuntimeError(f"skill library 锁等待超时: {lock_path}")
            time.sleep(0.2)
    try:
        yield
    finally:
        os.close(fd)
        os.unlink(lock_path)


class SkillLibrary:
    """技能库。线程安全 + 跨进程串行写；读无锁（文件系统快照）。"""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    # ------------------------------------------------------------------
    # 路径与索引
    # ------------------------------------------------------------------
    @property
    def index_path(self) -> str:
        return os.path.join(self.root, "index.json")

    @property
    def lock_path(self) -> str:
        return os.path.join(self.root, ".admit.lock")

    def _entry_path(self, entry: SkillEntry) -> str:
        return os.path.join(self.root, entry.category, f"{entry.name}.md")

    def _load_index(self) -> dict:
        if not os.path.isfile(self.index_path):
            return {"skills": {}}
        with open(self.index_path, encoding="utf-8") as f:
            return json.load(f)

    def _save_index(self, index: dict) -> None:
        # 原子写：临时文件 + rename（并发读到半截 JSON 的防线）
        fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.index_path)

    # ------------------------------------------------------------------
    # 读
    # ------------------------------------------------------------------
    def list(self, status: str | None = None, category: str | None = None) -> list[dict]:
        """索引列表（轻量，不读文件体）。"""
        skills = self._load_index().get("skills", {})
        out = []
        for name, meta in skills.items():
            if status and meta.get("status") != status:
                continue
            if category and meta.get("category") != category:
                continue
            out.append({"name": name, **meta})
        return sorted(out, key=lambda m: (m.get("category", ""), m.get("name", "")))

    def get(self, name: str) -> SkillEntry | None:
        """按名读条目全文（索引找路径，文件不在则全盘扫一次兜底）。"""
        meta = self._load_index().get("skills", {}).get(name)
        candidates = []
        if meta:
            candidates.append(os.path.join(self.root, meta["path"]))
        candidates.extend(
            os.path.join(dirpath, fname)
            for dirpath, _, files in os.walk(self.root)
            for fname in files if fname == f"{name}.md"
        )
        for path in candidates:
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as f:
                    return SkillEntry.from_markdown(f.read())
        return None

    def categories(self) -> list[str]:
        return sorted({m.get("category", "") for m in self._load_index().get("skills", {}).values()})

    # ------------------------------------------------------------------
    # 写（仅 coordinator 调用；文件锁串行化）
    # ------------------------------------------------------------------
    def admit(self, entry: SkillEntry, status: str = "admitted") -> str:
        """入库（新条目或覆盖更新）。返回落盘路径。

        同名处理（对齐 cap-x SkillLibrary.add_skill 语义）：occurrences 语义
        由 origin_tasks 并集表达；正文覆盖为最新验证版本；updated 刷新。
        """
        if not entry.name or not re.fullmatch(r"[a-z0-9][a-z0-9_\-]*", entry.name):
            raise ValueError(f"skill 名必须 kebab-case/snake_case: {entry.name!r}")
        entry.status = status
        with _PROCESS_LOCK, _file_lock(self.lock_path):
            old = self.get(entry.name)
            if old is not None:
                entry.origin_tasks = sorted(set(old.origin_tasks) | set(entry.origin_tasks))
                if not entry.created:
                    entry.created = old.created
            now = time.strftime("%Y-%m-%d")
            entry.created = entry.created or now
            entry.updated = now
            path = self._entry_path(entry)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(entry.to_markdown())
            index = self._load_index()
            index.setdefault("skills", {})[entry.name] = {
                "category": entry.category,
                "status": entry.status,
                "path": os.path.relpath(path, self.root),
                "description": entry.description or entry.problem.split("\n")[0][:120],
                "origin_tasks": entry.origin_tasks,
                "updated": entry.updated,
            }
            self._save_index(index)
        return path

    def retire(self, name: str) -> None:
        """条目退役（证伪/被取代）：不删文件，状态置 retired 并移出注入集。"""
        with _PROCESS_LOCK, _file_lock(self.lock_path):
            entry = self.get(name)
            if entry is None:
                raise KeyError(name)
            entry.status = "retired"
            with open(self._entry_path(entry), "w", encoding="utf-8") as f:
                f.write(entry.to_markdown())
            index = self._load_index()
            if name in index.get("skills", {}):
                index["skills"][name]["status"] = "retired"
            self._save_index(index)

    # ------------------------------------------------------------------
    # 注入（actor prompt 的 skill 段落）
    # ------------------------------------------------------------------
    def format_for_prompt(self, max_chars: int = 24000,
                          status: str = "admitted") -> str:
        """全量注入：拼接全部 admitted 条目全文（论文 E.3/E.4：actor 写候选前必读库）。

        超长时按 updated 倒序截断并显式标注（不静默截断——cap-x 报告纪律）。
        """
        entries = []
        for meta in self.list(status=status):
            entry = self.get(meta["name"])
            if entry is not None:
                entries.append(entry)
        if not entries:
            return "(skill library 当前为空——无先验条目，一切从 trace 诊断开始)"
        chunks = []
        total = 0
        dropped = 0
        for entry in sorted(entries, key=lambda e: e.updated, reverse=True):
            text = entry.to_markdown()
            if total + len(text) > max_chars and chunks:
                dropped += 1
                continue
            chunks.append(text)
            total += len(text)
        out = "\n\n---\n\n".join(chunks)
        if dropped:
            out += f"\n\n[注：另有 {dropped} 条 skill 因 token 预算未注入，"
            out += "可用 retrieve() 按关键词调取]"
        return out

    def retrieve(self, query: str, k: int = 5,
                 status: str = "admitted") -> list[SkillEntry]:
        """关键词打分检索（>20 条后的退路，roadmap 既定路线）。

        打分 = query 词与 name/category/description/when_to_apply 的重合计数；
        when_to_apply 命中权重 ×2（它是论文定义的检索 guard）。
        """
        words = {w.lower() for w in re.findall(r"[a-z0-9_一-鿿]+", query)}
        scored = []
        for meta in self.list(status=status):
            entry = self.get(meta["name"])
            if entry is None:
                continue
            hay_head = f"{entry.name} {entry.category} {entry.description}".lower()
            hay_guard = entry.when_to_apply.lower()
            score = sum(1 for w in words if w in hay_head)
            score += 2 * sum(1 for w in words if w in hay_guard)
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda x: (-x[0], x[1].name))
        return [e for _, e in scored[:k]]
