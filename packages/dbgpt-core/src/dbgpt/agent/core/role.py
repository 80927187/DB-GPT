"""基于角色（role-based）的对话角色类。"""

import json
import logging
import os
from abc import ABC
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Type, Union

from jinja2 import Environment, meta
from jinja2.sandbox import SandboxedEnvironment

from dbgpt._private.pydantic import BaseModel, ConfigDict, Field

from .action.base import ActionOutput
from .memory.agent_memory import (
    AgentMemory,
    AgentMemoryFragment,
    StructuredAgentMemoryFragment,
)
from .memory.llm import LLMImportanceScorer, LLMInsightExtractor
from .profile import Profile, ProfileConfig

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .agent import AgentMessage


class AgentRunMode(str, Enum):
    """智能体（agent）的运行模式。"""

    DEFAULT = "default"
    # 以循环模式（loop mode）运行智能体，直到对话结束（达到最大重试次数或遇到停止信号）。
    LOOP = "loop"


class Role(ABC, BaseModel):
    """基于角色（role-based）的对话角色类。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    profile: ProfileConfig = Field(
        ...,
        description="The profile of the role.",
    )
    memory: AgentMemory = Field(default_factory=AgentMemory)

    fixed_subgoal: Optional[str] = Field(None, description="Fixed subgoal")

    language: str = "en"
    is_human: bool = False
    is_team: bool = False

    template_env: SandboxedEnvironment = Field(default_factory=SandboxedEnvironment)

    # 任务进度跟踪（task progress tracking）：包含 'step'、'action'、'phase' 键的字典列表。
    # 这不是 Pydantic 字段，而是普通实例属性，因此可跨重试轮次保留，且不会序列化到记忆中。
    _task_progress: List[Dict] = []

    async def build_prompt(
        self,
        question: Optional[str] = None,
        is_system: bool = True,
        most_recent_memories: Optional[str] = None,
        resource_vars: Optional[Dict] = None,
        is_retry_chat: bool = False,
        **kwargs,
    ) -> str:
        """返回该角色的提示词模板（prompt template）。

        Returns:
            str：提示词模板。
        """
        if is_system:
            return self.current_profile.format_system_prompt(
                template_env=self.template_env,
                question=question,
                language=self.language,
                most_recent_memories=most_recent_memories,
                resource_vars=resource_vars,
                is_retry_chat=is_retry_chat,
                **kwargs,
            )
        else:
            return self.current_profile.format_user_prompt(
                template_env=self.template_env,
                question=question,
                language=self.language,
                most_recent_memories=most_recent_memories,
                resource_vars=resource_vars,
                **kwargs,
            )

    def identity_check(self) -> None:
        """检查角色身份（identity）。"""
        pass

    def get_name(self) -> str:
        """获取角色名称。"""
        return self.current_profile.get_name()

    @property
    def current_profile(self) -> Profile:
        """返回当前配置（profile）。"""
        profile = self.profile.create_profile(prefer_prompt_language=self.language)
        return profile

    def prompt_template(
        self,
        template_format: str = "f-string",
        language: str = "en",
        is_retry_chat: bool = False,
    ) -> str:
        """获取智能体提示词模板。"""
        self.language = language
        system_prompt = self.current_profile.get_system_prompt_template()
        # 通过沙箱环境（sandboxed environment）渲染，防止到达系统提示词模板的用户内容触发 SSTI。
        template = self.template_env.from_string(system_prompt)

        env = Environment()
        parsed_content = env.parse(system_prompt)
        variables = meta.find_undeclared_variables(parsed_content)

        role_params = {
            "role": self.role,
            "name": self.name,
            "goal": self.goal,
            "retry_goal": self.retry_goal,
            "expand_prompt": self.expand_prompt,
            "language": language,
            "constraints": self.constraints,
            "retry_constraints": self.retry_constraints,
            "examples": self.examples,
            "is_retry_chat": is_retry_chat,
        }
        param = role_params.copy()
        runtime_param_names = []
        for variable in variables:
            if variable not in role_params:
                runtime_param_names.append(variable)

        if template_format == "f-string":
            input_params = {}
            for variable in runtime_param_names:
                input_params[variable] = "{" + variable + "}"
            param.update(input_params)
        else:
            input_params = {}
            for variable in runtime_param_names:
                input_params[variable] = "{{" + variable + "}}"
            param.update(input_params)

        prompt_template = template.render(param)
        return prompt_template

    @property
    def name(self) -> str:
        """返回角色名称。"""
        return self.current_profile.get_name()

    @property
    def role(self) -> str:
        """返回角色的职责。"""
        return self.current_profile.get_role()

    @property
    def goal(self) -> Optional[str]:
        """返回角色目标。"""
        return self.current_profile.get_goal()

    @property
    def retry_goal(self) -> Optional[str]:
        """返回角色的重试目标。"""
        return self.current_profile.get_retry_goal()

    @property
    def constraints(self) -> Optional[List[str]]:
        """返回角色约束条件。"""
        return self.current_profile.get_constraints()

    @property
    def retry_constraints(self) -> Optional[List[str]]:
        """返回角色的重试约束条件。"""
        return self.current_profile.get_retry_constraints()

    @property
    def desc(self) -> Optional[str]:
        """返回角色描述。"""
        return self.current_profile.get_description()

    @property
    def expand_prompt(self) -> Optional[str]:
        """返回角色的扩展提示词介绍。"""
        return self.current_profile.get_expand_prompt()

    @property
    def write_memory_template(self) -> str:
        """返回当前的记忆保存模板。"""
        return self.current_profile.get_write_memory_template()

    @property
    def examples(self) -> Optional[str]:
        """返回当前示例模板。"""
        return self.current_profile.get_examples()

    @property
    def task_progress_summary(self) -> Optional[str]:
        """返回人类可读的任务进度摘要。

        列出智能体迄今执行的每个操作，并将最后一项标记为最近步骤。
        摘要会注入每次 LLM 调用，使模型不会忘记已完成的工作以及完成原始任务仍需执行的工作。
        """
        progress = getattr(self, "_task_progress", [])
        if not progress:
            return None
        # 运行时提示词保持英文原文，便于与既有调用方和解析逻辑保持兼容：任务进度（请勿重复已完成步骤）。
        lines = ["## Task Progress (do NOT repeat completed steps)"]
        for entry in progress:
            step = entry.get("step", "?")
            action = entry.get("action", "")
            phase = entry.get("phase", "")
            action_intention = entry.get("action_intention", "")
            status = entry.get("status", "done")
            snapshot_file = entry.get("snapshot_file", "")
            icon = "\u2705" if status == "done" else "\u274c"
            line = f"{icon} Step {step}: Action={action}"
            if action_intention:
                line += f" | Intention: {action_intention}"
            if phase:
                line += f" | Phase: {phase}"
            if snapshot_file:
                line += f" | ref: {os.path.basename(snapshot_file)}"
            lines.append(line)
        return "\n".join(lines)

    def _render_template(self, template: str, **kwargs):
        r_template = self.template_env.from_string(template)
        return r_template.render(**kwargs)

    @property
    def memory_importance_scorer(self) -> Optional[LLMImportanceScorer]:
        """创建记忆重要性评分器（memory importance scorer）。

        记忆重要性评分器用于评估记忆片段的重要性。
        """
        return None

    @property
    def memory_insight_extractor(self) -> Optional[LLMInsightExtractor]:
        """创建记忆洞察提取器（memory insight extractor）。

        记忆洞察提取器用于从记忆片段中提取高层次洞察。
        """
        return None

    @property
    def memory_fragment_class(self) -> Type[AgentMemoryFragment]:
        """返回记忆片段类（memory fragment class）。"""
        return AgentMemoryFragment

    async def read_memories(
        self,
        question: str,
    ) -> Union[str, List["AgentMessage"]]:
        """从记忆中读取历史内容。"""
        memories = await self.memory.read(question)
        recent_messages = [m.raw_observation for m in memories]
        return "".join(recent_messages)

    async def write_memories(
        self,
        question: str,
        ai_message: str,
        action_output: Optional[ActionOutput] = None,
        check_pass: bool = True,
        check_fail_reason: Optional[str] = None,
        current_retry_counter: Optional[int] = None,
    ) -> AgentMemoryFragment:
        """将内容写入记忆。

        建议根据实际需求重写此方法，将对话保存到记忆中。

        Args:
            question(str)：收到的问题。
            ai_message(str)：AI 消息，即 LLM 输出。
            action_output(ActionOutput)：操作输出。
            check_pass(bool)：检查是否通过。
            check_fail_reason(str)：检查失败原因。
            current_retry_counter(int)：当前重试计数器。

        Returns:
            AgentMemoryFragment：创建的记忆片段。
        """
        if not action_output:
            # 运行时异常提示保持英文原文，避免影响既有错误处理逻辑：保存记忆必须提供操作输出。
            raise ValueError("Action output is required to save to memory.")

        mem_thoughts = action_output.thoughts or ai_message
        action = action_output.action
        action_input = action_output.action_input
        phase = action_output.phase if hasattr(action_output, "phase") else None
        action_intention = (
            action_output.action_intention
            if hasattr(action_output, "action_intention")
            else None
        )
        action_reason = (
            action_output.action_reason
            if hasattr(action_output, "action_reason")
            else None
        )
        observation = check_fail_reason or action_output.observations

        # 当工具结果因输出过大而持久化到磁盘时，``content`` 保存带文件路径的
        # <persisted-output> 预览块，而 ``observations`` 保存完整内容。将预览块存入记忆，
        # 让 read_memories 重建有大小限制的 Observation，而不是完整输出；完整内容仍可通过
        # persisted_path / snapshot_path 在磁盘上获取。
        persisted_path = getattr(action_output, "persisted_path", None)
        if persisted_path and not check_fail_reason:
            observation = action_output.content

        memory_map = {
            "thought": mem_thoughts,
            "action": action,
            "observation": observation,
        }
        if action_input:
            memory_map["action_input"] = action_input
        if phase:
            memory_map["phase"] = phase
        if action_intention:
            memory_map["action_intention"] = action_intention
        if action_reason:
            memory_map["action_reason"] = action_reason
        if persisted_path:
            memory_map["persisted_path"] = persisted_path

        if current_retry_counter is not None and current_retry_counter == 0:
            memory_map["question"] = question

        # ------------------------------------------------------------------
        # 维护任务进度跟踪（可在缓冲区驱逐后保留）。
        # _task_progress 是实例上的普通列表，而不是 Pydantic 字段，因此不会被序列化/反序列化，
        # 并会在智能体对象的整个生命周期内保留在内存中。
        # ------------------------------------------------------------------
        snapshot_path: Optional[str] = None
        if check_pass and action:
            if not hasattr(self, "_task_progress") or self._task_progress is None:
                object.__setattr__(self, "_task_progress", [])
            progress: List[Dict] = self._task_progress  # type: ignore[assignment]
            step_num = (current_retry_counter or 0) + 1
            # 估算 observation 的 token 数量，用于预算跟踪。
            obs_tokens = len(observation) // 4 if observation else 0
            # 将完整操作细节写入快照文件，使第 1/2 层压缩（Layer 1/2 compaction）不会丢失
            # 精确值（action_input、observation）。
            snapshot_path = self._write_op_snapshot(
                step=step_num,
                action=action,
                action_input=action_input,
                observation=observation,
                thought=mem_thoughts,
                phase=phase,
                action_intention=action_intention,
                action_reason=action_reason,
            )
            progress.append(
                {
                    "step": step_num,
                    "action": action,
                    "phase": phase or "",
                    "action_intention": action_intention or "",
                    "action_reason": action_reason or "",
                    "status": "done",
                    "observation_tokens": obs_tokens,
                    "snapshot_file": snapshot_path or "",
                }
            )

        write_memory_template = self.write_memory_template
        memory_content = self._render_template(write_memory_template, **memory_map)

        fragment_cls: Type[AgentMemoryFragment] = self.memory_fragment_class
        if issubclass(fragment_cls, StructuredAgentMemoryFragment):
            fragment = fragment_cls(memory_map)
        else:
            fragment = fragment_cls(memory_content)
        fragment.snapshot_path = snapshot_path
        await self.memory.write(fragment)

        action_output.memory_fragments = {
            "memory": fragment.raw_observation,
            "id": fragment.id,
            "importance": fragment.importance,
        }
        return fragment

    def _write_op_snapshot(
        self,
        step: int,
        action: str,
        action_input: Optional[str],
        observation: Optional[str],
        thought: Optional[str],
        phase: Optional[str],
        action_intention: Optional[str] = None,
        action_reason: Optional[str] = None,
    ) -> Optional[str]:
        """将完整操作快照写入磁盘并返回文件路径。

        快照会保留完整的 action_input 和 observation，使第 1 层/第 2 层压缩不会丢失精确值
        （文件路径、计算结果、变量名等）。智能体之后可通过 ``read_file`` 操作读取此文件来恢复细节。

        如果智能体上下文中没有可用的 output_dir，则返回写入文件的绝对路径，否则返回 None。
        """
        # 从 AgentContext.output_dir 解析基础目录；若不可用，则回退到 DBGPT_HOME/workspace/op_snapshots。
        output_dir: Optional[str] = None
        ctx = getattr(self, "agent_context", None)
        if ctx is not None:
            output_dir = getattr(ctx, "output_dir", None)
        if not output_dir:
            home = os.environ.get("DBGPT_HOME", os.path.expanduser("~/.dbgpt"))
            output_dir = os.path.join(home, "workspace", "op_snapshots")

        conv_id = ""
        if ctx is not None:
            conv_id = getattr(ctx, "conv_id", "") or ""

        snapshot_dir = os.path.join(output_dir, conv_id) if conv_id else output_dir
        try:
            os.makedirs(snapshot_dir, exist_ok=True)
            safe_action = "".join(
                c if c.isalnum() or c in "-_" else "_" for c in action
            )
            filename = f"step_{step:03d}_{safe_action}.json"
            filepath = os.path.join(snapshot_dir, filename)
            payload = {
                "step": step,
                "action": action,
                "phase": phase or "",
                "action_intention": action_intention or "",
                "action_reason": action_reason or "",
                "thought": thought or "",
                "action_input": action_input or "",
                "observation": observation or "",
                "timestamp": datetime.utcnow().isoformat(),
                "conv_id": conv_id,
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            return filepath
        except Exception:
            # 日志字符串保持英文原文，便于检索既有日志：写入操作快照失败。
            logger.exception(
                "Failed to write op snapshot for step %d action %s", step, action
            )
            return None

    async def recovering_memory(self, action_outputs: List[ActionOutput]) -> None:
        """从操作输出中恢复记忆。"""
        fragments = []
        fragment_cls: Type[AgentMemoryFragment] = self.memory_fragment_class
        for action_output in action_outputs:
            if action_output.memory_fragments:
                fragment = fragment_cls.build_from(
                    observation=action_output.memory_fragments["memory"],
                    importance=action_output.memory_fragments.get("importance"),
                    memory_id=action_output.memory_fragments.get("id"),
                )
                fragments.append(fragment)
        await self.memory.write_batch(fragments)
