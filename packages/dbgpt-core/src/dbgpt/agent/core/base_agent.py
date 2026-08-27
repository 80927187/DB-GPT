"""可对话智能体的基础智能体类。"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from concurrent.futures import Executor, ThreadPoolExecutor
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, cast, final

from jinja2.sandbox import SandboxedEnvironment

from dbgpt._private.pydantic import ConfigDict, Field
from dbgpt.core import LLMClient, ModelMessageRoleType, PromptTemplate
from dbgpt.util.error_types import LLMChatError
from dbgpt.util.executor_utils import blocking_func_to_async
from dbgpt.util.tracer import SpanType, root_tracer
from dbgpt.util.utils import colored

from ..resource.base import Resource
from ..util.conv_utils import parse_conv_id
from ..util.llm.llm import LLMConfig, LLMStrategyType
from ..util.llm.llm_client import AIWrapper
from .action.base import Action, ActionOutput
from .agent import Agent, AgentContext, AgentMessage, AgentReviewInfo
from .context import ContextBudgetConfig, ContextManager
from .context.manager import ContextStatusCallback
from .memory.agent_memory import AgentMemory
from .memory.gpts.base import GptsMessage
from .memory.gpts.gpts_memory import GptsMemory
from .profile.base import ProfileConfig
from .role import AgentRunMode, Role

logger = logging.getLogger(__name__)


class ConversableAgent(Role, Agent):
    """ConversableAgent 是可与其他智能体通信的智能体。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent_context: Optional[AgentContext] = Field(None, description="Agent context")
    actions: List[Action] = Field(default_factory=list)
    resource: Optional[Resource] = Field(None, description="Resource")
    llm_config: Optional[LLMConfig] = None
    bind_prompt: Optional[PromptTemplate] = None
    run_mode: Optional[AgentRunMode] = Field(default=None, description="Run mode")
    max_retry_count: int = 3
    max_timeout: int = 600
    llm_client: Optional[AIWrapper] = None
    # 确认当前Agent是否需要进行流式输出
    stream_out: bool = True
    # 确认当前Agent是否需要进行参考资源展示
    show_reference: bool = False

    # 多层上下文管理（通过 enable_context_management() 初始化）
    _context_manager: Optional[ContextManager] = None

    executor: Executor = Field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=1),
        description="Executor for running tasks",
    )

    def __init__(self, **kwargs):
        """创建一个新智能体。"""
        Role.__init__(self, **kwargs)
        Agent.__init__(self)

    def init_context_management(
        self,
        config: Optional[ContextBudgetConfig] = None,
        model_name: Optional[str] = None,
        on_status_event: Optional[ContextStatusCallback] = None,
    ) -> None:
        """初始化多层上下文管理。

        请在智能体完成全部配置（如已设置 llm_client）后调用此方法。

        参数：
            config: 预算配置。若为 None，则从 agent_context 派生。
            model_name: 用于选择分词器的模型名称。
            on_status_event: 使用上下文状态字典调用的异步回调，
                以便调用方（如 SSE 层）推送实时更新。
        """
        if config is None:
            ctx = self.agent_context
            if ctx is None:
                config = ContextBudgetConfig()
            else:
                config = ContextBudgetConfig(
                    max_context_tokens=ctx.max_context_tokens,
                    warning_threshold=ctx.context_warning_threshold,
                    error_threshold=ctx.context_error_threshold,
                )
        llm_client = self.llm_client
        object.__setattr__(
            self,
            "_context_manager",
            ContextManager(
                config=config,
                model_name=model_name,
                llm_client=llm_client,
                on_status_event=on_status_event,
            ),
        )
        # 为第 2 层初始化 ToolResultStorage（逐结果持久化）。
        # 存储目录：{output_dir}/persisted_results/{conv_id}/，与 op_snapshots/
        # 同级。无法使用时回退到 DBGPT_HOME/workspace/persisted_results/。
        self._init_tool_result_storage(config)
        logger.info("Context management enabled for agent %s", self.name)

    def _init_tool_result_storage(self, config: ContextBudgetConfig) -> None:
        """初始化绑定到当前智能体会话的 ToolResultStorage。

        从 ``AgentContext.output_dir``（或 ``DBGPT_HOME/workspace``）
        解析存储目录，并通过 ``set_current_storage`` 将其绑定到
        当前异步上下文，使独立的 ``run_tool`` 函数无需更改签名
        即可获取该存储。
        """
        import os

        from .context.storage import ToolResultStorage

        ctx = self.agent_context
        output_dir: Optional[str] = None
        conv_id = ""
        if ctx is not None:
            output_dir = getattr(ctx, "output_dir", None)
            conv_id = getattr(ctx, "conv_id", "") or ""
        if not output_dir:
            home = os.environ.get("DBGPT_HOME", os.path.expanduser("~/.dbgpt"))
            output_dir = os.path.join(home, "workspace")
        storage_dir = os.path.join(output_dir, "persisted_results")
        if conv_id:
            storage_dir = os.path.join(storage_dir, conv_id)
        storage = ToolResultStorage(
            storage_dir=storage_dir,
            default_threshold=config.tool_result_threshold,
            preview_size=config.preview_size,
            tool_overrides=config.tool_overrides,
        )
        object.__setattr__(self, "_result_storage", storage)

    def check_available(self) -> None:
        """检查智能体是否可用。

        异常：
            ValueError: 智能体不可用时抛出。
        """
        self.identity_check()
        # 检查运行上下文
        if self.agent_context is None:
            raise ValueError(
                f"{self.name}[{self.role}] Missing context in which agent is running!"
            )

        # 检查动作
        if self.actions and len(self.actions) > 0:
            for action in self.actions:
                if action.resource_need and (
                    not self.resource
                    or not self.resource.get_resource_by_type(action.resource_need)
                ):
                    raise ValueError(
                        f"{self.name}[{self.role}] Missing resources"
                        f"[{action.resource_need}] required for runtime！"
                    )
        else:
            if not self.is_human and not self.is_team:
                raise ValueError(
                    f"This agent {self.name}[{self.role}] is missing action modules."
                )
        # 检查 LLM
        if not self.is_human and (
            self.llm_config is None or self.llm_config.llm_client is None
        ):
            raise ValueError(
                f"{self.name}[{self.role}] Model configuration is missing or model "
                "service is unavailable！"
            )

    @property
    def not_null_agent_context(self) -> AgentContext:
        """获取智能体上下文。

        返回：
            AgentContext: 智能体上下文。

        异常：
            ValueError: 智能体上下文未初始化时抛出。
        """
        if not self.agent_context:
            raise ValueError("Agent context is not initialized！")
        return self.agent_context

    @property
    def not_null_llm_config(self) -> LLMConfig:
        """获取 LLM 配置。"""
        if not self.llm_config:
            raise ValueError("LLM config is not initialized！")
        return self.llm_config

    @property
    def not_null_llm_client(self) -> LLMClient:
        """获取 LLM 客户端。"""
        llm_client = self.not_null_llm_config.llm_client
        if not llm_client:
            raise ValueError("LLM client is not initialized！")
        return llm_client

    async def blocking_func_to_async(
        self, func: Callable[..., Any], *args, **kwargs
    ) -> Any:
        """在执行器中运行可能阻塞的函数。"""
        if not asyncio.iscoroutinefunction(func):
            return await blocking_func_to_async(self.executor, func, *args, **kwargs)
        return await func(*args, **kwargs)

    async def preload_resource(self) -> None:
        """在初始化智能体前预加载资源。"""
        if self.resource:
            await self.resource.preload_resource()

    async def build(self, is_retry_chat: bool = False) -> "ConversableAgent":
        """构建智能体。"""
        # 预加载资源
        await self.preload_resource()
        # 检查智能体是否可用
        self.check_available()
        _language = self.not_null_agent_context.language
        if _language:
            self.language = _language

        # 初始化资源加载器
        for action in self.actions:
            action.init_resource(self.resource)

        # 初始化 LLM 服务
        if not self.is_human:
            if not self.llm_config or not self.llm_config.llm_client:
                raise ValueError("LLM client is not initialized！")
            self.llm_client = AIWrapper(llm_client=self.llm_config.llm_client)
            real_conv_id, _ = parse_conv_id(self.not_null_agent_context.conv_id)
            memory_session = f"{real_conv_id}_{self.role}_{self.name}"
            self.memory.initialize(
                self.name,
                self.llm_config.llm_client,
                importance_scorer=self.memory_importance_scorer,
                insight_extractor=self.memory_insight_extractor,
                session_id=memory_session,
            )
            # 克隆记忆结构
            self.memory = self.memory.structure_clone()
            action_outputs = await self.memory.gpts_memory.get_agent_history_memory(
                real_conv_id, self.role
            )
            await self.recovering_memory(action_outputs)
        return self

    def bind(self, target: Any) -> "ConversableAgent":
        """将资源绑定到智能体。"""
        # 支持绑定 Skill 实例，使智能体可通过 .bind(skill) 接收技能
        # 支持绑定 FileBasedSkill（Claude 风格）：将其转换为
        # 核心 Skill 实例。这样调用方既可传入核心 Skill，
        # 也可传入基于文件的技能解析结果。
        try:
            from dbgpt.agent.claude_skill import FileBasedSkill
        except Exception:
            FileBasedSkill = None  # type: ignore

        # 如果传入了 FileBasedSkill 实例，尝试将其转换为核心 Skill，
        # 以便下游代码统一处理技能。
        if FileBasedSkill is not None and isinstance(target, FileBasedSkill):
            try:
                from dbgpt.agent.skill.base import Skill, SkillMetadata, SkillType

                meta = target.metadata
                skill_type_val = SkillType.Custom
                if getattr(meta, "skill_type", None):
                    try:
                        skill_type_val = SkillType(meta.skill_type)
                    except Exception:
                        skill_type_val = SkillType.Custom

                core_meta = SkillMetadata(
                    name=meta.name,
                    description=meta.description,
                    version=getattr(meta, "version", "1.0.0") or "1.0.0",
                    author=getattr(meta, "author", None),
                    skill_type=skill_type_val,
                    tags=getattr(meta, "tags", []) or [],
                )

                prompt_template = None
                if hasattr(target, "get_prompt"):
                    try:
                        prompt_template = target.get_prompt()
                    except Exception:
                        pass
                if prompt_template is None and hasattr(target, "instructions"):
                    prompt_template = PromptTemplate.from_template(target.instructions)

                skill_obj = Skill(
                    metadata=core_meta,
                    prompt_template=prompt_template,
                    required_tools=getattr(meta, "required_tools", []) or [],
                    required_knowledge=getattr(meta, "required_knowledge", []) or [],
                    config=getattr(meta, "config", {}) or {},
                )

                # 用构建出的核心 Skill 实例替换 target
                target = skill_obj
            except Exception:
                # 转换失败时继续执行，交由后续检查处理
                pass

        try:
            # 局部导入，避免模块导入时出现循环依赖
            from dbgpt.agent.skill.base import SkillBase

            is_skill = isinstance(target, SkillBase)
        except Exception:
            is_skill = False
        if isinstance(target, LLMConfig):
            self.llm_config = target
        elif isinstance(target, GptsMemory):
            raise ValueError("GptsMemory is not supported!Please Use Agent Memory")
        elif isinstance(target, AgentContext):
            self.agent_context = target
        elif isinstance(target, Resource):
            self.resource = target
        elif isinstance(target, AgentMemory):
            self.memory = target
        elif isinstance(target, ProfileConfig):
            self.profile = target
        elif is_skill:
            # 将技能绑定到智能体，并将其提示模板设为 bind_prompt，
            # 使技能指令成为智能体的系统提示词。
            self._skill = target
            try:
                prompt_template = getattr(target, "prompt_template", None)
                if prompt_template is not None:
                    self.bind_prompt = cast(Optional[PromptTemplate], prompt_template)
            except Exception:
                pass
        elif isinstance(target, type) and issubclass(target, Action):
            self.actions.append(target())
        elif isinstance(target, Action):
            self.actions.append(target)
        elif isinstance(target, list) and all(
            [isinstance(item, type) and issubclass(item, Action) for item in target]
        ):
            for action in target:
                self.actions.append(action())
        elif isinstance(target, list) and all(
            [isinstance(item, Action) for item in target]
        ):
            self.actions.extend(target)
        elif isinstance(target, PromptTemplate):
            self.bind_prompt = target

        return self

    async def send(
        self,
        message: AgentMessage,
        recipient: Agent,
        reviewer: Optional[Agent] = None,
        request_reply: Optional[bool] = True,
        is_recovery: Optional[bool] = False,
        silent: Optional[bool] = False,
        is_retry_chat: bool = False,
        last_speaker_name: Optional[str] = None,
        rely_messages: Optional[List[AgentMessage]] = None,
        historical_dialogues: Optional[List[AgentMessage]] = None,
    ) -> None:
        """向接收方智能体发送消息。"""
        with root_tracer.start_span(
            "agent.send",
            metadata={
                "sender": self.name,
                "recipient": recipient.name,
                "reviewer": reviewer.name if reviewer else None,
                "agent_message": json.dumps(message.to_dict(), ensure_ascii=False),
                "request_reply": request_reply,
                "is_recovery": is_recovery,
                "conv_uid": self.not_null_agent_context.conv_id,
            },
        ):
            await recipient.receive(
                message=message,
                sender=self,
                reviewer=reviewer,
                request_reply=request_reply,
                is_recovery=is_recovery,
                silent=silent,
                is_retry_chat=is_retry_chat,
                last_speaker_name=last_speaker_name,
                historical_dialogues=historical_dialogues,
                rely_messages=rely_messages,
            )

    async def receive(
        self,
        message: AgentMessage,
        sender: Agent,
        reviewer: Optional[Agent] = None,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
        is_recovery: Optional[bool] = False,
        is_retry_chat: bool = False,
        last_speaker_name: Optional[str] = None,
        historical_dialogues: Optional[List[AgentMessage]] = None,
        rely_messages: Optional[List[AgentMessage]] = None,
    ) -> None:
        """接收其他智能体发送的消息。"""
        with root_tracer.start_span(
            "agent.receive",
            metadata={
                "sender": sender.name,
                "recipient": self.name,
                "reviewer": reviewer.name if reviewer else None,
                "agent_message": json.dumps(message.to_dict(), ensure_ascii=False),
                "request_reply": request_reply,
                "silent": silent,
                "is_recovery": is_recovery,
                "conv_uid": self.not_null_agent_context.conv_id,
                "is_human": self.is_human,
            },
        ):
            await self._a_process_received_message(message, sender)
            if request_reply is False or request_reply is None:
                return

            if not self.is_human:
                if isinstance(sender, ConversableAgent) and sender.is_human:
                    reply = await self.generate_reply(
                        received_message=message,
                        sender=sender,
                        reviewer=reviewer,
                        is_retry_chat=is_retry_chat,
                        last_speaker_name=last_speaker_name,
                        historical_dialogues=historical_dialogues,
                        rely_messages=rely_messages,
                    )
                else:
                    reply = await self.generate_reply(
                        received_message=message,
                        sender=sender,
                        reviewer=reviewer,
                        is_retry_chat=is_retry_chat,
                        historical_dialogues=historical_dialogues,
                        rely_messages=rely_messages,
                    )

                if reply is not None:
                    await self.send(reply, sender)

    def prepare_act_param(
        self,
        received_message: Optional[AgentMessage],
        sender: Agent,
        rely_messages: Optional[List[AgentMessage]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """为 act 方法准备参数。"""
        return {}

    @final
    async def generate_reply(
        self,
        received_message: AgentMessage,
        sender: Agent,
        reviewer: Optional[Agent] = None,
        rely_messages: Optional[List[AgentMessage]] = None,
        historical_dialogues: Optional[List[AgentMessage]] = None,
        is_retry_chat: bool = False,
        last_speaker_name: Optional[str] = None,
        **kwargs,
    ) -> AgentMessage:
        """根据收到的消息生成回复。"""
        stream_callback = kwargs.pop("stream_callback", None)

        async def _emit_stream(event_type: str, payload: Dict[str, Any]) -> None:
            if not stream_callback:
                return
            try:
                if asyncio.iscoroutinefunction(stream_callback):
                    await stream_callback(event_type, payload)
                    return
                result = stream_callback(event_type, payload)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("stream_callback error")

        logger.info(
            f"generate agent reply!sender={sender}, rely_messages_len={rely_messages}"
        )
        root_span = root_tracer.start_span(
            "agent.generate_reply",
            metadata={
                "sender": sender.name,
                "recipient": self.name,
                "reviewer": reviewer.name if reviewer else None,
                "received_message": json.dumps(received_message.to_dict()),
                "conv_uid": self.not_null_agent_context.conv_id,
                "rely_messages": (
                    [msg.to_dict() for msg in rely_messages] if rely_messages else None
                ),
            },
        )
        reply_message = None

        try:
            with root_tracer.start_span(
                "agent.generate_reply._init_reply_message",
            ) as span:
                # 初始化回复消息
                a_reply_message: Optional[
                    AgentMessage
                ] = await self._a_init_reply_message(received_message=received_message)
                if a_reply_message:
                    reply_message = a_reply_message
                else:
                    reply_message = self._init_reply_message(
                        received_message=received_message
                    )
                span.metadata["reply_message"] = reply_message.to_dict()

            fail_reason = None
            current_retry_counter = 0
            start_time = time.time()
            is_success = True
            observation = received_message.content or ""
            while current_retry_counter < self.max_retry_count:
                if current_retry_counter > 0:
                    a_reply_message: Optional[
                        AgentMessage
                    ] = await self._a_init_reply_message(
                        received_message=received_message,
                        rely_messages=rely_messages,
                    )
                    if a_reply_message:
                        retry_message = a_reply_message
                    else:
                        retry_message = self._init_reply_message(
                            received_message=received_message,
                            rely_messages=rely_messages,
                        )

                    retry_message.rounds = reply_message.rounds + 1

                    retry_message.content = fail_reason or observation
                    retry_message.current_goal = received_message.current_goal

                    # 当前消息是自我优化消息，
                    # 需要将其记录下来。
                    # 暂时将其设为由发起端发起，
                    # 以便组织历史记忆上下文。
                    await sender.send(
                        retry_message, self, reviewer, request_reply=False
                    )
                    reply_message.rounds = retry_message.rounds + 1

                # 手动重试模式下，将上一位发言者的全部消息加载为依赖消息 # noqa
                logger.info(
                    f"Depends on the number of historical messages:{len(rely_messages) if rely_messages else 0}！"  # noqa
                )
                thinking_messages, resource_info = await self._load_thinking_messages(
                    received_message=received_message,
                    sender=sender,
                    observation=observation,
                    rely_messages=rely_messages,
                    historical_dialogues=historical_dialogues,
                    context=reply_message.get_dict_context(),
                    is_retry_chat=is_retry_chat,
                    current_retry_counter=current_retry_counter,
                )
                with root_tracer.start_span(
                    "agent.generate_reply.thinking",
                    metadata={
                        "thinking_messages": json.dumps(
                            [msg.to_dict() for msg in thinking_messages],
                            ensure_ascii=False,
                        )
                    },
                ) as span:
                    # 1. 思考任务的处理方式
                    async def _llm_stream_callback(payload: Dict[str, Any]) -> None:
                        await _emit_stream(
                            "thinking_chunk",
                            {
                                "round": current_retry_counter + 1,
                                "delta_text": payload.get("delta_text", ""),
                                "delta_thinking": payload.get("delta_thinking", ""),
                            },
                        )

                    try:
                        llm_reply, model_name = await self.thinking(
                            thinking_messages,
                            sender,
                            stream_callback=_llm_stream_callback,
                        )
                    except LLMChatError as e:
                        # 第 4 层：出现 context_too_long 时进行响应式压缩
                        _ctx_mgr: Optional[ContextManager] = getattr(
                            self, "_context_manager", None
                        )
                        err_str = str(e).lower()
                        if _ctx_mgr and (
                            "context_too_long" in err_str
                            or "context_length_exceeded" in err_str
                            or "maximum context length" in err_str
                        ):
                            logger.warning(
                                "LLM context overflow detected — applying "
                                "reactive compaction (Layer 4)"
                            )
                            thinking_messages = await _ctx_mgr.reactive_compact(
                                thinking_messages
                            )
                            llm_reply, model_name = await self.thinking(
                                thinking_messages,
                                sender,
                                stream_callback=_llm_stream_callback,
                            )
                        else:
                            raise
                    reply_message.model_name = model_name
                    reply_message.content = llm_reply
                    reply_message.resource_info = resource_info
                    span.metadata["llm_reply"] = llm_reply
                    span.metadata["model_name"] = model_name
                    await _emit_stream(
                        "thinking",
                        {
                            "round": current_retry_counter + 1,
                            "llm_reply": llm_reply,
                            "model_name": model_name,
                        },
                    )

                with root_tracer.start_span(
                    "agent.generate_reply.review",
                    metadata={"llm_reply": llm_reply, "censored": self.name},
                ) as span:
                    # 2. 审查当前操作是否合法
                    approve, comments = await self.review(llm_reply, self)
                    reply_message.review_info = AgentReviewInfo(
                        approve=approve,
                        comments=comments,
                    )
                    span.metadata["approve"] = approve
                    span.metadata["comments"] = comments

                act_extent_param = self.prepare_act_param(
                    received_message=received_message,
                    sender=sender,
                    rely_messages=rely_messages,
                    historical_dialogues=historical_dialogues,
                )
                with root_tracer.start_span(
                    "agent.generate_reply.act",
                    metadata={
                        "llm_reply": llm_reply,
                        "sender": sender.name,
                        "reviewer": reviewer.name if reviewer else None,
                        "act_extent_param": act_extent_param,
                    },
                ) as span:
                    # 3. 根据思考结果执行动作
                    act_out: ActionOutput = await self.act(
                        message=reply_message,
                        sender=sender,
                        reviewer=reviewer,
                        is_retry_chat=is_retry_chat,
                        last_speaker_name=last_speaker_name,
                        **act_extent_param,
                    )
                    if act_out:
                        reply_message.action_report = act_out
                    span.metadata["action_report"] = (
                        act_out.to_dict() if act_out else None
                    )
                    await _emit_stream(
                        "act",
                        {
                            "round": current_retry_counter + 1,
                            "action_output": act_out.to_dict() if act_out else None,
                        },
                    )

                with root_tracer.start_span(
                    "agent.generate_reply.verify",
                    metadata={
                        "llm_reply": llm_reply,
                        "sender": sender.name,
                        "reviewer": reviewer.name if reviewer else None,
                    },
                ) as span:
                    # 4. 验证回复信息
                    check_pass, reason = await self.verify(
                        reply_message, sender, reviewer
                    )
                    is_success = check_pass
                    span.metadata["check_pass"] = check_pass
                    span.metadata["reason"] = reason

                question: str = received_message.content or ""
                ai_message: str = llm_reply or ""
                # 5. 自我优化错误答案
                if not check_pass:
                    if not act_out.have_retry:
                        logger.warning("No retry available!")
                        break
                    fail_reason = reason
                    observation = fail_reason
                    await self.write_memories(
                        question=question,
                        ai_message=ai_message,
                        action_output=act_out,
                        check_pass=check_pass,
                        check_fail_reason=fail_reason,
                        current_retry_counter=current_retry_counter,
                    )
                else:
                    # 回复成功
                    observation = act_out.observations
                    await self.write_memories(
                        question=question,
                        ai_message=ai_message,
                        action_output=act_out,
                        check_pass=check_pass,
                        current_retry_counter=current_retry_counter,
                    )
                    if self.run_mode != AgentRunMode.LOOP or act_out.terminate:
                        logger.debug(f"Agent {self.name} reply success!{reply_message}")
                        break
                time_cost = time.time() - start_time
                if time_cost > self.max_timeout:
                    logger.warning(
                        f"Agent {self.name} run time out!{time_cost} > "
                        f"{self.max_timeout}"
                    )
                    break

                # 继续运行下一轮
                current_retry_counter += 1
                # 发送错误消息并下达新的问题解决指令
                if current_retry_counter < self.max_retry_count:
                    await self.send(
                        reply_message, sender, reviewer, request_reply=False
                    )

            reply_message.success = is_success
            # 6. 调整最终消息
            await self.adjust_final_message(is_success, reply_message)
            return reply_message

        except Exception as e:
            logger.exception("Generate reply exception!")
            err_message = AgentMessage(content=str(e))
            err_message.success = False
            return err_message
        finally:
            if reply_message:
                root_span.metadata["reply_message"] = reply_message.to_dict()
            root_span.end()

    async def thinking(
        self,
        messages: List[AgentMessage],
        sender: Optional[Agent] = None,
        prompt: Optional[str] = None,
        stream_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """针对当前任务目标进行思考和推理。

        参数：
            messages(List[AgentMessage]): 待推理的消息
            prompt(str): 待推理的提示词
        """
        last_model = None
        last_err = None
        retry_count = 0
        llm_messages = [message.to_llm_message() for message in messages]
        # LLM 推理自动重试 3 次，以降低限速和网络不稳定
        # 导致的中断概率
        while retry_count < 3:
            llm_model = await self._a_select_llm_model(last_model)
            try:
                if prompt:
                    llm_messages = _new_system_message(prompt) + llm_messages

                if not self.llm_client:
                    raise ValueError("LLM client is not initialized!")
                response = await self.llm_client.create(
                    context=llm_messages[-1].pop("context", None),
                    messages=llm_messages,
                    llm_model=llm_model,
                    max_new_tokens=self.not_null_agent_context.max_new_tokens,
                    temperature=self.not_null_agent_context.temperature,
                    verbose=self.not_null_agent_context.verbose,
                    memory=self.memory.gpts_memory,
                    conv_id=self.not_null_agent_context.conv_id,
                    sender=sender.role if sender else "?",
                    stream_out=self.stream_out,
                    stream_callback=stream_callback,
                )
                return response, llm_model
            except LLMChatError as e:
                logger.error(f"model:{llm_model} generate Failed!{str(e)}")
                retry_count += 1
                last_model = llm_model
                last_err = str(e)
                await asyncio.sleep(10)

        if last_err:
            raise ValueError(last_err)
        else:
            raise ValueError("LLM model inference failed!")

    async def review(self, message: Optional[str], censored: Agent) -> Tuple[bool, Any]:
        """根据待审查对象审查消息。"""
        return True, None

    async def act(
        self,
        message: AgentMessage,
        sender: Agent,
        reviewer: Optional[Agent] = None,
        is_retry_chat: bool = False,
        last_speaker_name: Optional[str] = None,
        **kwargs,
    ) -> ActionOutput:
        """执行动作。"""
        # 将当前智能体的 ToolResultStorage 绑定到当前异步上下文，
        # 使独立的 run_tool() 可访问它并持久化超大结果。
        # act() 执行完成后，将令牌重置为先前的值。
        from .context.storage import get_current_storage, set_current_storage

        prev_storage = get_current_storage()
        bound_storage = getattr(self, "_result_storage", None)
        if bound_storage is not None:
            set_current_storage(bound_storage)
        try:
            last_out: Optional[ActionOutput] = None
            for i, action in enumerate(self.actions):
                if not message:
                    raise ValueError("The message content is empty!")

                with root_tracer.start_span(
                    "agent.act.run",
                    metadata={
                        "message": message,
                        "sender": sender.name if sender else None,
                        "recipient": self.name,
                        "reviewer": reviewer.name if reviewer else None,
                        "rely_action_out": last_out.to_dict() if last_out else None,
                        "conv_uid": self.not_null_agent_context.conv_id,
                        "action_index": i,
                        "total_action": len(self.actions),
                    },
                ) as span:
                    ai_message = message.content if message.content else ""
                    real_action = action.parse_action(
                        ai_message, default_action=action, **kwargs
                    )
                    if real_action is None:
                        continue

                    last_out = await real_action.run(
                        ai_message=message.content if message.content else "",
                        resource=None,
                        rely_action_out=last_out,
                        **kwargs,
                    )
                    span.metadata["action_out"] = (
                        last_out.to_dict() if last_out else None
                    )
            if not last_out:
                raise ValueError("Action should return value！")
            return last_out
        finally:
            # 恢复先前的存储绑定（支持嵌套的智能体调用）。
            set_current_storage(prev_storage)

    async def correctness_check(
        self, message: AgentMessage
    ) -> Tuple[bool, Optional[str]]:
        """验证结果的正确性。"""
        return True, None

    async def verify(
        self,
        message: AgentMessage,
        sender: Agent,
        reviewer: Optional[Agent] = None,
        **kwargs,
    ) -> Tuple[bool, Optional[str]]:
        """验证当前执行结果。"""
        # 检查审批结果
        if message.review_info and not message.review_info.approve:
            return False, message.review_info.comments

        # 检查动作运行结果
        action_output: Optional[ActionOutput] = message.action_report
        if action_output:
            if not action_output.is_exe_success:
                return False, action_output.content
            elif not action_output.content or len(action_output.content.strip()) < 1:
                return (
                    False,
                    "The current execution result is empty. Please rethink the "
                    "question and background and generate a new answer.. ",
                )

        # 检查智能体输出的正确性
        return await self.correctness_check(message)

    async def initiate_chat(
        self,
        recipient: Agent,
        reviewer: Optional[Agent] = None,
        message: Optional[str] = None,
        request_reply: bool = True,
        is_retry_chat: bool = False,
        last_speaker_name: Optional[str] = None,
        message_rounds: int = 0,
        historical_dialogues: Optional[List[AgentMessage]] = None,
        rely_messages: Optional[List[AgentMessage]] = None,
        **context,
    ):
        """发起与另一个智能体的对话。

        参数：
            recipient (Agent): 接收方智能体。
            reviewer (Agent): 审查方智能体。
            message (str): 要发送的消息。
        """
        agent_message = AgentMessage(
            content=message,
            current_goal=message,
            rounds=message_rounds,
            context=context,
        )
        with root_tracer.start_span(
            "agent.initiate_chat",
            span_type=SpanType.AGENT,
            metadata={
                "sender": self.name,
                "recipient": recipient.name,
                "reviewer": reviewer.name if reviewer else None,
                "agent_message": json.dumps(
                    agent_message.to_dict(), ensure_ascii=False
                ),
                "conv_uid": self.not_null_agent_context.conv_id,
            },
        ):
            await self.send(
                agent_message,
                recipient,
                reviewer,
                historical_dialogues=historical_dialogues,
                rely_messages=rely_messages,
                request_reply=request_reply,
                is_retry_chat=is_retry_chat,
                last_speaker_name=last_speaker_name,
            )

    async def adjust_final_message(
        self,
        is_success: bool,
        reply_message: AgentMessage,
    ):
        """在智能体回复后调整最终消息。"""
        return is_success, reply_message

    #######################################################################
    # 私有函数开始
    #######################################################################

    def _init_actions(self, actions: List[Type[Action]]):
        self.actions = []
        for idx, action in enumerate(actions):
            if issubclass(action, Action):
                self.actions.append(action(language=self.language))

    async def _a_append_message(
        self, message: AgentMessage, role, sender: Agent
    ) -> bool:
        gpts_message: GptsMessage = GptsMessage(
            conv_id=self.not_null_agent_context.conv_id,
            sender=sender.role,
            receiver=self.role,
            role=role,
            rounds=message.rounds,
            is_success=message.success,
            app_code=(
                sender.not_null_agent_context.gpts_app_code
                if isinstance(sender, ConversableAgent)
                else None
            ),
            app_name=(
                sender.not_null_agent_context.gpts_app_name
                if isinstance(sender, ConversableAgent)
                else None
            ),
            current_goal=message.current_goal,
            content=message.content if message.content else "",
            context=(
                json.dumps(message.context, ensure_ascii=False)
                if message.context
                else None
            ),
            review_info=(
                json.dumps(message.review_info.to_dict(), ensure_ascii=False)
                if message.review_info
                else None
            ),
            action_report=(
                json.dumps(message.action_report.to_dict(), ensure_ascii=False)
                if message.action_report
                else None
            ),
            model_name=message.model_name,
            resource_info=(
                json.dumps(message.resource_info) if message.resource_info else None
            ),
        )

        with root_tracer.start_span(
            "agent.save_message_to_memory",
            metadata={
                "gpts_message": gpts_message.to_dict(),
                "conv_uid": self.not_null_agent_context.conv_id,
            },
        ):
            await self.memory.gpts_memory.append_message(
                self.not_null_agent_context.conv_id, gpts_message
            )
            return True

    def _print_received_message(self, message: AgentMessage, sender: Agent):
        # 打印收到的消息
        print("\n", "-" * 80, flush=True, sep="")
        _print_name = self.name if self.name else self.role
        print(
            colored(
                sender.name if sender.name else sender.role,
                "yellow",
            ),
            "(to",
            f"{_print_name})-[{message.model_name or ''}]:\n",
            flush=True,
        )

        content = json.dumps(message.content, ensure_ascii=False)
        if content is not None:
            print(content, flush=True)

        review_info = message.review_info
        if review_info:
            name = sender.name if sender.name else sender.role
            pass_msg = "Pass" if review_info.approve else "Reject"
            review_msg = f"{pass_msg}({review_info.comments})"
            approve_print = f">>>>>>>>{name} Review info: \n{review_msg}"
            print(colored(approve_print, "green"), flush=True)

        action_report = message.action_report
        if action_report:
            name = sender.name if sender.name else sender.role
            action_msg = (
                "execution succeeded"
                if action_report.is_exe_success
                else "execution failed"
            )
            action_report_msg = f"{action_msg},\n{action_report.content}"
            action_print = f">>>>>>>>{name} Action report: \n{action_report_msg}"
            print(colored(action_print, "blue"), flush=True)

        print("\n", "-" * 80, flush=True, sep="")

    async def _a_process_received_message(self, message: AgentMessage, sender: Agent):
        valid = await self._a_append_message(message, None, sender)
        if not valid:
            raise ValueError(
                "Received message can't be converted into a valid ChatCompletion"
                " message. Either content or function_call must be provided."
            )

        self._print_received_message(message, sender)

    async def load_resource(self, question: str, is_retry_chat: bool = False):
        """加载智能体绑定的资源。"""
        if self.resource:
            resource_prompt, resource_reference = await self.resource.get_prompt(
                lang=self.language, question=question
            )
            return resource_prompt, resource_reference
        return None, None

    async def generate_resource_variables(
        self, resource_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """生成资源变量。"""
        out_schema: Optional[str] = ""
        if self.actions and len(self.actions) > 0:
            out_schema = self.actions[0].ai_out_schema
        if not resource_prompt:
            resource_prompt = ""
        now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return {
            "resource_prompt": resource_prompt,
            "out_schema": out_schema,
            "now_time": now_time,
        }

    def _excluded_models(
        self,
        all_models: List[str],
        order_llms: Optional[List[str]] = None,
        excluded_models: Optional[List[str]] = None,
    ):
        if not order_llms:
            order_llms = []
        if not excluded_models:
            excluded_models = []
        can_uses = []
        if order_llms and len(order_llms) > 0:
            for llm_name in order_llms:
                if llm_name in all_models and (
                    not excluded_models or llm_name not in excluded_models
                ):
                    can_uses.append(llm_name)
        else:
            for llm_name in all_models:
                if not excluded_models or llm_name not in excluded_models:
                    can_uses.append(llm_name)

        return can_uses

    def convert_to_agent_message(
        self,
        gpts_messages: List[GptsMessage],
        is_rery_chat: bool = False,
    ) -> Optional[List[AgentMessage]]:
        """将 GPT 消息转换为智能体消息。"""
        oai_messages: List[AgentMessage] = []
        # 以当前智能体为基准，所有收到的消息均为用户消息，
        # 所有发出的消息均为助手消息。
        if not gpts_messages:
            return None
        for item in gpts_messages:
            # 转换消息时优先转换执行结果，
            # 否则仅使用模型输出结果。
            content = item.content
            oai_messages.append(
                AgentMessage(
                    content=content,
                    context=(
                        json.loads(item.context) if item.context is not None else None
                    ),
                    action_report=(
                        ActionOutput.from_dict(json.loads(item.action_report))
                        if item.action_report
                        else None
                    ),
                    name=item.sender,
                    rounds=item.rounds,
                    model_name=item.model_name,
                    success=item.is_success,
                )
            )
        return oai_messages

    async def _a_select_llm_model(
        self, excluded_models: Optional[List[str]] = None
    ) -> str:
        logger.info(f"_a_select_llm_model:{excluded_models}")
        try:
            all_models = await self.not_null_llm_client.models()
            all_model_names = [item.model for item in all_models]
            # TODO 当前仅实现优先级和默认这两种策略。
            if self.not_null_llm_config.llm_strategy == LLMStrategyType.Priority:
                priority: List[str] = []
                strategy_context = self.not_null_llm_config.strategy_context
                if strategy_context is not None:
                    priority = json.loads(strategy_context)  # type: ignore
                can_uses = self._excluded_models(
                    all_model_names, priority, excluded_models
                )
            else:
                can_uses = self._excluded_models(all_model_names, None, excluded_models)
            if can_uses and len(can_uses) > 0:
                return can_uses[0]
            else:
                return "deepseek-chat"
        except Exception as e:
            logger.error(f"{self.role} get next llm failed!{str(e)}")
            raise ValueError(f"Failed to allocate model service,{str(e)}!")

    def _init_reply_message(
        self,
        received_message: AgentMessage,
        rely_messages: Optional[List[AgentMessage]] = None,
    ) -> AgentMessage:
        """根据收到的消息创建一条新消息。

        根据收到的消息初始化一条新消息。

        参数：
            received_message(AgentMessage): 收到的消息

        返回：
            AgentMessage: 新消息
        """
        return AgentMessage(
            content=received_message.content,
            current_goal=received_message.current_goal,
            context=received_message.context,
            rounds=received_message.rounds + 1,
        )

    async def _a_init_reply_message(
        self,
        received_message: AgentMessage,
        rely_messages: Optional[List[AgentMessage]] = None,
    ) -> Optional[AgentMessage]:
        """根据收到的消息创建一条新消息。

        如果返回值不为 None，则不会调用 `_init_reply_message` 方法。
        """
        return None

    def _convert_to_ai_message(
        self,
        gpts_messages: List[GptsMessage],
        is_rery_chat: bool = False,
    ) -> List[AgentMessage]:
        oai_messages: List[AgentMessage] = []
        # 以当前智能体为基准，所有收到的消息均为用户消息，
        # 所有发出的消息均为助手消息。
        for item in gpts_messages:
            if item.role:
                role = item.role
            else:
                if item.receiver == self.role:
                    role = ModelMessageRoleType.HUMAN
                elif item.sender == self.role:
                    role = ModelMessageRoleType.AI
                else:
                    continue

            # 转换消息时优先转换执行结果，
            # 否则仅使用模型输出结果。
            content = item.content
            if item.action_report:
                action_out = ActionOutput.from_dict(json.loads(item.action_report))
                if is_rery_chat:
                    if action_out is not None and action_out.content:
                        content = action_out.content
                else:
                    if (
                        action_out is not None
                        and action_out.is_exe_success
                        and action_out.content is not None
                    ):
                        content = action_out.content
            oai_messages.append(
                AgentMessage(
                    content=content,
                    role=role,
                    context=(
                        json.loads(item.context) if item.context is not None else None
                    ),
                )
            )
        return oai_messages

    async def build_system_prompt(
        self,
        question: Optional[str] = None,
        most_recent_memories: Optional[str] = None,
        resource_vars: Optional[Dict] = None,
        context: Optional[Dict[str, Any]] = None,
        is_retry_chat: bool = False,
    ):
        """构建系统提示词。"""
        system_prompt = None
        if self.bind_prompt:

            class _SafeDict(dict):
                def __missing__(self, key):
                    return ""

            prompt_param = {}
            if resource_vars:
                prompt_param.update(resource_vars)
            if context:
                prompt_param.update(context)
            if self.bind_prompt.template_format == "f-string":
                system_prompt = self.bind_prompt.format(**prompt_param)
            elif self.bind_prompt.template_format == "jinja2":
                # 在沙箱中渲染：bind_prompt.template 可能包含用户可控内容
                # （如所选技能的指令），直接使用 jinja2.Template 会导致
                # SSTI，进而引发 RCE。
                _env = SandboxedEnvironment()
                system_prompt = _env.from_string(self.bind_prompt.template).render(
                    prompt_param
                )
            else:
                logger.warning("Bind prompt template not exsit or  format not support!")
        if not system_prompt:
            param: Dict = context if context else {}
            system_prompt = await self.build_prompt(
                question=question,
                is_system=True,
                most_recent_memories=most_recent_memories,
                resource_vars=resource_vars,
                is_retry_chat=is_retry_chat,
                **param,
            )
        return system_prompt

    async def _load_thinking_messages(
        self,
        received_message: AgentMessage,
        sender: Agent,
        observation: Optional[str] = None,
        rely_messages: Optional[List[AgentMessage]] = None,
        historical_dialogues: Optional[List[AgentMessage]] = None,
        context: Optional[Dict[str, Any]] = None,
        is_retry_chat: bool = False,
        current_retry_counter: Optional[int] = None,
    ) -> Tuple[List[AgentMessage], Optional[Dict]]:
        question = received_message.content
        observation = observation or question
        if not question:
            raise ValueError("The received message content is empty!")
        most_recent_memories = ""
        memory_list = []
        # 根据当前观察读取记忆
        memories = await self.read_memories(observation)
        if isinstance(memories, list):
            memory_list = memories
        else:
            most_recent_memories = memories
        has_memories = True if memories else False
        reply_message_str = ""
        if context is None:
            context = {}
        # 注入任务进度摘要，使 LLM 始终了解已完成的工作，
        # 不受缓冲区中被逐出的记忆片段数量影响。
        task_progress = self.task_progress_summary
        if task_progress:
            context["task_progress"] = task_progress
        if rely_messages:
            copied_rely_messages = [m.copy() for m in rely_messages]
            # 直接依赖历史消息时，将执行结果
            # 内容作为依赖
            for message in copied_rely_messages:
                action_report: Optional[ActionOutput] = message.action_report
                if action_report:
                    # TODO：此处为原地修改，需要优化
                    message.content = action_report.content
                if message.name != self.role:
                    # TODO：使用名称
                    # 依赖消息并非来自当前智能体
                    if message.role == ModelMessageRoleType.HUMAN:
                        reply_message_str += f"Question: {message.content}\n"
                    elif message.role == ModelMessageRoleType.AI:
                        reply_message_str += f"Observation: {message.content}\n"
        if reply_message_str:
            most_recent_memories += "\n" + reply_message_str
        try:
            # 根据当前观察加载资源提示词
            resource_prompt_str, resource_references = await self.load_resource(
                observation, is_retry_chat=is_retry_chat
            )
        except Exception as e:
            logger.exception(f"Load resource error！{str(e)}")
            raise ValueError(f"Load resource error！{str(e)}")

        resource_vars = await self.generate_resource_variables(resource_prompt_str)

        system_prompt = await self.build_system_prompt(
            question=question,
            most_recent_memories=most_recent_memories,
            resource_vars=resource_vars,
            context=context,
            is_retry_chat=is_retry_chat,
        )
        user_prompt = await self.build_prompt(
            question=question,
            is_system=False,
            most_recent_memories=most_recent_memories,
            resource_vars=resource_vars,
            **context,
        )

        agent_messages = []
        if system_prompt:
            agent_messages.append(
                AgentMessage(
                    content=system_prompt,
                    role=ModelMessageRoleType.SYSTEM,
                )
            )
        if historical_dialogues and not has_memories:
            # 如果无法读取记忆，则需要依赖历史对话
            for i in range(len(historical_dialogues)):
                if i % 2 == 0:
                    # 从偶数编号开始，偶数编号为
                    # 用户信息
                    message = historical_dialogues[i]
                    message.role = ModelMessageRoleType.HUMAN
                    agent_messages.append(message)
                else:
                    # 奇数编号为 AI 信息
                    message = historical_dialogues[i]
                    message.role = ModelMessageRoleType.AI
                    agent_messages.append(message)

        if memory_list:
            agent_messages.extend(memory_list)

        # 多层上下文管理：超出预算时进行压缩
        ctx_mgr: Optional[ContextManager] = getattr(self, "_context_manager", None)
        if ctx_mgr is not None:
            agent_messages = await ctx_mgr.manage_context(
                messages=agent_messages,
                current_round=current_retry_counter or 0,
                task_progress=task_progress,
            )

        # 当前用户输入信息
        if not user_prompt and (not memory_list or not current_retry_counter):
            # 用户提示词为空，且当前重试次数为 0 或
            # 记忆为空
            user_prompt = f"Observation: {observation}"
        if user_prompt:
            agent_messages.append(
                AgentMessage(
                    content=user_prompt,
                    role=ModelMessageRoleType.HUMAN,
                )
            )
        return agent_messages, resource_references


def _new_system_message(content):
    """返回系统消息。"""
    return [{"content": content, "role": ModelMessageRoleType.SYSTEM}]


def _is_list_of_type(lst: List[Any], type_cls: type) -> bool:
    return all(isinstance(item, type_cls) for item in lst)
