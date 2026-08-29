"""大语言模型（LLM）接口。"""

import collections
import copy
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Coroutine,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from cachetools import TTLCache

from dbgpt._private.pydantic import BaseModel, model_to_dict
from dbgpt.core.interface.media import MediaContent, MediaContentType, MediaObject
from dbgpt.core.interface.message import ModelMessage, ModelMessageRoleType
from dbgpt.util import BaseParameters
from dbgpt.util.annotations import PublicAPI
from dbgpt.util.model_utils import GPUInfo

logger = logging.getLogger(__name__)


@dataclass
@PublicAPI(stability="beta")
class ModelInferenceMetrics:
    """用于评估大语言模型推理性能的指标类。"""

    collect_index: Optional[int] = 0

    start_time_ms: Optional[int] = None
    """模型推理开始时的时间戳（毫秒）。"""

    end_time_ms: Optional[int] = None
    """模型推理结束时的时间戳（毫秒）。"""

    current_time_ms: Optional[int] = None
    """模型推理返回部分输出（流式输出）时的当前时间戳（毫秒）。"""

    first_token_time_ms: Optional[int] = None
    """生成首个 token 时的时间戳（毫秒）。"""

    first_completion_time_ms: Optional[int] = None
    """生成首个补全结果时的时间戳（毫秒）。"""

    first_completion_tokens: Optional[int] = None
    """生成首个补全结果时的 token 数量。"""

    prompt_tokens: Optional[int] = None
    """输入提示词（prompt）中的 token 数量。"""

    completion_tokens: Optional[int] = None
    """生成补全结果中的 token 数量。"""

    total_tokens: Optional[int] = None
    """token 总数（提示词加补全结果）。"""

    speed_per_second: Optional[float] = None
    """平均每秒生成的 token 数，包括预填充（prefill）和解码（decode）时间。"""

    prefill_tokens_per_second: Optional[float] = None
    """预填充阶段每秒生成的 token 数。"""

    decode_tokens_per_second: Optional[float] = None
    """解码阶段平均每秒生成的 token 数。"""

    current_gpu_infos: Optional[List[GPUInfo]] = None
    """当前所有设备的 GPU 信息。"""

    avg_gpu_infos: Optional[List[GPUInfo]] = None
    """所有采集点的平均显存使用量。"""

    @staticmethod
    def create_metrics(
        last_metrics: Optional["ModelInferenceMetrics"] = None,
    ) -> "ModelInferenceMetrics":
        """创建模型推理指标。

        Args:
            last_metrics(ModelInferenceMetrics): 上一次的指标。

        Returns:
            ModelInferenceMetrics: 模型推理指标。
        """
        start_time_ms = last_metrics.start_time_ms if last_metrics else None
        first_token_time_ms = last_metrics.first_token_time_ms if last_metrics else None
        first_completion_time_ms = (
            last_metrics.first_completion_time_ms if last_metrics else None
        )
        first_completion_tokens = (
            last_metrics.first_completion_tokens if last_metrics else None
        )
        prompt_tokens = last_metrics.prompt_tokens if last_metrics else None
        completion_tokens = last_metrics.completion_tokens if last_metrics else None
        total_tokens = last_metrics.total_tokens if last_metrics else None
        speed_per_second = last_metrics.speed_per_second if last_metrics else None
        prefill_tokens_per_second = (
            last_metrics.prefill_tokens_per_second if last_metrics else None
        )
        decode_tokens_per_second = (
            last_metrics.decode_tokens_per_second if last_metrics else None
        )
        current_gpu_infos = last_metrics.current_gpu_infos if last_metrics else None
        avg_gpu_infos = last_metrics.avg_gpu_infos if last_metrics else None

        if not start_time_ms:
            start_time_ms = time.time_ns() // 1_000_000
        current_time_ms = time.time_ns() // 1_000_000
        end_time_ms = current_time_ms

        return ModelInferenceMetrics(
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            current_time_ms=current_time_ms,
            first_token_time_ms=first_token_time_ms,
            first_completion_time_ms=first_completion_time_ms,
            first_completion_tokens=first_completion_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            speed_per_second=speed_per_second,
            prefill_tokens_per_second=prefill_tokens_per_second,
            decode_tokens_per_second=decode_tokens_per_second,
            current_gpu_infos=current_gpu_infos,
            avg_gpu_infos=avg_gpu_infos,
        )

    def to_dict(self) -> Dict:
        """将模型推理指标转换为字典。"""
        return asdict(self)

    def to_printable_string(self) -> str:
        """将指标格式化为易读的字符串。

        Returns:
            str: 包含首 token 延迟、预填充速度、解码速度、提示词 token 数和补全 token 数的格式化字符串。
        """
        lines = []

        # 如果可行，计算首 token 延迟
        first_token_latency = None
        if self.first_token_time_ms is not None and self.start_time_ms is not None:
            first_token_latency = (
                self.first_token_time_ms - self.start_time_ms
            ) / 1000.0

        # 添加分节标题
        lines.append("=== Model Inference Metrics ===")

        # 延迟指标
        lines.append("\n▶ Latency:")
        if first_token_latency is not None:
            lines.append(f"  • First Token Latency: {first_token_latency:.3f}s")
        else:
            lines.append("  • First Token Latency: N/A")

        # 速度指标
        lines.append("\n▶ Speed:")
        if self.prefill_tokens_per_second is not None:
            lines.append(
                f"  • Prefill Speed: {self.prefill_tokens_per_second:.2f} tokens/s"
            )
        else:
            lines.append("  • Prefill Speed: N/A")

        if self.decode_tokens_per_second is not None:
            lines.append(
                f"  • Decode Speed: {self.decode_tokens_per_second:.2f} tokens/s"
            )
        else:
            lines.append("  • Decode Speed: N/A")

        # token 数量
        lines.append("\n▶ Tokens:")
        if self.prompt_tokens is not None:
            lines.append(f"  • Prompt Tokens: {self.prompt_tokens}")
        else:
            lines.append("  • Prompt Tokens: N/A")

        if self.completion_tokens is not None:
            lines.append(f"  • Completion Tokens: {self.completion_tokens}")
        else:
            lines.append("  • Completion Tokens: N/A")

        if self.total_tokens is not None:
            lines.append(f"  • Total Tokens: {self.total_tokens}")

        return "\n".join(lines)


@dataclass
@PublicAPI(stability="beta")
class ModelRequestContext:
    """表示大语言模型请求上下文的类。"""

    stream: bool = False
    """是否以流式方式返回响应。"""

    cache_enable: bool = False
    """是否为模型推理启用缓存。"""

    user_name: Optional[str] = None
    """模型请求的用户名。"""

    sys_code: Optional[str] = None
    """模型请求的系统代码。"""

    conv_uid: Optional[str] = None
    """模型推理的会话 ID。"""

    span_id: Optional[str] = None
    """模型推理的跨度 ID。"""

    chat_mode: Optional[str] = None
    """模型推理的聊天模式。"""

    chat_param: Optional[str] = None
    """聊天模式的参数。"""

    extra: Optional[Dict[str, Any]] = field(default_factory=dict)
    """模型推理的额外信息。"""

    request_id: Optional[str] = None
    """模型推理的请求 ID。"""

    is_reasoning_model: Optional[bool] = False
    """模型是否为推理模型。"""


@dataclass
@PublicAPI(stability="beta")
class ModelOutput:
    """表示大语言模型输出的类。"""

    content: Union[MediaContent, List[MediaContent]]
    """生成的文本。"""
    error_code: int
    """模型推理的错误码。模型推理成功时错误码为 0。"""
    incremental: bool = False
    model_context: Optional[Dict] = None
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    metrics: Optional[ModelInferenceMetrics] = None
    """模型推理的一些指标。"""

    def __init__(
        self,
        error_code: int,
        text: Optional[str] = None,
        content: Optional[
            Union[
                MediaContent, List[MediaContent], Dict[str, Any], List[Dict[str, Any]]
            ]
        ] = None,
        **kwargs,
    ):
        if text is not None and content is not None:
            raise ValueError("Cannot pass both text and content")
        elif text is not None:
            self.content = MediaContent.build_text(text)
        elif content is not None:
            self.content = MediaContent.parse_content(content)
        else:
            raise ValueError("Must pass either text or content")
        self.error_code = error_code
        for k, v in kwargs.items():
            if k in [
                "incremental",
                "model_context",
                "finish_reason",
                "usage",
                "metrics",
            ]:
                setattr(self, k, v)

    def to_dict(self) -> Dict:
        """将模型输出转换为字典。"""
        text = self.gen_text_with_thinking()
        return {
            "error_code": self.error_code,
            "text": text,
            "incremental": self.incremental,
            "model_context": self.model_context,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "metrics": self.metrics,
        }

    @property
    def success(self) -> bool:
        """检查模型推理是否成功。"""
        return self.error_code == 0

    @property
    def has_text(self) -> bool:
        """检查模型输出是否包含文本内容。"""
        if isinstance(self.content, MediaContent):
            return self.content.type == MediaContentType.TEXT
        elif isinstance(self.content, list):
            return any(c.type == MediaContentType.TEXT for c in self.content)
        return False

    @property
    def text(self) -> str:
        """生成的文本。"""
        if isinstance(self.content, MediaContent):
            return self.content.get_text()
        elif isinstance(self.content, list) and all(
            isinstance(c, MediaContent) for c in self.content
        ):
            return MediaContent.last_text(self.content)
        raise ValueError("The content is not text")

    @property
    def has_thinking(self) -> bool:
        """检查模型输出是否包含思考内容。"""
        if isinstance(self.content, MediaContent):
            return self.content.type == MediaContentType.THINKING
        elif isinstance(self.content, list) and self.content:
            return any(c.type == MediaContentType.THINKING for c in self.content)
        else:
            return False

    @property
    def thinking_text(self) -> Optional[str]:
        """推理内容。"""
        if not self.content:
            return None
        if isinstance(self.content, MediaContent):
            if self.content.type == MediaContentType.THINKING:
                return self.content.get_thinking()
            return None
        elif isinstance(self.content, list) and all(
            isinstance(c, MediaContent) for c in self.content
        ):
            # 在大多数情况下只有一段思考内容
            thinking_content = [
                c for c in self.content if c.type == MediaContentType.THINKING
            ]
            if not thinking_content:
                return None
            # 返回最后一段思考内容
            return thinking_content[-1].get_thinking()
        return None

    def gen_text_with_thinking(self, new_text: Optional[str] = None) -> str:
        from dbgpt.vis.tags.vis_thinking import VisThinking

        msg = ""
        if self.has_thinking:
            msg = self.thinking_text or ""
            msg = VisThinking().sync_display(content=msg)
            msg += "\n"
        if new_text:
            msg += new_text
        elif self.has_text:
            msg += self.text or ""
        return msg

    @text.setter
    def text(self, value: str):
        """设置生成的文本。"""
        if not isinstance(value, str):
            raise ValueError("text must be a string")
        # 构建新的 MediaContent 对象并将其赋给 content
        self.content = MediaContent(
            type="text",
            object=MediaObject(data=value, format="text"),
        )

    @classmethod
    def build_thinking(cls, thinking: str, error_code: int = 0) -> "ModelOutput":
        """根据思考内容创建 ModelOutput 对象。"""
        return cls(
            error_code=error_code,
            content=MediaContent.build_thinking(thinking),
        )

    @classmethod
    def build(
        cls,
        text: Optional[str] = None,
        thinking: Optional[str] = None,
        error_code: int = 0,
        usage: Optional[Dict[str, Any]] = None,
        finish_reason: Optional[str] = None,
        is_reasoning_model: bool = False,
        metrics: Optional[ModelInferenceMetrics] = None,
    ) -> "ModelOutput":
        if thinking and text:
            # 同时包含思考内容和文本
            content = [
                # 首先放入思考内容
                MediaContent.build_thinking(thinking),
                MediaContent.build_text(text),
            ]
        elif text:
            # 仅包含文本
            content = MediaContent.build_text(text)
        elif is_reasoning_model or thinking:
            # 构建空的思考内容
            # 处理空数据
            content = MediaContent.build_thinking(thinking)
        else:
            content = MediaContent.build_text("")

        return cls(
            error_code=error_code,
            content=content,
            usage=usage,
            finish_reason=finish_reason,
            metrics=metrics,
        )

    @property
    def error_message(self) -> str:
        """获取错误消息。
        当 error_code 不为 0 时返回错误消息。
        """
        return self.text if self.has_text else "Unknown error"


_ModelMessageType = Union[List[ModelMessage], List[Dict[str, Any]]]


@dataclass
@PublicAPI(stability="beta")
class ModelRequest:
    """模型请求。"""

    model: str
    """模型名称。"""

    messages: _ModelMessageType
    """输入消息。"""

    temperature: Optional[float] = None
    """模型推理的温度参数。"""

    top_p: Optional[float] = None
    """模型推理的 Top-p 参数。"""

    max_new_tokens: Optional[int] = None
    """最多生成的 token 数量。"""

    stop: Optional[Union[str, List[str]]] = None
    """模型推理的停止条件。"""
    stop_token_ids: Optional[List[int]] = None
    """模型推理的停止 token ID。"""
    context_len: Optional[int] = None
    """模型推理的上下文长度。"""
    echo: Optional[bool] = False
    """是否回显输入消息。"""
    span_id: Optional[str] = None
    """模型推理的跨度 ID。"""

    context: Optional[ModelRequestContext] = field(
        default_factory=lambda: ModelRequestContext()
    )
    """模型推理的上下文。"""

    @property
    def stream(self) -> bool:
        """是否以流式方式返回响应。"""
        return bool(self.context and self.context.stream)

    def copy(self) -> "ModelRequest":
        """复制模型请求。

        Returns:
            ModelRequest: 复制后的模型请求。
        """
        new_request = copy.deepcopy(self)
        # 将消息转换为 List[ModelMessage]
        new_request.messages = new_request.get_messages()
        return new_request

    def to_dict(self) -> Dict[str, Any]:
        """将模型请求转换为字典。

        Returns:
            Dict[str, Any]: 字典形式的模型请求。
        """
        new_reqeust = copy.deepcopy(self)
        new_messages = []
        for message in new_reqeust.messages:
            if isinstance(message, dict):
                new_messages.append(message)
            else:
                new_messages.append(message.dict())
        new_reqeust.messages = new_messages
        # 跳过值为 None 的字段
        return {k: v for k, v in asdict(new_reqeust).items() if v is not None}

    def to_trace_metadata(self) -> Dict[str, Any]:
        """将模型请求转换为追踪元数据。

        Returns:
            Dict[str, Any]: 追踪元数据。
        """
        metadata = self.to_dict()
        metadata["prompt"] = self.messages_to_string()
        return metadata

    def get_messages(self) -> List[ModelMessage]:
        """获取消息。

        如果消息不是 ModelMessage 列表，则会转换为 ModelMessage 列表。

        Returns:
            List[ModelMessage]: 消息列表。
        """
        messages = []
        for message in self.messages:
            if isinstance(message, dict):
                messages.append(ModelMessage(**message))
            else:
                messages.append(message)
        return messages

    def get_single_user_message(self) -> Optional[ModelMessage]:
        """获取单条用户消息。

        Returns:
            Optional[ModelMessage]: 单条用户消息。
        """
        messages = self.get_messages()
        if len(messages) != 1 and messages[0].role != ModelMessageRoleType.HUMAN:
            raise ValueError("The messages is not a single user message")
        return messages[0]

    @staticmethod
    def build_request(
        model: str,
        messages: List[ModelMessage],
        context: Optional[Union[ModelRequestContext, Dict[str, Any], BaseModel]] = None,
        stream: bool = False,
        echo: bool = False,
        **kwargs,
    ):
        """构建模型请求。

        Args:
            model(str): 模型名称。
            messages(List[ModelMessage]): 消息列表。
            context(Optional[Union[ModelRequestContext, Dict[str, Any], BaseModel]]):
                请求上下文。
            stream(bool): 是否以流式方式返回响应。默认为 False。
            echo(bool): 是否回显输入消息。默认为 False。
            **kwargs: 其他参数。
        """
        if not context:
            context = ModelRequestContext(stream=stream)
        elif not isinstance(context, ModelRequestContext):
            context_dict = None
            if isinstance(context, dict):
                context_dict = context
            elif isinstance(context, BaseModel):
                context_dict = model_to_dict(context)
            if context_dict and "stream" not in context_dict:
                context_dict["stream"] = stream
            if context_dict:
                context = ModelRequestContext(**context_dict)
            else:
                context = ModelRequestContext(stream=stream)
        return ModelRequest(
            model=model,
            messages=messages,
            context=context,
            echo=echo,
            **kwargs,
        )

    @staticmethod
    def _build(model: str, prompt: str, **kwargs):
        return ModelRequest(
            model=model,
            messages=[ModelMessage(role=ModelMessageRoleType.HUMAN, content=prompt)],
            **kwargs,
        )

    def to_common_messages(
        self, support_system_role: bool = True
    ) -> List[Dict[str, Any]]:
        """将消息转换为通用格式（如 OpenAI API）。

        此函数会将最后一条用户消息移动到列表末尾。

        Args:
            support_system_role (bool): 是否支持 system 角色。

        Returns:
            List[Dict[str, Any]]: OpenAI API 格式的消息。

        Raises:
            ValueError: 消息角色不受支持时抛出。

        Examples:
            .. code-block:: python

                from dbgpt.core.interface.message import (
                    ModelMessage,
                    ModelMessageRoleType,
                )

                messages = [
                    ModelMessage(role=ModelMessageRoleType.HUMAN, content="Hi"),
                    ModelMessage(
                        role=ModelMessageRoleType.AI, content="Hi, I'm a robot."
                    ),
                    ModelMessage(
                        role=ModelMessageRoleType.HUMAN, content="Who are your"
                    ),
                ]
                openai_messages = ModelRequest.to_openai_messages(messages)
                assert openai_messages == [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hi, I'm a robot."},
                    {"role": "user", "content": "Who are your"},
                ]
        """
        messages = [
            m if isinstance(m, ModelMessage) else ModelMessage(**m)
            for m in self.messages
        ]
        return ModelMessage.to_common_messages(
            messages, support_system_role=support_system_role
        )

    def messages_to_string(self) -> str:
        """将消息转换为字符串。

        Returns:
            str: 字符串格式的消息。
        """
        return ModelMessage.messages_to_string(self.get_messages())

    def split_messages(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        """拆分消息。

        Returns:
            Tuple[List[Dict[str, Any]], List[str]]: 通用消息和系统消息。
        """
        messages = self.get_messages()
        common_messages = []
        system_messages = []
        for message in messages:
            if message.role == ModelMessageRoleType.HUMAN:
                common_messages.append({"role": "user", "content": message.content})
            elif message.role == ModelMessageRoleType.SYSTEM:
                system_messages.append(message.content)
            elif message.role == ModelMessageRoleType.AI:
                common_messages.append(
                    {"role": "assistant", "content": message.content}
                )
            else:
                pass
        return common_messages, system_messages


@dataclass
class ModelExtraMedata(BaseParameters):
    """表示大语言模型额外元数据的类。"""

    prompt_roles: List[str] = field(
        default_factory=lambda: [
            ModelMessageRoleType.SYSTEM,
            ModelMessageRoleType.HUMAN,
            ModelMessageRoleType.AI,
        ],
        metadata={"help": "The roles of the prompt"},  # 提示词的角色
    )

    prompt_sep: Optional[str] = field(
        default="\n",
        metadata={"help": "The separator of the prompt between multiple rounds"},  # 多轮提示词之间的分隔符
    )

    # 可在模型仓库的 tokenizer 配置中查看聊天模板，通常位于 tokenizer_config.json
    prompt_chat_template: Optional[str] = field(
        default=None,
        metadata={
            # 聊天模板，参见 Hugging Face 文档
            "help": "The chat template, see: "
            "https://huggingface.co/docs/transformers/main/en/chat_templating"
        },
    )

    @property
    def support_system_message(self) -> bool:
        """模型是否支持 system 消息。

        Returns:
            bool: 模型是否支持 system 消息。
        """
        return ModelMessageRoleType.SYSTEM in self.prompt_roles


@dataclass
@PublicAPI(stability="beta")
class ModelMetadata(BaseParameters):
    """表示大语言模型的类。"""

    model: Union[str, List[str]] = field(
        metadata={"help": "Model name"},  # 模型名称
    )
    label: Optional[str] = field(
        default=None,
        metadata={"help": "Model label"},  # 模型标签
    )
    context_length: Optional[int] = field(
        default=None,
        metadata={"help": "Context length of model"},  # 模型上下文长度
    )
    max_output_length: Optional[int] = field(
        default=None,
        metadata={"help": "Max output length of model"},  # 模型最大输出长度
    )
    description: Optional[str] = field(
        default=None,
        metadata={"help": "Model description"},  # 模型描述
    )
    link: Optional[str] = field(
        default=None,
        metadata={"help": "Model link"},  # 模型链接
    )
    chat_model: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether the model is a chat model"},  # 是否为聊天模型
    )
    function_calling: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether the model is a function calling model"},  # 是否为函数调用模型
    )
    metadata: Optional[Dict[str, Any]] = field(
        default_factory=dict,
        metadata={"help": "Model metadata"},  # 模型元数据
    )
    ext_metadata: Optional[ModelExtraMedata] = field(
        default_factory=ModelExtraMedata,
        metadata={"help": "Model extra metadata"},  # 模型额外元数据
    )

    @classmethod
    def from_dict(
        cls, data: dict, ignore_extra_fields: bool = False
    ) -> "ModelMetadata":
        """根据字典创建新的模型元数据。"""
        if "ext_metadata" in data:
            data["ext_metadata"] = ModelExtraMedata(**data["ext_metadata"])
        return cls(**data)


class MessageConverter(ABC):
    r"""消息转换器的抽象基类。

    不同大语言模型的消息格式可能不同，此类用于将消息转换为对应模型的格式。

    Examples:
        >>> from typing import List
        >>> from dbgpt.core.interface.message import ModelMessage, ModelMessageRoleType
        >>> from dbgpt.core.interface.llm import MessageConverter, ModelMetadata
        >>> class RemoveSystemMessageConverter(MessageConverter):
        ...     def convert(
        ...         self,
        ...         messages: List[ModelMessage],
        ...         model_metadata: Optional[ModelMetadata] = None,
        ...     ) -> List[ModelMessage]:
        ...         # 转换消息，将 system 消息合并到最后一条用户消息
        ...         system_message = None
        ...         other_messages = []
        ...         sep = "\\n"
        ...         for message in messages:
        ...             if message.role == ModelMessageRoleType.SYSTEM:
        ...                 system_message = message
        ...             else:
        ...                 other_messages.append(message)
        ...         if system_message and other_messages:
        ...             other_messages[-1].content = (
        ...                 system_message.content + sep + other_messages[-1].content
        ...             )
        ...         return other_messages
        >>> messages = [
        ...     ModelMessage(
        ...         role=ModelMessageRoleType.SYSTEM,
        ...         content="You are a helpful assistant",
        ...     ),
        ...     ModelMessage(role=ModelMessageRoleType.HUMAN, content="Who are you"),
        ... ]
        >>> converter = RemoveSystemMessageConverter()
        >>> converted_messages = converter.convert(messages, None)
        >>> assert converted_messages == [
        ...     ModelMessage(
        ...         role=ModelMessageRoleType.HUMAN,
        ...         content="You are a helpful assistant\\nWho are you",
        ...     ),
        ... ]
    """

    @abstractmethod
    def convert(
        self,
        messages: List[ModelMessage],
        model_metadata: Optional[ModelMetadata] = None,
    ) -> List[ModelMessage]:
        """转换消息。

        Args:
            messages(List[ModelMessage]): 消息列表。
            model_metadata(ModelMetadata): 模型元数据。

        Returns:
            List[ModelMessage]: 转换后的消息列表。
        """


class DefaultMessageConverter(MessageConverter):
    """默认消息转换器。"""

    def __init__(self, prompt_sep: Optional[str] = None):
        """创建默认消息转换器。"""
        self._prompt_sep = prompt_sep

    def convert(
        self,
        messages: List[ModelMessage],
        model_metadata: Optional[ModelMetadata] = None,
    ) -> List[ModelMessage]:
        """转换消息。

        消息转换分为三个步骤：

        1. 仅保留 system、human 和 AI 消息。

        2. 将最后一条用户消息移动到列表末尾。

        3. 如果模型不支持 system 消息，则转换为不含 system 消息的格式。

        Args:
            messages(List[ModelMessage]): 消息列表。
            model_metadata(ModelMetadata): 模型元数据。

        Returns:
            List[ModelMessage]: 转换后的消息列表。
        """
        # 1. 仅保留 system、human 和 AI 消息
        messages = list(filter(lambda m: m.pass_to_model, messages))
        # 2. 将最后一条用户消息移动到列表末尾
        messages = self.move_last_user_message_to_end(messages)

        if not model_metadata or not model_metadata.ext_metadata:
            logger.warning("No model metadata, skip message system message conversion")
            return messages
        if not model_metadata.ext_metadata.support_system_message:
            # 3. 转换为不含 system 消息的格式
            return self.convert_to_no_system_message(messages, model_metadata)
        return messages

    def convert_to_no_system_message(
        self,
        messages: List[ModelMessage],
        model_metadata: Optional[ModelMetadata] = None,
    ) -> List[ModelMessage]:
        r"""将消息转换为不含 system 消息的格式。

        Examples:
            >>> # 转换为不含 system 消息的格式，将 system 消息合并到最后一条用户消息
            >>> from typing import List
            >>> from dbgpt.core.interface.message import (
            ...     ModelMessage,
            ...     ModelMessageRoleType,
            ... )
            >>> from dbgpt.core.interface.llm import (
            ...     DefaultMessageConverter,
            ...     ModelMetadata,
            ... )
            >>> messages = [
            ...     ModelMessage(
            ...         role=ModelMessageRoleType.SYSTEM,
            ...         content="You are a helpful assistant",
            ...     ),
            ...     ModelMessage(
            ...         role=ModelMessageRoleType.HUMAN, content="Who are you"
            ...     ),
            ... ]
            >>> converter = DefaultMessageConverter()
            >>> model_metadata = ModelMetadata(model="test")
            >>> converted_messages = converter.convert_to_no_system_message(
            ...     messages, model_metadata
            ... )
            >>> assert converted_messages == [
            ...     ModelMessage(
            ...         role=ModelMessageRoleType.HUMAN,
            ...         content="You are a helpful assistant\nWho are you",
            ...     ),
            ... ]
        """
        if not model_metadata or not model_metadata.ext_metadata:
            logger.warning("No model metadata, skip message conversion")
            return messages
        ext_metadata = model_metadata.ext_metadata
        system_messages = []
        result_messages = []
        for message in messages:
            if message.role == ModelMessageRoleType.SYSTEM:
                # 不支持 system 消息，将其追加到最后一条用户消息
                system_messages.append(message)
            elif message.role in [
                ModelMessageRoleType.HUMAN,
                ModelMessageRoleType.AI,
            ]:
                result_messages.append(message)
        prompt_sep = self._prompt_sep or ext_metadata.prompt_sep or "\n"
        system_message_str = None
        if len(system_messages) > 1:
            logger.warning("Your system messages have more than one message")
            system_message_str = prompt_sep.join([m.content for m in system_messages])
        elif len(system_messages) == 1:
            system_message_str = system_messages[0].content

        if system_message_str and result_messages:
            # 不支持 system 消息，将 system 消息合并到最后一条用户消息
            result_messages[-1].content = (
                system_message_str + prompt_sep + result_messages[-1].content
            )
        return result_messages

    def move_last_user_message_to_end(
        self, messages: List[ModelMessage]
    ) -> List[ModelMessage]:
        """尝试将最后一条用户消息移动到列表末尾。

        Examples:
            >>> from typing import List
            >>> from dbgpt.core.interface.message import (
            ...     ModelMessage,
            ...     ModelMessageRoleType,
            ... )
            >>> from dbgpt.core.interface.llm import DefaultMessageConverter
            >>> messages = [
            ...     ModelMessage(
            ...         role=ModelMessageRoleType.SYSTEM,
            ...         content="You are a helpful assistant",
            ...     ),
            ...     ModelMessage(
            ...         role=ModelMessageRoleType.HUMAN, content="Who are you"
            ...     ),
            ...     ModelMessage(role=ModelMessageRoleType.AI, content="I'm a robot"),
            ...     ModelMessage(
            ...         role=ModelMessageRoleType.HUMAN, content="What's your name"
            ...     ),
            ...     ModelMessage(
            ...         role=ModelMessageRoleType.SYSTEM,
            ...         content="You are a helpful assistant",
            ...     ),
            ... ]
            >>> converter = DefaultMessageConverter()
            >>> converted_messages = converter.move_last_user_message_to_end(messages)
            >>> assert converted_messages == [
            ...     ModelMessage(
            ...         role=ModelMessageRoleType.SYSTEM,
            ...         content="You are a helpful assistant",
            ...     ),
            ...     ModelMessage(
            ...         role=ModelMessageRoleType.HUMAN, content="Who are you"
            ...     ),
            ...     ModelMessage(role=ModelMessageRoleType.AI, content="I'm a robot"),
            ...     ModelMessage(
            ...         role=ModelMessageRoleType.SYSTEM,
            ...         content="You are a helpful assistant",
            ...     ),
            ...     ModelMessage(
            ...         role=ModelMessageRoleType.HUMAN, content="What's your name"
            ...     ),
            ... ]

        Args:
            messages(List[ModelMessage]): 消息列表。

        Returns:
            List[ModelMessage]: 转换后的消息列表。
        """
        last_user_input_index = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == ModelMessageRoleType.HUMAN:
                last_user_input_index = i
                break
        if last_user_input_index is not None:
            last_user_input = messages.pop(last_user_input_index)
            messages.append(last_user_input)
        return messages


@PublicAPI(stability="beta")
class LLMClient(ABC):
    """大语言模型客户端的抽象基类。"""

    # 将模型元数据缓存 60 秒
    _MODEL_CACHE_ = TTLCache(maxsize=100, ttl=60)

    @property
    def cache(self) -> collections.abc.MutableMapping:
        """返回用于缓存模型元数据的缓存对象。

        可以重写此属性以使用自定义缓存对象。
        Returns:
            collections.abc.MutableMapping: 缓存对象。
        """
        return self._MODEL_CACHE_

    @abstractmethod
    async def generate(
        self,
        request: ModelRequest,
        message_converter: Optional[MessageConverter] = None,
    ) -> ModelOutput:
        """根据给定的模型请求生成响应。

        不同大语言模型的消息格式可能不同，可以使用消息转换器将消息转换为对应模型的格式。

        Args:
            request(ModelRequest): 模型请求。
            message_converter(MessageConverter): 消息转换器。

        Returns:
            ModelOutput: 模型输出。

        """

    @abstractmethod
    async def generate_stream(
        self,
        request: ModelRequest,
        message_converter: Optional[MessageConverter] = None,
    ) -> AsyncIterator[ModelOutput]:
        """根据给定的模型请求生成响应流。

        不同大语言模型的消息格式可能不同，可以使用消息转换器将消息转换为对应模型的格式。

        Args:
            request(ModelRequest): 模型请求。
            message_converter(MessageConverter): 消息转换器。

        Returns:
            AsyncIterator[ModelOutput]: 模型输出流。
        """

    @abstractmethod
    async def models(self) -> List[ModelMetadata]:
        """获取所有模型。

        Returns:
            List[ModelMetadata]: 模型元数据列表。
        """

    @abstractmethod
    async def count_token(self, model: str, prompt: str) -> int:
        """计算给定提示词中的 token 数量。

        Args:
            model(str): 模型名称。
            prompt(str): 提示词。

        Returns:
            int: token 数量。
        """

    async def covert_message(
        self,
        request: ModelRequest,
        message_converter: Optional[MessageConverter] = None,
    ) -> ModelRequest:
        """转换消息。

        如果未提供消息转换器，将返回原始请求。

        Args:
            request(ModelRequest): 模型请求。
            message_converter(MessageConverter): 消息转换器。

        Returns:
            ModelRequest: 转换后的模型请求。
        """
        if not message_converter:
            return request
        new_request = request.copy()
        model_metadata = await self.get_model_metadata(request.model)
        new_messages = message_converter.convert(request.get_messages(), model_metadata)
        new_request.messages = new_messages
        return new_request

    async def cached_models(self) -> List[ModelMetadata]:
        """从缓存或大语言模型服务器获取所有模型。

        如果缓存中没有模型元数据，将从大语言模型服务器获取。

        Returns:
            List[ModelMetadata]: 模型元数据列表。
        """
        key = "____$llm_client_models$____"
        if key not in self.cache:
            models = await self.models()
            self.cache[key] = models
            for model in models:
                model_metadata_key = (
                    f"____$llm_client_models_metadata_{model.model}$____"
                )
                self.cache[model_metadata_key] = model
        return self.cache[key]

    async def get_model_metadata(self, model: str) -> ModelMetadata:
        """获取模型元数据。

        Args:
            model(str): 模型名称。

        Returns:
            ModelMetadata: 模型元数据。

        Raises:
            ValueError: 找不到模型时抛出。
        """
        model_metadata_key = f"____$llm_client_models_metadata_{model}$____"
        if model_metadata_key not in self.cache:
            await self.cached_models()
        model_metadata = self.cache.get(model_metadata_key)
        if not model_metadata:
            raise ValueError(f"Model {model} not found")
        return model_metadata

    def __call__(
        self, *args, **kwargs
    ) -> Coroutine[Any, Any, ModelOutput] | ModelOutput:
        """返回模型输出。

        调用大语言模型客户端，根据给定消息生成响应。

        请勿在生产环境中使用此方法，它仅用于调试。
        """
        import asyncio

        from dbgpt.util import get_or_create_event_loop

        try:
            # 检查当前是否处于事件循环中
            loop = asyncio.get_running_loop()
            # 如果处于事件循环中，使用异步调用
            if loop.is_running():
                # 当前处于异步环境，但这是同步方法，因此返回协程对象供调用方 await
                return self.async_call(*args, **kwargs)
            else:
                loop = get_or_create_event_loop()
                return loop.run_until_complete(self.async_call(*args, **kwargs))
        except RuntimeError:
            # 如果不在事件循环中，使用同步调用
            loop = get_or_create_event_loop()
            return loop.run_until_complete(self.async_call(*args, **kwargs))

    async def async_call(self, *args, **kwargs) -> ModelOutput:
        """异步返回模型输出。

        请勿在生产环境中使用此方法，它仅用于调试。
        """
        req = self._build_call_request(*args, **kwargs)
        return await self.generate(req)

    async def async_call_stream(self, *args, **kwargs) -> AsyncIterator[ModelOutput]:
        """异步返回模型输出流。

        请勿在生产环境中使用此方法，它仅用于调试。
        """
        req = self._build_call_request(*args, **kwargs)
        async for output in self.generate_stream(req):  # type: ignore
            yield output

    def _build_call_request(self, *args, **kwargs) -> ModelRequest:
        """为 call 方法构建模型请求。"""
        messages = kwargs.get("messages")
        model = kwargs.get("model")

        if messages:
            del kwargs["messages"]
            model_messages = ModelMessage.from_openai_messages(messages)
        else:
            model_messages = [ModelMessage.build_human_message(args[0])]

        if not model:
            if hasattr(self, "default_model"):
                model = getattr(self, "default_model")
            else:
                raise ValueError("The default model is not set")

        if "model" in kwargs:
            del kwargs["model"]

        return ModelRequest.build_request(model, model_messages, **kwargs)
