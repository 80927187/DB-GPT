"""用于大语言模型（LLM）的 AIWrapper 封装。"""

import asyncio
import json
import logging
import traceback
from typing import Any, Callable, Dict, Optional, Union

from dbgpt.core import LLMClient, ModelOutput, ModelRequestContext
from dbgpt.core.interface.output_parser import BaseOutputParser
from dbgpt.util.error_types import LLMChatError
from dbgpt.util.tracer import root_tracer

from ..llm.llm import _build_model_request

logger = logging.getLogger(__name__)


class AIWrapper:
    """用于大语言模型（LLM）的调用封装器。"""

    cache_path_root: str = ".cache"
    extra_kwargs = {
        "cache_seed",
        "filter_func",
        "allow_format_str_template",
        "context",
        "llm_model",
        "memory",
        "conv_id",
        "sender",
        "stream_out",
        "stream_callback",
    }

    def __init__(
        self, llm_client: LLMClient, output_parser: Optional[BaseOutputParser] = None
    ):
        """创建一个 AIWrapper 实例。"""
        self.llm_echo = False
        self.model_cache_enable = False
        self._llm_client = llm_client
        self._output_parser = output_parser or BaseOutputParser(is_stream_out=False)

    @classmethod
    def instantiate(
        cls,
        template: Optional[Union[str, Callable]] = None,
        context: Optional[Dict] = None,
        allow_format_str_template: Optional[bool] = False,
    ):
        """使用上下文实例化模板。"""
        if not context or template is None:
            return template
        if isinstance(template, str):
            return template.format(**context) if allow_format_str_template else template
        return template(context)

    def _construct_create_params(self, create_config: Dict, extra_kwargs: Dict) -> Dict:
        """使用额外参数初始化创建配置（create_config）。"""
        # 校验配置
        prompt = create_config.get("prompt")
        messages = create_config.get("messages")
        if prompt is None and messages is None:
            raise ValueError(
                # 创建配置必须提供 prompt 或 messages，且不能同时提供。
                "Either prompt or messages should be in create config but not both."
            )

        context = extra_kwargs.get("context")
        if context is None:
            # 未提供上下文时无需进行实例化。
            return create_config
        # 实例化 prompt 或 messages
        allow_format_str_template = extra_kwargs.get("allow_format_str_template", False)
        # 复制一份配置，避免修改原始对象
        params = create_config.copy()
        params["context"] = context

        if prompt is not None:
            # 实例化 prompt
            params["prompt"] = self.instantiate(
                prompt, context, allow_format_str_template
            )
        elif context and messages and isinstance(messages, list):
            # 实例化 messages
            params["messages"] = [
                (
                    {
                        **m,
                        "content": self.instantiate(
                            m["content"], context, allow_format_str_template
                        ),
                    }
                    if m.get("content")
                    else m
                )
                for m in messages
            ]
        return params

    def _separate_create_config(self, config):
        """将配置拆分为 create_config 和 extra_kwargs。"""
        create_config = {k: v for k, v in config.items() if k not in self.extra_kwargs}
        extra_kwargs = {k: v for k, v in config.items() if k in self.extra_kwargs}
        return create_config, extra_kwargs

    def _get_key(self, config):
        """获取配置的唯一标识符。

        参数:
            config (dict or list): 配置对象。

        返回:
            tuple: 可用作字典键的唯一标识符。
        """
        non_cache_key = ["api_key", "base_url", "api_type", "api_version"]
        copied = False
        for key in non_cache_key:
            if key in config:
                config, copied = config.copy() if not copied else config, True
                config.pop(key)
        return json.dumps(config, sort_keys=True, ensure_ascii=False)

    async def create(self, verbose: bool = False, **config):
        """创建大语言模型客户端请求。"""
        # 将输入配置合并为完整配置
        full_config = {**config}
        # 将配置拆分为 create_config 和 extra_kwargs
        create_config, extra_kwargs = self._separate_create_config(full_config)

        # 构造创建参数
        params = self._construct_create_params(create_config, extra_kwargs)
        # 获取 cache_seed、filter_func 和 context 等附加参数
        filter_func = extra_kwargs.get("filter_func")
        context = extra_kwargs.get("context")
        llm_model = extra_kwargs.get("llm_model")
        memory = extra_kwargs.get("memory", None)
        conv_id = extra_kwargs.get("conv_id", None)
        sender = extra_kwargs.get("sender", None)
        stream_out = extra_kwargs.get("stream_out", True)
        stream_callback = extra_kwargs.get("stream_callback")

        try:
            response = await self._completions_create(
                llm_model,
                params,
                conv_id,
                sender,
                memory,
                stream_out,
                verbose,
                stream_callback,
            )
        except LLMChatError as e:
            # 保留原始英文日志信息，便于与现有日志检索规则兼容。
            logger.debug(f"{llm_model} generate failed!{str(e)}")
            raise e
        else:
            pass_filter = filter_func is None or filter_func(
                context=context, response=response
            )
            if pass_filter:
                # 响应通过过滤器时返回该响应
                return response
            else:
                return None

    async def generate_text(
        self,
        prompt: str,
        llm_model: Optional[str] = None,
        max_new_tokens: int = 2000,
        temperature: float = 0.3,
        conv_id: Optional[str] = None,
    ) -> str:
        """根据单个 prompt 生成文本。

        这是 :meth:`create` 的轻量便捷封装，适用于上下文摘要（第三层压缩，Layer 3
        compaction）等不需要完整消息列表的简单文本生成场景。

        参数:
            prompt: 发送给模型的用户提示词。
            llm_model: 模型名称。必填；``AIWrapper`` 不保存默认模型，因此调用方（例如
                ``ContextManager``）必须传入智能体使用的模型名称。
            max_new_tokens: 要生成的最大 token 数。
            temperature: 采样温度。
            conv_id: 用于链路追踪的会话 ID。

        返回:
            生成的文本。失败时返回空字符串（调用方应处理结果为空的情况）。
        """
        if not llm_model:
            # 保留原始英文异常信息，避免影响依赖该文本的调用方。
            raise ValueError("llm_model is required for generate_text()")
        messages = [{"role": "user", "content": prompt}]
        try:
            response = await self.create(
                messages=messages,
                llm_model=llm_model,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stream_out=False,
                conv_id=conv_id,
            )
        except LLMChatError:
            # 保留原始英文日志信息，便于与现有日志检索规则兼容。
            logger.exception("generate_text: LLM call failed")
            return ""
        return response or ""

    def _get_span_metadata(self, payload: Dict) -> Dict:
        metadata = {k: v for k, v in payload.items()}

        metadata["messages"] = list(
            map(lambda m: m if isinstance(m, dict) else m.dict(), metadata["messages"])
        )
        return metadata

    def _llm_messages_convert(self, params):
        gpts_messages = params["messages"]
        # TODO：待实现消息格式转换

        return gpts_messages

    async def _completions_create(
        self,
        llm_model,
        params,
        conv_id: Optional[str] = None,
        sender: Optional[str] = None,
        memory: Optional[Any] = None,
        stream_out: bool = True,
        verbose: bool = False,
        stream_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ):
        payload = {
            "model": llm_model,
            "prompt": params.get("prompt"),
            "messages": self._llm_messages_convert(params),
            "temperature": float(params.get("temperature")),
            "max_new_tokens": int(params.get("max_new_tokens")),
            "echo": self.llm_echo,
        }
        # 保留原始英文请求日志格式，便于与现有日志检索规则兼容。
        logger.info(f"Request: \n{payload}")
        span = root_tracer.start_span(
            "Agent.llm_client.no_streaming_call",
            metadata=self._get_span_metadata(payload),
        )
        payload["span_id"] = span.span_id
        payload["model_cache_enable"] = self.model_cache_enable
        if params.get("context") is not None:
            payload["context"] = ModelRequestContext(extra=params["context"])
        try:
            model_request = _build_model_request(payload)
            str_prompt = model_request.messages_to_string()
            model_output: Optional[ModelOutput] = None
            previous_text = ""
            previous_thinking_text = ""

            async def _emit_stream_callback(payload: Dict[str, Any]) -> None:
                if not stream_callback:
                    return
                try:
                    if asyncio.iscoroutinefunction(stream_callback):
                        await stream_callback(payload)
                        return
                    result = stream_callback(payload)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    # 保留原始英文日志信息，便于与现有日志检索规则兼容。
                    logger.exception("stream_callback error")

            async for output in self._llm_client.generate_stream(model_request.copy()):
                model_output = output
                delta_text = ""
                delta_thinking = ""
                if output.has_text:
                    current_text = output.text or ""
                    if current_text.startswith(previous_text):
                        delta_text = current_text[len(previous_text) :]
                        previous_text = current_text
                    else:
                        delta_text = current_text
                        previous_text = current_text
                if output.has_thinking:
                    current_thinking = output.thinking_text or ""
                    if current_thinking.startswith(previous_thinking_text):
                        delta_thinking = current_thinking[len(previous_thinking_text) :]
                        previous_thinking_text = current_thinking
                    else:
                        delta_thinking = current_thinking
                        previous_thinking_text = current_thinking
                if delta_text or delta_thinking:
                    await _emit_stream_callback(
                        {
                            "delta_text": delta_text,
                            "delta_thinking": delta_thinking,
                        }
                    )
                if memory and stream_out:
                    # 仅用于触发相关模块加载；保留 noqa 指令以避免未使用导入告警。
                    from ... import GptsMemory  # noqa: F401

                    if model_output:
                        temp_message = {
                            "sender": sender,
                            "receiver": "?",
                            "model": llm_model,
                            "markdown": model_output.gen_text_with_thinking(),
                        }
                        await memory.push_message(
                            conv_id,
                            temp_message,
                        )
            if not model_output:
                # 保留原始英文异常信息，避免影响依赖该文本的调用方。
                raise ValueError("LLM generate stream is null!")
            parsed_output = model_output.gen_text_with_thinking()

            if verbose:
                print("\n", "-" * 80, flush=True, sep="")
                # 以下调试输出保留英文标签，便于与现有日志和调试记录兼容。
                print(f"String Prompt[verbose]: \n{str_prompt}")
                print(f"LLM Output[verbose]: \n{parsed_output}")
                print("-" * 80, "\n", flush=True, sep="")
            return parsed_output
        except Exception as e:
            # 保留原始英文日志信息，便于与现有日志检索规则兼容。
            logger.error(
                f"Call LLMClient error, {str(e)}, detail: {traceback.format_exc()}"
            )
            raise LLMChatError(original_exception=e) from e
        finally:
            span.end()
