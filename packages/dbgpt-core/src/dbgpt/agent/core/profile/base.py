"""配置（profile）模块。"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set

import cachetools
from jinja2.meta import find_undeclared_variables
from jinja2.sandbox import Environment, SandboxedEnvironment

from dbgpt._private.pydantic import BaseModel, ConfigDict, Field, model_validator
from dbgpt.util.configure import ConfigInfo, DynConfig

VALID_TEMPLATE_KEYS = {
    "role",
    "name",
    "goal",
    "resource_prompt",
    "expand_prompt",
    "language",
    "constraints",
    "examples",
    "out_schema",
    "most_recent_memories",
    "question",
}

_DEFAULT_SYSTEM_TEMPLATE = """\
You are a {{ role }}, {% if name %}named {{ name }}.
{% endif %}your goal is {% if is_retry_chat %}{{ retry_goal }}{% else %}{{ goal }}{% endif %}.\
Please think step-by-step to achieve your goals based on user input. You can use the resources given below.
At the same time, please strictly abide by the constraints and specifications in the "IMPORTANT REMINDER" below.
[Important Constraints]
- It is strictly prohibited to directly call any tool from resources in the task plan, even if they are listed in the available resources.
- All tool invocations must be performed only via the ToolExpert agent.
- The ToolExpert is responsible for managing and proxying all tool invocations. The Planner should only issue high-level intents to the ToolExpert for using tools.
{% if resource_prompt %}\
Given resources information:
{{ resource_prompt }} 
{% endif %}
{% if expand_prompt %}\
{{ expand_prompt }} 
{% endif %}\

*** IMPORTANT REMINDER ***
Please answer in English.
The current time is:{{now_time}}.

{% if is_retry_chat %}\
{% if retry_constraints %}\
{% for retry_constraint in retry_constraints %}\
{{ loop.index }}. {{ retry_constraint }}
{% endfor %}\
{% endif %}\
{% else %}\
{% if constraints %}\
{% for constraint in constraints %}\
{{ loop.index }}. {{ constraint }}
{% endfor %}\
{% endif %}\
{% endif %}\



{% if examples %}\
You can refer to the following examples:
{{ examples }}\
{% endif %}\

{% if out_schema %} {{ out_schema }} {% endif %}\
"""  # noqa

_DEFAULT_SYSTEM_TEMPLATE_ZH = """\
你是一个 {{ role }}, {% if name %}名字叫 {{ name }}.
{% endif %}你的目标是 {% if is_retry_chat %}{{ retry_goal }}{% else %}{{ goal }}{% endif %}.\
请一步一步思考完根据下面给出的已知信息和用户问题完成目标，同时请严格遵守下面"重要提醒"中的约束和规范。
【重要约束】
- 严禁在任务计划中直接调用任何 resource 中的 tool，即使它们在资源列表中被列出。
- 所有 tool 的调用必须通过 ToolExpert agent 实现。
- ToolExpert 的职责是统一管理、代理所有工具的调用，Planner 只应向 ToolExpert 发出工具的使用意图。
{% if resource_prompt %}\
已知资源信息：
{{ resource_prompt }} 
{% endif %}\
{% if expand_prompt %}\
{{ expand_prompt }} 
{% endif %}\

*** 重要提醒 ***
请用简体中文进行回答.
当前时间是:{{now_time}}
{% if is_retry_chat %}\
{% if retry_constraints %}\
{% for retry_constraint in retry_constraints %}\
{{ loop.index }}. {{ retry_constraint }}
{% endfor %}\
{% endif %}\
{% else %}\
{% if constraints %}\
{% for constraint in constraints %}\
{{ loop.index }}. {{ constraint }}
{% endfor %}\
{% endif %}\
{% endif %}\

{% if examples %}\
你也可以参考如下对话示例:
{{ examples }}\
{% endif %}\

{% if out_schema %} {{ out_schema }} {% endif %}\
"""  # noqa


_DEFAULT_USER_TEMPLATE = """\
{% if most_recent_memories %}\
Most recent message:
{{ most_recent_memories }}
{% endif %}\

{% if question %}\
User input: {{ question }}
{% endif %}
"""

_DEFAULT_USER_TEMPLATE_ZH = """\
{% if most_recent_memories %}\
最近消息记录:
{{ most_recent_memories }}
{% endif %}\

{% if question %}\
用户输入: {{ question }}
{% endif %}
"""

_DEFAULT_WRITE_MEMORY_TEMPLATE = """\
{% if question %}Question: {{ question }} {% endif %}
{% if thought %}Thought: {{ thought }} {% endif %}
{% if action %}Action: {{ action }} {% endif %}
{% if observation %}Observation: {{ observation }} {% endif %}
"""
_DEFAULT_WRITE_MEMORY_TEMPLATE_ZH = """\
{% if question %}问题: {{ question }} {% endif %}
{% if thought %}思考答案: {{ thought }} {% endif %}
{% if action %}行动结果: {{ action }} {% endif %}
{% if observation %}观察: {{ observation }} {% endif %}
"""


class Profile(ABC):
    """配置接口（profile interface）。"""

    @abstractmethod
    def get_name(self) -> str:
        """返回当前智能体名称。"""

    @abstractmethod
    def get_role(self) -> str:
        """返回当前智能体职责。"""

    def get_goal(self) -> Optional[str]:
        """返回当前智能体目标。"""
        return None

    def get_retry_goal(self) -> Optional[str]:
        """返回当前智能体的重试目标。"""
        return None

    def get_constraints(self) -> Optional[List[str]]:
        """返回当前智能体约束条件。"""
        return None

    def get_retry_constraints(self) -> Optional[List[str]]:
        """返回当前智能体的重试约束条件。"""
        return None

    def get_description(self) -> Optional[str]:
        """返回当前智能体描述。

        此描述不会用于生成提示词。
        """
        return None

    def get_expand_prompt(self) -> Optional[str]:
        """返回当前智能体的扩展提示词。"""
        return None

    def get_examples(self) -> Optional[str]:
        """返回当前智能体示例。"""
        return None

    @abstractmethod
    def get_system_prompt_template(self) -> str:
        """返回当前智能体的系统提示词模板。"""

    @abstractmethod
    def get_user_prompt_template(self) -> str:
        """返回当前智能体的用户提示词模板。"""

    @abstractmethod
    def get_write_memory_template(self) -> str:
        """返回当前智能体的记忆保存模板。"""

    def format_system_prompt(
        self,
        template_env: Optional[Environment] = None,
        question: Optional[str] = None,
        language: str = "en",
        most_recent_memories: Optional[str] = None,
        resource_vars: Optional[Dict[str, Any]] = None,
        is_retry_chat: bool = False,
        **kwargs,
    ) -> str:
        """格式化系统提示词。

        Args:
            template_env(Optional[Environment])：Jinja2 模板环境。
            question(Optional[str])：问题。
            language(str)：当前上下文语言。
            most_recent_memories(Optional[str])：最近的记忆内容，从记忆中读取。
            resource_vars(Optional[Dict[str, Any]])：资源变量。

        Returns:
            str：格式化后的系统提示词。
        """
        return self._format_prompt(
            self.get_system_prompt_template(),
            template_env=template_env,
            question=question,
            language=language,
            most_recent_memories=most_recent_memories,
            resource_vars=resource_vars,
            is_retry_chat=is_retry_chat,
            **kwargs,
        )

    def format_user_prompt(
        self,
        template_env: Optional[Environment] = None,
        question: Optional[str] = None,
        language: str = "en",
        most_recent_memories: Optional[str] = None,
        resource_vars: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """格式化用户提示词。

        Args:
            template_env(Optional[Environment])：Jinja2 模板环境。
            question(Optional[str])：问题。
            language(str)：当前上下文语言。
            most_recent_memories(Optional[str])：最近的记忆内容，从记忆中读取。
            resource_vars(Optional[Dict[str, Any]])：资源变量。

        Returns:
            str：格式化后的用户提示词。
        """
        return self._format_prompt(
            self.get_user_prompt_template(),
            template_env=template_env,
            question=question,
            language=language,
            most_recent_memories=most_recent_memories,
            resource_vars=resource_vars,
            **kwargs,
        )

    @property
    def _sub_render_keys(self) -> Set[str]:
        """返回子渲染（sub-render）键。

        如果值为字符串且键属于子渲染键，则会对其进行渲染。

        Returns:
            Set[str]：子渲染键。
        """
        return {"role", "name", "goal", "expand_prompt", "constraints"}

    def _format_prompt(
        self,
        template: str,
        template_env: Optional[Environment] = None,
        question: Optional[str] = None,
        language: str = "en",
        most_recent_memories: Optional[str] = None,
        resource_vars: Optional[Dict[str, Any]] = None,
        is_retry_chat: bool = False,
        **kwargs,
    ) -> str:
        """格式化提示词。"""
        if not template_env:
            template_env = SandboxedEnvironment()
        pass_vars = {
            "role": self.get_role(),
            "name": self.get_name(),
            "goal": self.get_goal(),
            "retry_goal": self.get_retry_goal(),
            "expand_prompt": self.get_expand_prompt(),
            "language": language,
            "constraints": self.get_constraints(),
            "retry_constraints": self.get_retry_constraints(),
            "most_recent_memories": (
                most_recent_memories if most_recent_memories else None
            ),
            "is_retry_chat": is_retry_chat,
            "examples": self.get_examples(),
            "question": question,
        }
        if resource_vars:
            # 合并资源变量。
            pass_vars.update(resource_vars)
        pass_vars.update(kwargs)

        # 解析模板，查找模板中的所有变量。
        template_vars = find_undeclared_variables(template_env.parse(template))

        # 仅保留有效的模板键变量。
        filtered_data = {
            key: pass_vars[key] for key in template_vars if key in pass_vars
        }

        def _render_template(_template_env, _template: str, **_kwargs):
            r_template = _template_env.from_string(_template)
            return r_template.render(**_kwargs)

        for key in filtered_data.keys():
            value = filtered_data[key]
            if key in self._sub_render_keys and value:
                if isinstance(value, str):
                    # 渲染子模板。
                    filtered_data[key] = _render_template(
                        template_env, value, **pass_vars
                    )
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, str):
                            value[i] = _render_template(template_env, item, **pass_vars)
        return _render_template(template_env, template, **filtered_data)


class DefaultProfile(BaseModel, Profile):
    """默认配置（default profile）。"""

    name: str = Field("", description="The name of the agent.")
    role: str = Field("", description="The role of the agent.")
    goal: Optional[str] = Field(None, description="The goal of the agent.")
    retry_goal: Optional[str] = Field(None, description="The retry goal of the agent.")
    constraints: Optional[List[str]] = Field(
        None, description="The constraints of the agent."
    )
    retry_constraints: Optional[List[str]] = Field(
        None, description="The retry constraints of the agent."
    )
    desc: Optional[str] = Field(
        None, description="The description of the agent, not used to generate prompt."
    )
    resource_introduction: Optional[str] = Field(
        None,
        description="The resource introduction of the agent, not used to generate prompt.",  # noqa
    )  # noqa
    expand_prompt: Optional[str] = Field(
        None, description="The expand prompt of the agent."
    )

    examples: Optional[str] = Field(
        None, description="The examples of the agent prompt."
    )

    system_prompt_template: str = Field(
        _DEFAULT_SYSTEM_TEMPLATE, description="The system prompt template of the agent."
    )
    user_prompt_template: str = Field(
        _DEFAULT_USER_TEMPLATE, description="The user prompt template of the agent."
    )

    write_memory_template: str = Field(
        _DEFAULT_WRITE_MEMORY_TEMPLATE,
        description="The save memory template of the agent.",
    )

    def get_name(self) -> str:
        """返回当前智能体名称。"""
        return self.name

    def get_role(self) -> str:
        """返回当前智能体职责。"""
        return self.role

    def get_goal(self) -> Optional[str]:
        """返回当前智能体目标。"""
        return self.goal

    def get_retry_goal(self) -> Optional[str]:
        """返回当前智能体的重试目标。"""
        return self.retry_goal

    def get_constraints(self) -> Optional[List[str]]:
        """返回当前智能体约束条件。"""
        return self.constraints

    def get_retry_constraints(self) -> Optional[List[str]]:
        """返回当前智能体的重试约束条件。"""
        return self.retry_constraints

    def get_description(self) -> Optional[str]:
        """返回当前智能体描述。

        此描述不会用于生成提示词。
        """
        return self.desc

    def get_expand_prompt(self) -> Optional[str]:
        """返回当前智能体的扩展提示词。"""
        return self.expand_prompt

    def get_examples(self) -> Optional[str]:
        """返回当前智能体示例。"""
        return self.examples

    def get_system_prompt_template(self) -> str:
        """返回当前智能体的系统提示词模板。"""
        return self.system_prompt_template

    def get_user_prompt_template(self) -> str:
        """返回当前智能体的用户提示词模板。"""
        return self.user_prompt_template

    def get_write_memory_template(self) -> str:
        """返回当前智能体的记忆保存模板。"""
        return self.write_memory_template


class ProfileFactory:
    """配置工厂接口（profile factory interface）。

    用于创建配置。
    """

    @abstractmethod
    def create_profile(
        self,
        profile_id: int,
        name: Optional[str] = None,
        role: Optional[str] = None,
        goal: Optional[str] = None,
        prefer_prompt_language: Optional[str] = None,
        prefer_model: Optional[str] = None,
    ) -> Optional[Profile]:
        """创建配置。"""


class LLMProfileFactory(ProfileFactory):
    """通过 LLM 创建配置。

    基于 LLM 自动生成，通常先指定配置生成规则，明确目标人群中智能体配置的组成和属性，
    然后提供少量样本，最后由 LLM 生成所有智能体的配置。
    """

    def create_profile(
        self,
        profile_id: int,
        name: Optional[str] = None,
        role: Optional[str] = None,
        goal: Optional[str] = None,
        prefer_prompt_language: Optional[str] = None,
        prefer_model: Optional[str] = None,
    ) -> Optional[Profile]:
        """通过 LLM 创建配置。

        TODO：实现此方法。
        """
        pass


class DatasetProfileFactory(ProfileFactory):
    """通过数据集（dataset）创建配置。

    使用现有数据集生成智能体配置。

    某些情况下，数据集包含大量真实人物信息；先将这些信息整理为自然语言提示词，
    再用它生成智能体配置。
    """

    def create_profile(
        self,
        profile_id: int,
        name: Optional[str] = None,
        role: Optional[str] = None,
        goal: Optional[str] = None,
        prefer_prompt_language: Optional[str] = None,
        prefer_model: Optional[str] = None,
    ) -> Optional[Profile]:
        """通过数据集创建配置。

        TODO：实现此方法。
        """
        pass


class CompositeProfileFactory(ProfileFactory):
    """通过组合多个配置工厂创建配置。"""

    def __init__(self, factories: List[ProfileFactory]):
        """创建组合配置工厂。"""
        self.factories = factories

    def create_profile(
        self,
        profile_id: int,
        name: Optional[str] = None,
        role: Optional[str] = None,
        goal: Optional[str] = None,
        prefer_prompt_language: Optional[str] = None,
        prefer_model: Optional[str] = None,
    ) -> Optional[Profile]:
        """通过组合多个配置工厂创建配置。

        TODO：实现此方法。
        """
        pass


class ProfileConfig(BaseModel):
    """配置设置（profile configuration）。

    未指定 factory 时，必须指定 name 和 role。
    如果同时指定 factory、name 和 role，将优先使用 factory。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    profile_id: int = Field(0, description="The profile ID.")
    name: str | ConfigInfo | None = DynConfig(..., description="The name of the agent.")
    role: str | ConfigInfo | None = DynConfig(..., description="The role of the agent.")
    goal: str | ConfigInfo | None = DynConfig(None, description="The retry goal.")
    retry_goal: str | ConfigInfo | None = DynConfig(None, description="The goal.")
    constraints: List[str] | ConfigInfo | None = DynConfig(None, is_list=True)
    retry_constraints: List[str] | ConfigInfo | None = DynConfig(None, is_list=True)
    desc: str | ConfigInfo | None = DynConfig(
        None, description="The description of the agent."
    )
    expand_prompt: str | ConfigInfo | None = DynConfig(
        None, description="The expand prompt."
    )
    examples: str | ConfigInfo | None = DynConfig(None, description="The examples.")

    system_prompt_template: str | ConfigInfo | None = DynConfig(
        _DEFAULT_SYSTEM_TEMPLATE, description="The prompt template."
    )
    user_prompt_template: str | ConfigInfo | None = DynConfig(
        _DEFAULT_USER_TEMPLATE, description="The user prompt template."
    )
    write_memory_template: str | ConfigInfo | None = DynConfig(
        _DEFAULT_WRITE_MEMORY_TEMPLATE, description="The save memory template."
    )
    factory: ProfileFactory | None = Field(None, description="The profile factory.")

    @model_validator(mode="before")
    @classmethod
    def check_before(cls, values):
        """在验证（validation）前执行检查。"""
        if isinstance(values, dict):
            return values
        if values["factory"] is None:
            if values["name"] is None:
                raise ValueError("name must be specified if factory is not specified")
            if values["role"] is None:
                raise ValueError("role must be specified if factory is not specified")
        return values

    @cachetools.cached(cachetools.TTLCache(maxsize=100, ttl=10))
    def create_profile(
        self,
        profile_id: Optional[int] = None,
        prefer_prompt_language: Optional[str] = None,
        prefer_model: Optional[str] = None,
    ) -> Profile:
        """创建配置。

        如果指定了 factory，则使用该工厂创建配置。
        """
        factory_profile = None
        if profile_id is None:
            profile_id = self.profile_id
        name = self.name
        role = self.role
        goal = self.goal
        retry_goal = self.retry_goal
        retry_constraints = self.retry_constraints
        constraints = self.constraints
        desc = self.desc
        expand_prompt = self.expand_prompt
        system_prompt_template = self.system_prompt_template
        user_prompt_template = self.user_prompt_template
        write_memory_template = self.write_memory_template
        examples = self.examples
        call_args = {
            "prefer_prompt_language": prefer_prompt_language,
            "prefer_model": prefer_model,
        }
        if isinstance(name, ConfigInfo):
            name = name.query(**call_args)
        if isinstance(role, ConfigInfo):
            role = role.query(**call_args)
        if isinstance(goal, ConfigInfo):
            goal = goal.query(**call_args)
        if isinstance(retry_goal, ConfigInfo):
            retry_goal = retry_goal.query(**call_args)
        if isinstance(constraints, ConfigInfo):
            constraints = constraints.query(**call_args)
        if isinstance(retry_constraints, ConfigInfo):
            retry_constraints = retry_constraints.query(**call_args)
        if isinstance(desc, ConfigInfo):
            desc = desc.query(**call_args)
        if isinstance(expand_prompt, ConfigInfo):
            expand_prompt = expand_prompt.query(**call_args)
        if isinstance(examples, ConfigInfo):
            examples = examples.query(**call_args)
        if isinstance(system_prompt_template, ConfigInfo):
            system_prompt_template.default = (
                _DEFAULT_SYSTEM_TEMPLATE
                if prefer_prompt_language == "en"
                else _DEFAULT_SYSTEM_TEMPLATE_ZH
            )
            system_prompt_template = system_prompt_template.query(**call_args)
        if isinstance(user_prompt_template, ConfigInfo):
            user_prompt_template.default = (
                _DEFAULT_USER_TEMPLATE
                if prefer_prompt_language == "en"
                else _DEFAULT_USER_TEMPLATE_ZH
            )
            user_prompt_template = user_prompt_template.query(**call_args)
        if isinstance(write_memory_template, ConfigInfo):
            write_memory_template = write_memory_template.query(**call_args)

        if self.factory is not None:
            factory_profile = self.factory.create_profile(
                profile_id,
                name,
                role,
                goal,
                prefer_prompt_language,
                prefer_model,
            )

        if factory_profile is not None:
            return factory_profile
        return DefaultProfile(
            name=name,
            role=role,
            goal=goal,
            retry_goal=retry_goal,
            constraints=constraints,
            retry_constraints=retry_constraints,
            desc=desc,
            expand_prompt=expand_prompt,
            examples=examples,
            system_prompt_template=system_prompt_template,
            user_prompt_template=user_prompt_template,
            write_memory_template=write_memory_template,
        )

    def __hash__(self):
        """返回哈希值。"""
        return hash(self.profile_id)
