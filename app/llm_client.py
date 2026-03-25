"""
智能沟通翻译助手 - LLM调用模块

支持多种大模型API：OpenAI GPT、Anthropic Claude、硅基流动、通义千问、火山云豆包 等
"""

import os
from typing import AsyncGenerator
from abc import ABC, abstractmethod
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# LLM客户端抽象基类
# ============================================================================

class BaseLLMClient(ABC):
    """LLM客户端抽象基类"""
    
    @abstractmethod
    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """发送聊天请求并获取响应"""
        pass
    
    @abstractmethod
    async def stream_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> AsyncGenerator[str, None]:
        """发送聊天请求并获取流式响应"""
        pass


# ============================================================================
# OpenAI GPT客户端
# ============================================================================

class OpenAIClient(BaseLLMClient):
    """OpenAI GPT模型客户端"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        import openai
        
        client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url if self.base_url != "https://api.openai.com/v1" else None
        )
        
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    async def stream_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> AsyncGenerator[str, None]:
        import openai
        
        client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url if self.base_url != "https://api.openai.com/v1" else None
        )
        
        stream = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# ============================================================================
# Anthropic Claude客户端
# ============================================================================

class AnthropicClient(BaseLLMClient):
    """Anthropic Claude模型客户端"""
    
    def __init__(self, api_key: str = None, model: str = "claude-3-haiku-20240307"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        
    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        import anthropic
        
        client = anthropic.AsyncAnthropic(api_key=self.api_key)
        
        response = await client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        
        return response.content[0].text
    
    async def stream_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> AsyncGenerator[str, None]:
        import anthropic
        
        client = anthropic.AsyncAnthropic(api_key=self.api_key)
        
        async with client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        ) as stream:
            async for text in stream.text_stream:
                yield text


# ============================================================================
# 硅基流动（SiliconFlow）客户端 - 兼容OpenAI格式
# ============================================================================

class SiliconFlowClient(BaseLLMClient):
    """硅基流动API客户端（国内可访问）"""
    
    def __init__(self, api_key: str = None, model: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
        self.base_url = "https://api.siliconflow.cn/v1"
        self.model = model
        
    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        import openai
        
        client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    async def stream_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> AsyncGenerator[str, None]:
        import openai
        
        client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        stream = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# ============================================================================
# 通义千问（阿里云）客户端
# ============================================================================

class QwenClient(BaseLLMClient):
    """阿里云通义千问API客户端"""
    
    def __init__(self, api_key: str = None, model: str = "qwen-plus"):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.model = model
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        import openai
        
        client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    async def stream_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> AsyncGenerator[str, None]:
        import openai
        
        client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        stream = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# ============================================================================
# 火山云豆包（Doubao）客户端
# ============================================================================

class DoubaoClient(BaseLLMClient):
    """火山云豆包API客户端"""
    
    def __init__(self, api_key: str = None, model: str = "doubao-pro"):
        self.api_key = api_key or os.getenv("DOUBAO_API_KEY")
        # 火山云方舟API地址（不同region可能不同）
        self.base_url = os.getenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
        self.model = model
        
    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        import openai
        
        client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    async def stream_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> AsyncGenerator[str, None]:
        import openai
        
        client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        stream = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# ============================================================================
# LLM客户端工厂
# ============================================================================

class LLMClientFactory:
    """LLM客户端工厂类"""
    
    _clients = {
        "openai": OpenAIClient,
        "claude": AnthropicClient,
        "siliconflow": SiliconFlowClient,
        "qwen": QwenClient,
        "doubao": DoubaoClient,
    }
    
    @classmethod
    def create(cls, provider: str = None, **kwargs) -> BaseLLMClient:
        """
        创建LLM客户端实例
        
        Args:
            provider: 提供商名称 ("openai", "claude", "siliconflow", "qwen")
            **kwargs: 传递给客户端的其他参数
            
        Returns:
            LLM客户端实例
        """
        if provider is None:
            # 自动检测可用提供商
            provider = cls._detect_available_provider()
        
        provider = provider.lower()
        
        if provider not in cls._clients:
            available = ", ".join(cls._clients.keys())
            raise ValueError(f"不支持的提供商: {provider}，可用选项: {available}")
        
        return cls._clients[provider](**kwargs)
    
    @classmethod
    def _detect_available_provider(cls) -> str:
        """检测可用的LLM提供商"""
        if os.getenv("DOUBAO_API_KEY"):
            return "doubao"
        if os.getenv("DASHSCOPE_API_KEY"):
            return "qwen"
        if os.getenv("SILICONFLOW_API_KEY"):
            return "siliconflow"
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("ANTHROPIC_API_KEY"):
            return "claude"
        
        # 默认使用火山云豆包（国内可访问）
        return "doubao"
    
    @classmethod
    def register(cls, name: str, client_class: type):
        """注册新的LLM客户端"""
        cls._clients[name.lower()] = client_class


# ============================================================================
# 便捷函数
# ============================================================================

def get_llm_client(provider: str = None, **kwargs) -> BaseLLMClient:
    """获取LLM客户端的便捷函数"""
    return LLMClientFactory.create(provider, **kwargs)
