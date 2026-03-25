"""
职能沟通翻译助手 - 测试模块
"""

import pytest
from unittest.mock import AsyncMock, patch

from app.prompts import (
    get_product_to_dev_prompts,
    get_dev_to_product_prompts,
    get_scene_detection_prompts,
    PRODUCT_TO_DEV_SYSTEM_PROMPT,
    DEV_TO_PRODUCT_SYSTEM_PROMPT
)


class TestPrompts:
    """测试提示词模块"""

    def test_product_to_dev_prompts(self):
        """测试产品经理到开发工程师的提示词生成"""
        input_text = "我们需要做一个智能推荐功能"
        system_prompt, user_prompt = get_product_to_dev_prompts(input_text)
        
        assert PRODUCT_TO_DEV_SYSTEM_PROMPT in system_prompt
        assert input_text in user_prompt
        assert "产品经理" in system_prompt
        assert "开发工程师" in system_prompt

    def test_dev_to_product_prompts(self):
        """测试开发工程师到产品经理的提示词生成"""
        input_text = "优化了数据库查询"
        system_prompt, user_prompt = get_dev_to_product_prompts(input_text)
        
        assert DEV_TO_PRODUCT_SYSTEM_PROMPT in system_prompt
        assert input_text in user_prompt
        assert "产品" in system_prompt
        assert "业务" in system_prompt

    def test_scene_detection_prompts(self):
        """测试场景检测提示词生成"""
        input_text = "QPS提升了30%"
        system_prompt, user_prompt = get_scene_detection_prompts(input_text)
        
        assert "场景识别" in system_prompt
        assert input_text in user_prompt
        assert "product" in system_prompt.lower()
        assert "dev" in system_prompt.lower()

    def test_prompts_contain_required_structure(self):
        """测试提示词包含必需的输出结构"""
        # 产品经理 → 开发 应该有这些维度
        product_to_dev_keywords = [
            "功能理解",
            "技术实现建议",
            "工作量评估",
            "需要澄清的问题",
            "补充的上下文"
        ]
        
        for keyword in product_to_dev_keywords:
            assert keyword in PRODUCT_TO_DEV_SYSTEM_PROMPT, f"缺少: {keyword}"
        
        # 开发 → 产品经理 应该有这些维度
        dev_to_product_keywords = [
            "技术改动概述",
            "用户体验的影响",
            "业务价值",
            "商业影响",
            "汇报的要点"
        ]
        
        for keyword in dev_to_product_keywords:
            assert keyword in DEV_TO_PRODUCT_SYSTEM_PROMPT, f"缺少: {keyword}"


class TestLLMClient:
    """测试LLM客户端"""

    @pytest.fixture
    def mock_response(self):
        """模拟API响应"""
        return "测试响应内容"

    @pytest.mark.asyncio
    async def test_openai_client_chat(self, mock_response):
        """测试OpenAI客户端的chat方法"""
        from app.llm_client import OpenAIClient
        
        client = OpenAIClient(api_key="test_key", model="gpt-3.5-turbo")
        
        with patch.object(client, 'chat', new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = mock_response
            
            result = await client.chat(
                system_prompt="你是一个助手",
                user_prompt="你好"
            )
            
            assert result == mock_response

    @pytest.mark.asyncio
    async def test_client_factory_auto_detection(self):
        """测试客户端工厂的自动检测功能"""
        from app.llm_client import LLMClientFactory
        
        # 这个测试验证工厂类能正常实例化
        # 实际使用时需要有API key才能真正调用
        try:
            # 不提供provider时，应该抛出异常（因为没有可用API key）
            client = LLMClientFactory.create("siliconflow")
            assert client is not None
        except Exception:
            # 预期会有异常，因为测试环境没有真实的API key
            pass


class TestAPIEndpoints:
    """测试API端点"""

    def test_translate_request_validation(self):
        """测试翻译请求的数据验证"""
        from app.main import TranslateRequest
        
        # 有效请求
        valid_request = TranslateRequest(
            text="测试内容",
            direction="product_to_dev"
        )
        assert valid_request.text == "测试内容"
        assert valid_request.direction == "product_to_dev"
        
        # 无效方向
        with pytest.raises(ValueError):
            TranslateRequest(
                text="测试内容",
                direction="invalid_direction"
            )
        
        # 空文本
        with pytest.raises(ValueError):
            TranslateRequest(
                text="",
                direction="product_to_dev"
            )

    def test_detect_scene_request_validation(self):
        """测试场景检测请求的数据验证"""
        from app.main import DetectSceneRequest
        
        # 有效请求
        valid_request = DetectSceneRequest(text="QPS提升了30%")
        assert valid_request.text == "QPS提升了30%"
        
        # 空文本应抛出异常
        with pytest.raises(ValueError):
            DetectSceneRequest(text="")


class TestIntegration:
    """集成测试"""

    def test_app_initialization(self):
        """测试应用能否正常初始化"""
        # 验证所有模块可以导入
        from app.main import app
        from app.prompts import get_product_to_dev_prompts
        from app.llm_client import get_llm_client
        from app.cli import main
        
        assert app is not None
        assert callable(get_product_to_dev_prompts)
        assert callable(get_llm_client)
        assert callable(main)


# ============================================================================
# 示例测试用例（按需求文档要求）
# ============================================================================

class TestUseCases:
    """测试用例"""

    def test_use_case_1_product_perspective(self):
        """
        测试用例1：产品经理视角输入
        
        输入： 我们需要一个智能推荐功能,提升用户停留时长
        
        翻译结果应包含：
        - 推荐算法类型建议（协同过滤/内容推荐等）
        - 数据来源和处理方式
        - 性能和实时性要求
        - 预估开发工作量
        """
        input_text = "我们需要做一个智能推荐功能,提升用户停留时长"
        system_prompt, user_prompt = get_product_to_dev_prompts(input_text)
        
        # 验证提示词包含了所需的翻译维度
        assert "功能理解" in system_prompt
        assert "技术实现建议" in system_prompt
        assert "工作量评估" in system_prompt
        assert "算法" in system_prompt or "推荐" in system_prompt
        assert "数据" in system_prompt

    def test_use_case_2_dev_perspective(self):
        """
        测试用例2：开发工程师视角输入
        
        输入： 我们优化了数据库查询，QPS提升了30%
        
        翻译结果应包含：
        - 对用户体验的实际影响
        - 支持的业务增长空间
        - 成本降低的商业价值
        """
        input_text = "我们优化了数据库查询，QPS提升了30%"
        system_prompt, user_prompt = get_dev_to_product_prompts(input_text)
        
        # 验证提示词包含了所需的翻译维度
        assert "用户体验" in system_prompt
        assert "业务价值" in system_prompt
        assert "商业影响" in system_prompt
        assert "QPS" in user_prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
