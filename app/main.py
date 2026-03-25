"""
智能沟通翻译助手 - FastAPI后端服务

提供RESTful API接口，支持流式输出
"""

import os
import json
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.llm_client import get_llm_client, BaseLLMClient
from app.prompts import (
    get_product_to_dev_prompts,
    get_dev_to_product_prompts,
    get_scene_detection_prompts
)

# ============================================================================
# 应用生命周期管理
# ============================================================================

llm_client: Optional[BaseLLMClient] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global llm_client
    
    # 启动时初始化LLM客户端
    provider = os.getenv("LLM_PROVIDER", None)
    
    # 根据提供商获取对应的模型配置
    kwargs = {}
    if provider:
        provider = provider.lower()
        if provider == "doubao":
            model = os.getenv("DOUBAO_MODEL", None)
        elif provider == "openai":
            model = os.getenv("OPENAI_MODEL", None)
        elif provider == "qwen":
            model = os.getenv("DASHSCOPE_MODEL", None)
        elif provider == "siliconflow":
            model = os.getenv("SILICONFLOW_MODEL", None)
        else:
            model = None
    else:
        model = None
    
    if model:
        kwargs["model"] = model
    
    try:
        llm_client = get_llm_client(provider, **kwargs)
        print(f"✓ LLM客户端初始化成功，提供商: {provider or '自动检测'}, 模型: {model or '默认'}")
    except Exception as e:
        print(f"✗ LLM客户端初始化失败: {e}")
        raise
    
    yield
    
    # 关闭时清理资源
    print("✗ 应用关闭")

# ============================================================================
# FastAPI应用初始化
# ============================================================================

app = FastAPI(
    title="职能沟通翻译助手",
    description="帮助产品经理和开发工程师更好地理解彼此的AI翻译工具",
    version="1.0.0",
    lifespan=lifespan
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 处理 Chrome DevTools 探测请求
@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def chrome_devtools_probe():
    """处理 Chrome DevTools 探测请求，避免 404 错误"""
    return {}

# ============================================================================
# 数据模型
# ============================================================================

class TranslateRequest(BaseModel):
    """翻译请求模型"""
    text: str = Field(..., min_length=1, max_length=5000, description="待翻译的文本")
    direction: str = Field(..., pattern="^(product_to_dev|dev_to_product)$", description="翻译方向")

class TranslateResponse(BaseModel):
    """翻译响应模型"""
    result: str
    direction: str
    detected_scene: Optional[str] = None

class DetectSceneRequest(BaseModel):
    """场景检测请求模型"""
    text: str = Field(..., min_length=1, max_length=5000)

class DetectSceneResponse(BaseModel):
    """场景检测响应模型"""
    scene: str  # "product" | "dev" | "unknown"

# ============================================================================
# API路由
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """返回前端页面"""
    return FileResponse("static/index.html")

@app.get("/api/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "ok",
        "llm_provider": os.getenv("LLM_PROVIDER", "auto"),
        "llm_model": os.getenv("LLM_MODEL", "default")
    }

@app.post("/api/translate", response_model=TranslateResponse)
async def translate(request: TranslateRequest):
    """
    翻译接口（非流式）
    
    - **text**: 待翻译的文本
    - **direction**: 翻译方向 (product_to_dev | dev_to_product)
    """
    if llm_client is None:
        raise HTTPException(status_code=503, detail="LLM服务未初始化")
    
    try:
        # 获取对应的提示词
        if request.direction == "product_to_dev":
            system_prompt, user_prompt = get_product_to_dev_prompts(request.text)
        else:
            system_prompt, user_prompt = get_dev_to_product_prompts(request.text)
        
        # 调用LLM
        result = await llm_client.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.7,
            max_tokens=2000
        )
        
        return TranslateResponse(
            result=result,
            direction=request.direction
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"翻译失败: {str(e)}")

@app.get("/api/translate/stream")
async def translate_stream(text: str, direction: str):
    """
    流式翻译接口
    
    - **text**: 待翻译的文本 (URL编码)
    - **direction**: 翻译方向 (product_to_dev | dev_to_product)
    """
    if llm_client is None:
        raise HTTPException(status_code=503, detail="LLM服务未初始化")
    
    if direction not in ["product_to_dev", "dev_to_product"]:
        raise HTTPException(status_code=400, detail="无效的翻译方向")
    
    async def event_generator():
        try:
            # 获取对应的提示词
            if direction == "product_to_dev":
                system_prompt, user_prompt = get_product_to_dev_prompts(text)
            else:
                system_prompt, user_prompt = get_dev_to_product_prompts(text)
            
            # 流式调用LLM
            async for chunk in llm_client.stream_chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7,
                max_tokens=2000
            ):
                yield {
                    "event": "message",
                    "data": json.dumps({"content": chunk})
                }
            
            # 发送完成信号
            yield {
                "event": "done",
                "data": json.dumps({"direction": direction})
            }
            
        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }
    
    return EventSourceResponse(event_generator())

@app.post("/api/detect-scene", response_model=DetectSceneResponse)
async def detect_scene(request: DetectSceneRequest):
    """
    场景检测接口
    
    自动识别输入内容是产品经理视角还是开发工程师视角
    """
    if llm_client is None:
        raise HTTPException(status_code=503, detail="LLM服务未初始化")
    
    try:
        system_prompt, user_prompt = get_scene_detection_prompts(request.text)
        
        result = await llm_client.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0,
            max_tokens=50
        )
        
        # 清理结果
        scene = result.strip().lower()
        if scene not in ["product", "dev", "unknown"]:
            scene = "unknown"
        
        return DetectSceneResponse(scene=scene)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"场景检测失败: {str(e)}")

# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
