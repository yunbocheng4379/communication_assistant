"""
智能沟通翻译助手 - FastAPI后端服务

提供RESTful API接口，支持流式输出
"""

import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 直接导入 app 模块
from app.main import app

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=False
    )
