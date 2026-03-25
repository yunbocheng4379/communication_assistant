# 职能沟通翻译助手

让产品经理和开发工程师更好地理解彼此的AI翻译工具。

## 功能特性

- **产品经理 → 开发工程师**：将业务需求翻译成技术实现方案
- **开发工程师 → 产品经理**：将技术方案翻译成业务价值和用户价值
- **自动场景识别**：智能识别输入内容的视角类型
- **流式输出**：实时显示AI生成的翻译结果
- **多LLM支持**：支持OpenAI、Claude、硅基流动、通义千问、火山云豆包等多种大模型

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API Key

创建 `.env` 文件，添加以下环境变量（至少选择一个LLM提供商）：

```bash
# OpenAI
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-3.5-turbo

# Anthropic Claude
ANTHROPIC_API_KEY=your_anthropic_api_key

# 硅基流动（国内可访问）
SILICONFLOW_API_KEY=your_siliconflow_api_key

# 阿里云通义千问
DASHSCOPE_API_KEY=your_dashscope_api_key

# 火山云豆包（国内可访问，推荐）
DOUBAO_API_KEY=your_doubao_api_key
DOUBAO_MODEL=doubao-pro
DOUBAO_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
```

### 3. 运行服务

```bash
python main.py
```

或使用uvicorn：

```bash
uvicorn main:app --reload --port 8000
```

### 4. 访问页面

打开浏览器访问：`http://localhost:8000`

## 使用说明

1. **选择翻译方向**：产品经理 → 开发 或 开发 → 产品经理
2. **输入内容**：在左侧文本框输入待翻译内容，或点击示例快速体验
3. **开始翻译**：点击"开始翻译"按钮，或使用快捷键 `Ctrl+Enter`
4. **查看结果**：右侧实时显示翻译结果
5. **自动识别**：点击"自动识别输入视角"，AI会智能判断输入内容属于哪种视角

## 项目结构

```
communication_assistant/
├── main.py              # 服务入口文件
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI应用定义
│   ├── prompts.py       # 提示词工程
│   └── llm_client.py    # LLM客户端
├── static/
│   └── index.html       # Web前端界面
├── tests/
│   └── test_app.py      # 单元测试
├── requirements.txt     # Python依赖
├── .env                 # 环境变量配置
└── README.md            # 项目文档
```

## API接口

### 健康检查

```
GET /api/health
```

### 翻译（非流式）

```
POST /api/translate
Content-Type: application/json

{
    "text": "我们需要做一个智能推荐功能",
    "direction": "product_to_dev"
}
```

### 翻译（流式）

```
GET /api/translate/stream?text=xxx&direction=product_to_dev
```

### 场景识别

```
POST /api/detect-scene
Content-Type: application/json

{
    "text": "优化了数据库查询，QPS提升了30%"
}
```

## 测试用例

### 测试用例1：产品经理视角输入

**输入：**

```
我们需要做一个智能推荐功能,提升用户停留时长
```

**翻译结果（开发工程师视角）应包含：**

- 推荐算法类型（协同过滤、内容推荐、混合推荐）
- 数据来源和特征工程方案
- 性能指标要求（延迟、吞吐量）
- 预估开发工期
- 需要澄清的关键问题（实时性要求、数据规模）

### 测试用例2：开发工程师视角输入

**输入：**

```
我们优化了数据库查询，QPS提升了30%
```

**翻译结果（产品经理视角）应包含：**

- 用业务语言解释这次改动（更快、更稳定）
- 对用户体验的实际改善（减少等待时间）
- 业务价值体现（支持业务增长、降低运营成本）
- 向领导汇报的核心信息

## 提示词设计说明

### 核心设计思路

本系统的核心是精心设计的提示词工程，让LLM能够准确理解并转换产品经理和开发工程师两种不同的思维模式。

**三大设计维度：**

1. **角色定义**
   - 产品经理 → 开发：让LLM扮演"资深技术架构师"
   - 开发 → 产品经理：让LLM扮演"资深产品顾问"

2. **输出格式约束**
   - 统一的结构化输出格式
   - 每个方向都有5个必须包含的维度
   - 确保输出的完整性和一致性

3. **关键原则**
   - 明确的边界和约束
   - 避免的常见错误
   - 必须遵循的价值导向

### 两种视角的差异

**产品经理视角特征：**
- 关注用户价值、商业目标、功能描述
- 使用业务术语（用户价值、转化率、留存率）
- 较少涉及具体技术实现

**开发工程师视角特征：**
- 关注技术实现、工作量评估、具体细节
- 使用技术术语（QPS、TPS、缓存、API）
- 较少涉及业务价值

### 提示词关键设计点

#### 产品经理 → 开发

```python
# 关键技术要点：
1. 功能理解 → 确保真正理解业务目标，而非表面需求
2. 技术实现建议 → 推荐算法/架构，但指出需要确定的方向
3. 工作量评估 → 务实的工期预估，列出风险点
4. 需要澄清的问题 → 主动提出产品经理没有说明但开发必须知道的问题
5. 补充上下文 → 站在开发者角度，补充遗漏的重要信息
```

#### 开发 → 产品经理

```python
# 关键技术要点：
1. 技术改动概述 → 用类比方式让非技术人员理解
2. 用户体验影响 → 始终聚焦"这对用户意味着什么"
3. 业务价值 → 将技术指标转化为业务语言
   - QPS提升30% = 支持3倍用户增长
   - 延迟降低90% = 用户等待时间从5秒变为0.5秒
4. 商业影响 → 成本、收入、风险
5. 汇报要点 → 3句话总结，准备好可能被问的问题
```

### 场景识别实现

通过专门的场景识别提示词，自动判断输入内容属于哪种视角：

```python
# 判断标准：
- 产品经理：业务术语、用户需求、商业目标
- 开发工程师：技术术语、性能指标、系统架构

# 输出约束：
- 直接输出一个单词："product" | "dev" | "unknown"
- 不输出任何其他内容
- temperature=0 确保结果稳定
```

## 技术栈

- **后端**：Python 3.10 + FastAPI
- **前端**：HTML + CSS + JavaScript（原生实现，无框架依赖）
- **AI**：支持 OpenAI GPT、Claude、硅基流动、通义千问、火山云豆包 等多种LLM
- **流式输出**：SSE（Server-Sent Events）

## 环境要求

- Python 3.10+
- 一个可用的LLM API（OpenAI/Claude/硅基流动/通义千问/火山云豆包）

## 火山云豆包配置说明

火山云豆包模型通过"方舟平台"提供服务，配置步骤：

1. **获取API Key**：登录[火山引擎控制台](https://console.volcengine.com/)，进入"方舟平台"创建API Key

2. **选择Region**：
   - 华北（北京）：`https://ark.cn-beijing.volces.com/api/v3`
   - 华东（上海）：`https://ark.cn-shanghai.volces.com/api/v3`

3. **选择模型**：推荐使用 `doubao-pro`（Doubao-Pro-32K）或 `doubao-lite`（Doubao-Lite-32K）

4. **配置示例**：

```bash
DOUBAO_API_KEY=your_api_key_here
DOUBAO_MODEL=doubao-pro
DOUBAO_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
LLM_PROVIDER=doubao
```

**注意**：火山云豆包API与OpenAI格式兼容，因此可以直接复用 `openai` Python SDK。

## License

MIT License
