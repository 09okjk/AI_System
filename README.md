# AI Agent 后端服务 v2.1

一个现代化的 AI 智能代理后端服务，基于 Python FastAPI 构建，专注于提供高性能的 API 接口服务。

## 🎯 项目特点

- **纯后端架构**: 专注于 API 服务，前后端完全分离
- **模块化设计**: API接口按功能分离，便于维护和扩展
- **多模型支持**: 集成多种 LLM 提供商（DashScope、Xinference、OpenAI、Anthropic 等）
- **MCP 工具系统**: 支持 Model Context Protocol 工具扩展
- **语音处理**: 集成语音识别和语音合成功能（SensVoice、Whisper、CosyVoice、Edge TTS）
- **CosyVoice音色一致性**: 支持speaker ID缓存，确保多次合成保持相同音色
- **MongoDB数据管理**: 支持文字和图片混合数据的增删改查
- **完整日志系统**: 结构化日志记录，运行步骤可视化
- **配置管理**: 灵活的配置系统，支持动态更新
- **测试框架**: 完整的测试套件，包括单元测试、集成测试和性能测试
- **智能启动**: 环境检查、依赖验证、配置管理一体化

## 🏗️ 架构设计

```
AI_Agent_Backend/
├── main.py                  # FastAPI 应用定义
├── start_server.py         # 启动脚本（推荐使用）
├── test_runner.py          # 测试运行器
├── install_mongodb.sh      # MongoDB 安装脚本
├── requirements.txt        # 依赖包列表
├── .env.example           # 环境变量示例
├── api/                    # API模块目录（新增）
│   ├── __init__.py        # 路由注册
│   ├── core_api.py        # 核心接口（系统状态、对话）
│   ├── mcp_api.py         # MCP配置接口
│   ├── llm_api.py         # LLM配置接口
│   ├── speech_api.py      # 语音处理接口
│   └── mongodb_api.py     # MongoDB数据管理接口（新增）
├── src/
│   ├── __init__.py
│   ├── models.py          # 数据模型定义（已扩展）
│   ├── logger.py          # 日志系统
│   ├── config.py          # 配置管理
│   ├── utils.py           # 工具函数
│   ├── mcp.py             # MCP管理器
│   ├── llm.py             # LLM管理器
│   ├── speech.py          # 语音处理器
│   └── mongodb_manager.py # MongoDB管理器（新增）
├── config/                 # 配置文件目录
│   ├── mcp_configs.json
│   ├── llm_configs.json
│   └── app_config.json
├── tools/                  # MCP 工具目录
│   └── math_server.py     # 数学计算工具示例
├── logs/                   # 日志文件目录
├── tests/                  # 测试文件目录
├── assets/                 # 资源文件目录
└── reference_audio/        # 参考音频文件目录
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- MongoDB 4.4+（新增）
- pip 或 conda

### 1. 安装依赖

```bash
# 克隆项目
git clone https://github.com/09okjk/AI_System.git
cd AI_System

# 安装核心依赖
pip install -r requirements.txt
```

### 2. 安装和配置MongoDB

```bash
# 使用安装脚本（Ubuntu 24.04）
chmod +x install_mongodb.sh
./install_mongodb.sh

# 或手动安装
sudo apt update
sudo apt install -y mongodb

# 启动MongoDB服务
sudo systemctl start mongod
sudo systemctl enable mongod
```

### 3. 环境变量配置

1. 复制示例环境文件：

```bash
cp .env.example .env
```

2. 编辑 `.env` 文件，配置您的 API 密钥和MongoDB连接：

```env
# DashScope API 配置
DASHSCOPE_API_KEY=your_dashscope_api_key_here
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# Xinference 配置
XINFERENCE_ENDPOINT=http://localhost:9997/v1

# OpenAI API 配置
OPENAI_API_KEY=your_openai_api_key_here

# MongoDB 配置（新增）
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_DATABASE=ai_system
MONGODB_USERNAME=
MONGODB_PASSWORD=

# 语音模型配置
COSYVOICE_MODEL_DIR=pretrained_models/CosyVoice2-0.5B
WHISPER_MODEL_SIZE=base
SPEECH_DEVICE=cpu

# 服务器配置
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
LOG_LEVEL=INFO
```

### 4. 启动服务

**推荐方式（使用 start_server.py）：**

```bash
# 首次运行 - 检查环境
python start_server.py --check-only

# 开发模式（带热重载）
python start_server.py --reload

# 生产模式
python start_server.py --host 0.0.0.0 --port 8000 --workers 4
```

**或者直接启动（不推荐）：**

```bash
python main.py --reload
```

服务启动后，访问以下地址：

- **API 文档**: http://localhost:8000/docs
- **ReDoc 文档**: http://localhost:8000/redoc
- **健康检查**: http://localhost:8000/api/health

## 📚 API 接口文档

### 系统状态

- `GET /api/health` - 健康检查
- `GET /api/status` - 系统详细状态

### MongoDB 数据管理（新增）

#### 数据文档管理

- `POST /api/data/documents` - 创建数据文档
- `GET /api/data/documents` - 列出数据文档（支持分页、搜索、标签筛选）
- `GET /api/data/documents/{id}` - 获取特定数据文档
- `PUT /api/data/documents/{id}` - 更新数据文档
- `DELETE /api/data/documents/{id}` - 删除数据文档
- `GET /api/data/documents/search` - 搜索数据文档
- `GET /api/data/statistics` - 获取数据统计信息

#### 数据项管理

- `POST /api/data/documents/{id}/items` - 向文档添加数据项
- `PUT /api/data/documents/{id}/items/{sequence}` - 更新数据项
- `DELETE /api/data/documents/{id}/items/{sequence}` - 删除数据项

#### 图片管理

- `POST /api/data/upload-image` - 上传图片并转换为base64

### MCP 工具管理

- `GET /api/mcp/configs` - 获取所有 MCP 配置
- `POST /api/mcp/configs` - 创建 MCP 配置
- `PUT /api/mcp/configs/{id}` - 更新 MCP 配置
- `DELETE /api/mcp/configs/{id}` - 删除 MCP 配置
- `POST /api/mcp/configs/{id}/test` - 测试 MCP 配置

### LLM 模型管理

- `GET /api/llm/configs` - 获取所有 LLM 配置
- `POST /api/llm/configs` - 创建 LLM 配置
- `PUT /api/llm/configs/{id}` - 更新 LLM 配置
- `DELETE /api/llm/configs/{id}` - 删除 LLM 配置
- `POST /api/llm/configs/{id}/test` - 测试 LLM 配置

### 语音处理

- `POST /api/speech/recognize` - 语音识别（支持 SensVoice、Whisper）
- `POST /api/speech/synthesize` - 语音合成（支持 CosyVoice、Edge TTS）

#### CosyVoice 音色一致性功能

新增speaker ID缓存机制，确保语音合成音色一致性：

```bash
# 使用指定speaker ID进行合成，保持音色一致
curl -X POST "http://localhost:8000/api/speech/synthesize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "要合成的文本",
    "synthesis_mode": "zero_shot",
    "speaker_id": "my_custom_voice"
  }'

# 使用新的参考音频自动创建speaker
curl -X POST "http://localhost:8000/api/speech/synthesize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "要合成的文本",
    "reference_audio": "path/to/reference.wav",
    "reference_text": "参考音频对应的文本"
  }'
```

详细文档请参阅：[CosyVoice Speaker Management](docs/CosyVoice_Speaker_Management.md)

### 对话功能

- `POST /api/chat/text` - 文本对话
- `POST /api/chat/stream` - 流式对话
- `POST /api/chat/voice` - 语音对话（语音输入 + 文本和语音输出）

## 🗄️ MongoDB 数据管理使用示例

### 创建数据文档

```bash
curl -X POST "http://localhost:8000/api/data/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "产品介绍文档",
    "description": "包含产品图片和描述的文档",
    "data_list": [
      {
        "sequence": 1,
        "text": "这是我们的新产品介绍",
        "image": null,
        "image_filename": null,
        "image_mimetype": null
      },
      {
        "sequence": 2,
        "text": "产品特性说明",
        "image": "base64_encoded_image_data_here",
        "image_filename": "product.jpg",
        "image_mimetype": "image/jpeg"
      }
    ],
    "tags": ["产品", "介绍", "营销"],
    "metadata": {
      "category": "产品文档",
      "version": "1.0"
    }
  }'
```

### 上传图片

```bash
curl -X POST "http://localhost:8000/api/data/upload-image" \
  -F "image=@/path/to/your/image.jpg"
```

### 搜索文档

```bash
curl -X GET "http://localhost:8000/api/data/documents/search?q=产品&limit=10"
```

### 获取统计信息

```bash
curl -X GET "http://localhost:8000/api/data/statistics"
```

### 数据模型结构

```json
{
  "id": "文档ID",
  "name": "数据名称",
  "description": "数据描述",
  "data_list": [
    {
      "sequence": 1,
      "text": "文字内容",
      "image": "base64编码的图片数据",
      "image_filename": "图片文件名",
      "image_mimetype": "图片MIME类型"
    }
  ],
  "tags": ["标签1", "标签2"],
  "metadata": {
    "自定义字段": "值"
  },
  "created_at": "创建时间",
  "updated_at": "更新时间",
  "version": "版本号"
}
```

## 🧪 测试系统

### 运行所有测试

```bash
python test_runner.py --test-type all
```

### 运行特定类型测试

```bash
# 功能测试
python test_runner.py --test-type integration

# 性能测试
python test_runner.py --test-type performance --concurrent-users 20

# 生成测试报告
python test_runner.py --output test_report.json
```

## 📋 配置管理

### MongoDB数据文档配置示例

```json
{
  "name": "客户信息文档",
  "description": "包含客户基本信息和照片",
  "data_list": [
    {
      "sequence": 1,
      "text": "客户姓名：张三",
      "image": null
    },
    {
      "sequence": 2,
      "text": "客户照片",
      "image": "base64_encoded_photo_data",
      "image_filename": "customer_photo.jpg",
      "image_mimetype": "image/jpeg"
    }
  ],
  "tags": ["客户", "CRM", "重要"],
  "metadata": {
    "department": "销售部",
    "priority": "high"
  }
}
```

### MCP 工具配置示例

```json
{
  "name": "数学计算工具",
  "description": "提供基本的数学运算功能",
  "command": "python",
  "args": ["tools/math_server.py"],
  "transport": "stdio",
  "auto_start": false,
  "restart_on_failure": true,
  "timeout": 30
}
```

### LLM 模型配置示例

```json
{
  "name": "千问大模型",
  "provider": "dashscope",
  "model_name": "qwen-plus",
  "api_key": "your_api_key",
  "temperature": 0.7,
  "max_tokens": 2000,
  "is_default": true,
  "enabled": true
}
```

## 📊 日志系统

### 日志级别

- **DEBUG**: 调试信息
- **INFO**: 一般信息
- **WARNING**: 警告信息
- **ERROR**: 错误信息
- **CRITICAL**: 严重错误

### 日志格式

```json
{
  "timestamp": "2025-06-11T06:43:48.000Z",
  "level": "INFO",
  "logger": "main",
  "message": "🚀 AI Agent Backend 正在启动...",
  "request_id": "uuid-string",
  "extra": {
    "api_call": {
      "method": "POST",
      "endpoint": "/api/data/documents",
      "status_code": 200,
      "processing_time": 1.23
    }
  }
}
```

### 查看日志

```bash
# 实时查看应用日志
tail -f logs/ai_agent.log

# 查看错误日志
tail -f logs/ai_agent_error.log

# 使用 jq 格式化查看
tail -f logs/ai_agent.log | jq '.'
```

## 🔧 开发指南

### 环境检查

在开发前，建议运行环境检查：

```bash
python start_server.py --check-only
```

### 模块化架构说明

项目采用模块化设计，API接口按功能分离：

- **api/core_api.py**: 核心系统接口（健康检查、状态、对话）
- **api/mcp_api.py**: MCP工具配置管理接口
- **api/llm_api.py**: LLM模型配置管理接口
- **api/speech_api.py**: 语音处理相关接口
- **api/mongodb_api.py**: MongoDB数据管理接口

### 添加新的 API 接口

1. 在对应的API模块中添加新的路由处理函数
2. 在 `src/models.py` 中定义请求/响应模型
3. 添加相应的日志记录
4. 编写单元测试
5. 更新API文档

### 添加新的数据处理功能

1. 在 `src/mongodb_manager.py` 中实现数据处理逻辑
2. 在 `api/mongodb_api.py` 中添加API接口
3. 在 `src/models.py` 中定义数据模型
4. 添加相应的验证逻辑

### 集成新的 LLM 提供商

1. 在 `src/llm.py` 中添加新的 LLM 客户端类
2. 在 `src/models.py` 中更新 `LLMProvider` 枚举
3. 更新配置验证逻辑
4. 添加测试用例

### 添加新的 MCP 工具

1. 在 `tools/` 目录下编写 MCP 服务器脚本
2. 通过 API 或配置文件添加 MCP 配置
3. 测试工具功能
4. 更新文档

### 语音处理扩展

支持的语音引擎：

- **语音识别**: SensVoice (FunASR)、OpenAI Whisper
- **语音合成**: CosyVoice、Edge TTS

添加新引擎需要：

1. 在 `src/speech.py` 中实现相应的处理器类
2. 更新初始化逻辑
3. 添加配置选项

## 🐛 故障排除

### 常见问题

1. **服务启动失败**

   ```bash
   # 检查依赖
   python start_server.py --check-only
   
   # 查看详细错误
   python start_server.py --log-level DEBUG
   ```

2. **MongoDB连接失败**

   ```bash
   # 检查MongoDB服务状态
   sudo systemctl status mongod
   
   # 启动MongoDB服务
   sudo systemctl start mongod
   
   # 检查连接配置
   echo $MONGODB_HOST $MONGODB_PORT
   ```

3. **模型调用失败**
   - 检查 API 密钥配置：`cat .env | grep API_KEY`
   - 验证网络连接
   - 查看模型配置：`curl http://localhost:8000/api/llm/configs`

4. **MCP 工具无法启动**
   - 检查工具脚本权限：`ls -la tools/`
   - 测试工具配置：`curl -X POST http://localhost:8000/api/mcp/configs/{id}/test`
   - 查看 MCP 日志

5. **语音处理失败**
   - 检查模型文件路径
   - 验证音频格式支持
   - 查看语音处理器状态

6. **数据上传失败**
   - 检查文件大小限制（默认5MB）
   - 验证图片格式支持
   - 查看MongoDB存储空间

### 调试技巧

```bash
# 启用调试模式
python start_server.py --log-level DEBUG

# 检查系统状态
curl http://localhost:8000/api/health

# 查看详细状态
curl http://localhost:8000/api/status

# 检查MongoDB连接
curl http://localhost:8000/api/data/statistics

# 测试特定功能
python test_runner.py --test-type integration
```

## 📈 性能优化

### 推荐配置

**开发环境**:

```bash
python start_server.py --reload --log-level DEBUG
```

**生产环境**:

```bash
python start_server.py --workers 4 --log-level INFO
```

**容器化部署**:

```dockerfile
FROM python:3.9-slim

# 安装MongoDB客户端
RUN apt-get update && apt-get install -y mongodb-clients

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

CMD ["python", "start_server.py", "--host", "0.0.0.0", "--workers", "4"]
```

### 监控指标

- API 响应时间
- 错误率和成功率
- 并发用户数
- 内存使用率
- CPU 使用率
- 语音处理延迟
- MCP 工具状态
- MongoDB 连接状态
- 数据存储使用量

## 🚀 部署指南

### Docker 部署

```bash
# 构建镜像
docker build -t ai-agent-backend .

# 运行容器（包含MongoDB）
docker-compose up -d

# 或单独运行
docker run -p 8000:8000 -v $(pwd)/.env:/app/.env ai-agent-backend
```

### 系统服务部署

```bash
# 创建系统服务
sudo systemctl edit --force --full ai-agent-backend.service

# 启动服务
sudo systemctl enable ai-agent-backend
sudo systemctl start ai-agent-backend
```

### MongoDB集群部署

对于生产环境，建议配置MongoDB副本集：

```bash
# 配置副本集
mongo --eval "rs.initiate()"

# 添加副本节点
mongo --eval "rs.add('hostname:27017')"
```

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'Add amazing feature'`
4. 推送到分支：`git push origin feature/amazing-feature`
5. 创建 Pull Request

### 代码规范

- 使用 Black 进行代码格式化
- 遵循 PEP 8 规范
- 添加适当的类型注解
- 编写完整的文档字符串
- API接口按功能模块分离
- 数据模型定义完整的验证规则

### 提交规范

- feat: 新功能
- fix: 修复问题
- docs: 文档更新
- style: 代码格式调整
- refactor: 代码重构
- test: 测试相关
- chore: 构建过程或辅助工具变动

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🆘 支持

如有问题，请：

1. 查看文档和 FAQ
2. 运行环境检查：`python start_server.py --check-only`
3. 搜索现有的 Issues
4. 创建新的 Issue 并提供详细信息
5. 联系维护者：[@09okjk](https://github.com/09okjk)

## 📝 更新日志

### v2.1.0 (2025-06-11)

- 🎉 **重大更新**: 添加MongoDB数据管理功能
- 🔧 **架构重构**: API接口模块化分离
- 📄 **数据管理**: 支持文字和图片混合数据的完整CRUD操作
- 🖼️ **图片处理**: 支持图片上传和base64编码存储
- 🔍 **搜索功能**: 文档全文搜索和标签筛选
- 📊 **统计功能**: 数据使用情况统计和分析
- 🏗️ **模块分离**:
  - `api/core_api.py` - 核心系统接口
  - `api/mcp_api.py` - MCP工具管理
  - `api/llm_api.py` - LLM模型管理
  - `api/speech_api.py` - 语音处理
  - `api/mongodb_api.py` - 数据管理
- 🔧 **数据管理器**: 新增 `src/mongodb_manager.py`
- 📝 **模型扩展**: 扩展数据模型支持MongoDB操作
- 🚀 **向后兼容**: 保持与 `start_server.py` 的完全兼容

### v2.0.0 (2025-06-05)

- 🎉 重构为纯后端架构
- ✨ 添加 MCP 工具系统支持
- 🎤 集成语音识别和合成功能
- 📊 实现结构化日志记录
- 🧪 完善测试框架
- 🔧 优化启动脚本和环境检查
- 📚 完善 API 文档

---

**Made with ❤️ by [09okjk](https://github.com/09okjk)**

## 🔄 本次更新重点

1. **MongoDB数据管理** - 完整的数据库支持，包含文字和图片混合数据
2. **模块化重构** - API接口按功能分离，提高代码可维护性
3. **图片处理** - 支持图片上传、base64编码和数据库存储
4. **数据搜索** - 全文搜索和标签筛选功能
5. **统计分析** - 数据使用情况的详细统计
6. **向后兼容** - 保持与现有启动方式的完全兼容
7. **环境配置** - 新增MongoDB相关的环境变量配置
8. **部署支持** - 提供MongoDB安装脚本和配置指南

这次更新将AI Agent后端服务扩展为一个功能完整的数据管理平台，同时保持了原有的所有功能特性。
